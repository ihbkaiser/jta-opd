from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch

from b200_experiment.distributed import contiguous_partition
from b200_experiment.evaluation import load_benchmark, render_evaluation_prompt
from b200_experiment.heldout_probe import HeldoutRootStore, select_heldout_records
from b200_experiment.one_step_kl_probe import OneStepKLProbeConfig
from b200_experiment.probe_logging import OneStepKLProbeLogger
from b200_experiment.scoring import generate_on_policy
from b200_experiment.trajectory_kl import (
    FullVocabularyTrajectoryKLEvaluator,
    TrajectoryKLEvaluation,
)
from b200_experiment.trajectory_probe import (
    OnPolicyTrajectoryBuilder,
    TrajectoryProbeBatch,
)


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class PreparedOneStepProbe:
    optimizer_step: int
    rollout_id: int
    ppo_group_index: int
    started_at: float


@dataclass(frozen=True)
class BranchTrajectoryEvaluation:
    batch: TrajectoryProbeBatch
    evaluation: TrajectoryKLEvaluation


class OneStepKLProbeRuntime:
    """Run paired branch-specific on-policy trajectory KL diagnostics."""

    def __init__(
        self,
        config: OneStepKLProbeConfig,
        *,
        evaluator: FullVocabularyTrajectoryKLEvaluator,
        builder: OnPolicyTrajectoryBuilder,
        logger: OneStepKLProbeLogger,
        teacher,
        run_id: str,
        distributed,
        benchmark: str,
        heldout_example_hash: str,
        heldout_root_hash: str,
    ) -> None:
        self.config = config
        self.evaluator = evaluator
        self.builder = builder
        self.logger = logger
        self.teacher = teacher
        self.run_id = str(run_id)
        self.distributed = distributed
        self.benchmark = str(benchmark)
        self.heldout_example_hash = str(heldout_example_hash)
        self.heldout_root_hash = str(heldout_root_hash)
        self.failure_policy = config.failure_policy

    def should_probe(self, optimizer_step: int) -> bool:
        return self.config.should_probe(optimizer_step)

    def prepare_before_step(
        self,
        _student,
        optimizer_step: int,
        *,
        rollout_id: int,
        ppo_group_index: int,
        **_ignored: Any,
    ) -> PreparedOneStepProbe:
        # Fixed roots are materialized at setup. Generation intentionally happens
        # after each branch update so each branch visits its own future states.
        return PreparedOneStepProbe(
            optimizer_step=int(optimizer_step),
            rollout_id=int(rollout_id),
            ppo_group_index=int(ppo_group_index),
            started_at=time.perf_counter(),
        )

    def score_student(
        self, student, prepared: PreparedOneStepProbe
    ) -> BranchTrajectoryEvaluation:
        batch = self.builder.build(student, optimizer_step=prepared.optimizer_step)
        evaluation = self.evaluator.evaluate(
            student,
            self.teacher,
            batch.rollout,
            problem_ids=batch.problem_ids,
        )
        return BranchTrajectoryEvaluation(batch=batch, evaluation=evaluation)

    @staticmethod
    def _check_prepared(
        prepared: PreparedOneStepProbe,
        *,
        optimizer_step: int,
        rollout_id: int,
        ppo_group_index: int,
    ) -> None:
        if int(optimizer_step) != prepared.optimizer_step:
            raise ValueError("Prepared probe optimizer step does not match write step")
        if int(rollout_id) != prepared.rollout_id:
            raise ValueError("Prepared probe rollout does not match write rollout")
        if int(ppo_group_index) != prepared.ppo_group_index:
            raise ValueError("Prepared probe PPO group does not match write group")

    def _local_problem_rows(
        self,
        optimizer_step: int,
        uniform: BranchTrajectoryEvaluation,
        cmt: BranchTrajectoryEvaluation,
    ) -> list[dict[str, Any]]:
        u = uniform.evaluation
        c = cmt.evaluation
        if u.problem_ids != c.problem_ids:
            raise ValueError("Uniform and CMT trajectory problem IDs do not align")
        rows: list[dict[str, Any]] = []
        for index, problem_id in enumerate(u.problem_ids):
            uniform_kl = float(u.per_problem_kl[index].item())
            cmt_kl = float(c.per_problem_kl[index].item())
            rows.append(
                {
                    "run_id": self.run_id,
                    "optimizer_step": int(optimizer_step),
                    "problem_id": str(problem_id),
                    "uniform_trajectory_kl": uniform_kl,
                    "cmt_trajectory_kl": cmt_kl,
                    "trajectory_gap": uniform_kl - cmt_kl,
                    "uniform_mean_length": float(
                        u.per_problem_mean_length[index].item()
                    ),
                    "cmt_mean_length": float(c.per_problem_mean_length[index].item()),
                    "uniform_early_eos_rate": float(
                        u.per_problem_early_eos_rate[index].item()
                    ),
                    "cmt_early_eos_rate": float(
                        c.per_problem_early_eos_rate[index].item()
                    ),
                }
            )
        return rows

    def write_pair(
        self,
        prepared: PreparedOneStepProbe,
        *,
        optimizer_step: int,
        rollout_id: int,
        ppo_group_index: int,
        kl_after_uniform: BranchTrajectoryEvaluation,
        kl_after_cmt: BranchTrajectoryEvaluation,
        uniform_update_loss: float,
        cmt_update_loss: float,
        uniform_branch_time_sec: float,
        cmt_eval_time_sec: float,
    ) -> None:
        self._check_prepared(
            prepared,
            optimizer_step=optimizer_step,
            rollout_id=rollout_id,
            ppo_group_index=ppo_group_index,
        )
        uniform = kl_after_uniform
        cmt = kl_after_cmt
        if uniform.batch.sampling_seed != cmt.batch.sampling_seed:
            raise ValueError("Uniform and CMT branches must use matched sampling seeds")
        gathered = self.distributed.all_gather_objects(
            self._local_problem_rows(optimizer_step, uniform, cmt)
        )
        problem_rows = [row for rank_rows in gathered for row in rank_rows]
        if len(problem_rows) != self.config.subset_size:
            raise ValueError(
                "Gathered trajectory problem rows do not match the held-out subset"
            )
        u = uniform.evaluation
        c = cmt.evaluation
        self.logger.upsert(
            {
                "run_id": self.run_id,
                "optimizer_step": int(optimizer_step),
                "rollout_id": int(rollout_id),
                "ppo_group_index": int(ppo_group_index),
                "benchmark": self.benchmark,
                "subset_size": self.config.subset_size,
                "num_rollouts_per_problem": self.config.num_rollouts_per_problem,
                "horizon": self.config.horizon,
                "sampling_temperature": self.config.sampling_temperature,
                "sampling_top_p": self.config.sampling_top_p,
                "sampling_seed": int(uniform.batch.sampling_seed),
                "heldout_example_hash": self.heldout_example_hash,
                "heldout_root_hash": self.heldout_root_hash,
                "uniform_trajectory_hash": uniform.batch.trajectory_hash,
                "cmt_trajectory_hash": cmt.batch.trajectory_hash,
                "metric": self.config.metric,
                "uniform_trajectory_kl": float(u.mean),
                "cmt_trajectory_kl": float(c.mean),
                "trajectory_gap": float(u.mean - c.mean),
                "uniform_valid_state_count": int(u.valid_state_count),
                "cmt_valid_state_count": int(c.valid_state_count),
                "uniform_mean_length": float(u.mean_length),
                "cmt_mean_length": float(c.mean_length),
                "uniform_early_eos_rate": float(u.early_eos_rate),
                "cmt_early_eos_rate": float(c.early_eos_rate),
                "uniform_update_loss": float(uniform_update_loss),
                "cmt_update_loss": float(cmt_update_loss),
                "probe_time_sec": float(time.perf_counter() - prepared.started_at),
                "uniform_branch_time_sec": float(uniform_branch_time_sec),
                "cmt_branch_time_sec": float(cmt_eval_time_sec),
            },
            problem_rows,
        )


def _tokenize_prompts(tokenizer, prompts: list[str], device: torch.device):
    original_padding_side = getattr(tokenizer, "padding_side", None)
    tokenizer.padding_side = "left"
    try:
        encoded = tokenizer(
            prompts,
            padding=True,
            truncation=False,
            return_tensors="pt",
            add_special_tokens=False,
        )
    finally:
        if original_padding_side is not None:
            tokenizer.padding_side = original_padding_side
    return encoded["input_ids"].to(device), encoded["attention_mask"].to(device)


def prepare_one_step_kl_probe(
    config: dict[str, Any],
    *,
    method: str,
    student,
    teacher,
    tokenizer,
    rollout_engine,
    output_dir: str | Path,
    distributed,
    device: torch.device,
    resume: bool,
    benchmark_loader: Callable = load_benchmark,
) -> tuple[OneStepKLProbeRuntime | None, dict[str, Any] | None]:
    settings = OneStepKLProbeConfig.from_mapping(
        config.get("one_step_kl_probe"), method=method
    )
    if not settings.enabled:
        return None, None

    specifications = config.get("evaluation", {}).get("benchmarks", {})
    if settings.benchmark not in specifications:
        raise ValueError(
            f"Trajectory probe benchmark is not configured: {settings.benchmark}"
        )
    if settings.subset_size % int(distributed.world_size) != 0:
        raise ValueError(
            "one_step_kl_probe.subset_size must be divisible by world_size"
        )
    records, benchmark_schema = benchmark_loader(
        settings.benchmark, specifications[settings.benchmark]
    )
    selected = select_heldout_records(records, settings.subset_size, settings.seed)
    selected_ids = [str(row["id"]) for row in selected]
    example_hash = _canonical_hash(
        [
            {
                "id": str(row["id"]),
                "problem": str(row["problem"]),
                "answer": str(row.get("answer", "")),
            }
            for row in selected
        ]
    )
    manifest_fields = {
        "benchmark": settings.benchmark,
        "benchmark_file": str(benchmark_schema.get("file", "")),
        "subset_size": settings.subset_size,
        "seed": settings.seed,
        "selected_ids": selected_ids,
        "heldout_example_hash": example_hash,
        "student_model": str(config.get("models", {}).get("student_path", "")),
        "teacher_model": str(config.get("models", {}).get("teacher_path", "")),
        "tokenizer": str(getattr(tokenizer, "name_or_path", "")),
        "num_rollouts_per_problem": settings.num_rollouts_per_problem,
        "horizon": settings.horizon,
        "sampling_temperature": settings.sampling_temperature,
        "sampling_top_p": settings.sampling_top_p,
    }
    store = HeldoutRootStore(Path(output_dir) / settings.artifact_subdir, distributed)
    if resume:
        roots = store.load(device, expected=manifest_fields)
    else:
        if store.manifest_path.exists():
            raise FileExistsError(
                "Trajectory probe roots already exist; resume or choose a new output"
            )
        start, end = contiguous_partition(
            settings.subset_size, distributed.rank, distributed.world_size
        )
        local_records = selected[start:end]
        prompts = [
            render_evaluation_prompt(tokenizer, row, config) for row in local_records
        ]
        prompt_ids, attention_mask = _tokenize_prompts(tokenizer, prompts, device)
        store.save(
            prompt_ids,
            attention_mask,
            [str(row["id"]) for row in local_records],
            manifest_fields,
        )
        roots = store.load(device, expected=manifest_fields)

    generate_fn = (
        rollout_engine.generate if rollout_engine is not None else generate_on_policy
    )
    builder = OnPolicyTrajectoryBuilder(
        distributed,
        roots=roots,
        num_rollouts_per_problem=settings.num_rollouts_per_problem,
        horizon=settings.horizon,
        seed=settings.seed,
        temperature=settings.sampling_temperature,
        top_p=settings.sampling_top_p,
        eos_token_id=int(tokenizer.eos_token_id),
        pad_token_id=int(tokenizer.pad_token_id),
        generation_batch_size=settings.generation_batch_size,
        generate_fn=generate_fn,
    )
    evaluator = FullVocabularyTrajectoryKLEvaluator(
        distributed,
        horizon=settings.horizon,
        num_rollouts_per_problem=settings.num_rollouts_per_problem,
        temperature=1.0,
        score_micro_batch_size=settings.score_micro_batch_size,
        score_chunk_steps=int(config.get("selector", {}).get("score_chunk_steps", 128)),
    )
    logger = OneStepKLProbeLogger(
        output_dir,
        enabled=True,
        is_main=distributed.is_main,
        artifact_subdir=settings.artifact_subdir,
    )
    runtime = OneStepKLProbeRuntime(
        settings,
        evaluator=evaluator,
        builder=builder,
        logger=logger,
        teacher=teacher,
        run_id=str(config.get("experiment", {}).get("run_id", Path(output_dir).name)),
        distributed=distributed,
        benchmark=settings.benchmark,
        heldout_example_hash=example_hash,
        heldout_root_hash=roots.root_hash,
    )
    metadata = {
        "enabled": True,
        "state_source": settings.state_source,
        "benchmark": settings.benchmark,
        "subset_size": settings.subset_size,
        "num_rollouts_per_problem": settings.num_rollouts_per_problem,
        "horizon": settings.horizon,
        "sampling_temperature": settings.sampling_temperature,
        "sampling_top_p": settings.sampling_top_p,
        "seed": settings.seed,
        "interval_steps": settings.interval_steps,
        "metric": settings.metric,
        "failure_policy": settings.failure_policy,
        "artifact_subdir": settings.artifact_subdir,
        "heldout_example_hash": example_hash,
        "heldout_root_hash": roots.root_hash,
    }
    return runtime, metadata
