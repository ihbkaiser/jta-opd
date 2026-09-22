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
from b200_experiment.heldout_kl import FiniteSupportKLEvaluator
from b200_experiment.heldout_probe import (
    HeldoutPrefixStore,
    select_heldout_records,
)
from b200_experiment.one_step_kl_probe import (
    OneStepKLProbeConfig,
    paired_improvement,
)
from b200_experiment.probe_logging import OneStepKLProbeLogger
from b200_experiment.scoring import RolloutBatch, generate_on_policy


@dataclass(frozen=True)
class PreparedOneStepProbe:
    reference: Any
    kl_before: float
    optimizer_step: int
    started_at: float


class OneStepKLProbeRuntime:
    def __init__(
        self,
        config: OneStepKLProbeConfig,
        *,
        evaluator: FiniteSupportKLEvaluator,
        logger: OneStepKLProbeLogger,
        teacher,
        rollout: RolloutBatch,
        run_id: str,
        benchmark: str,
        subset_size: int,
        heldout_example_hash: str,
        heldout_prefix_hash: str,
    ) -> None:
        self.config = config
        self.evaluator = evaluator
        self.logger = logger
        self.teacher = teacher
        self.rollout = rollout
        self.run_id = str(run_id)
        self.benchmark = str(benchmark)
        self.subset_size = int(subset_size)
        self.heldout_example_hash = str(heldout_example_hash)
        self.heldout_prefix_hash = str(heldout_prefix_hash)
        self.failure_policy = config.failure_policy

    def should_probe(self, optimizer_step: int) -> bool:
        return self.config.should_probe(optimizer_step)

    def prepare_before_step(self, student, optimizer_step: int) -> PreparedOneStepProbe:
        started = time.perf_counter()
        reference, kl_before = self.evaluator.build_reference(
            student,
            self.teacher,
            self.rollout,
            prefix_hash=self.heldout_prefix_hash,
        )
        return PreparedOneStepProbe(
            reference=reference,
            kl_before=float(kl_before),
            optimizer_step=int(optimizer_step),
            started_at=started,
        )

    def score_student(self, student, prepared: PreparedOneStepProbe) -> float:
        return float(self.evaluator.score_student(student, prepared.reference))

    def write_pair(
        self,
        prepared: PreparedOneStepProbe,
        *,
        optimizer_step: int,
        rollout_id: int,
        ppo_group_index: int,
        kl_after_uniform: float,
        kl_after_cmt: float,
        uniform_update_loss: float,
        cmt_update_loss: float,
        uniform_branch_time_sec: float,
        cmt_eval_time_sec: float,
    ) -> None:
        if int(optimizer_step) != prepared.optimizer_step:
            raise ValueError("Prepared probe optimizer step does not match write step")
        improvements = paired_improvement(
            prepared.kl_before, kl_after_uniform, kl_after_cmt
        )
        self.logger.upsert(
            {
                "run_id": self.run_id,
                "optimizer_step": int(optimizer_step),
                "rollout_id": int(rollout_id),
                "ppo_group_index": int(ppo_group_index),
                "benchmark": self.benchmark,
                "subset_size": self.subset_size,
                "heldout_example_hash": self.heldout_example_hash,
                "heldout_prefix_hash": self.heldout_prefix_hash,
                "valid_prefix_token_count": int(
                    prepared.reference.global_valid_token_count
                ),
                "top_k": self.config.top_k,
                "metric": self.config.metric,
                "kl_before": prepared.kl_before,
                "kl_after_uniform": float(kl_after_uniform),
                "kl_after_cmt": float(kl_after_cmt),
                **improvements,
                "uniform_update_loss": float(uniform_update_loss),
                "cmt_update_loss": float(cmt_update_loss),
                "probe_time_sec": float(time.perf_counter() - prepared.started_at),
                "uniform_branch_time_sec": float(uniform_branch_time_sec),
                "cmt_eval_time_sec": float(cmt_eval_time_sec),
            }
        )


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _tokenize_evaluation_prompts(tokenizer, prompts: list[str], device) -> dict:
    previous_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        encoded = tokenizer(
            prompts,
            add_special_tokens=False,
            padding=True,
            truncation=False,
            return_tensors="pt",
        )
    finally:
        tokenizer.padding_side = previous_side
    return {key: value.to(device) for key, value in encoded.items()}


def _concatenate_rollouts(
    prompt_ids: torch.Tensor,
    prompt_attention_mask: torch.Tensor,
    rollouts: list[RolloutBatch],
    *,
    pad_token_id: int,
) -> RolloutBatch:
    if not rollouts:
        raise ValueError("Held-out prefix generation produced no rollout batches")
    maximum_response = max(item.response_ids.shape[1] for item in rollouts)
    response_ids, valid_masks, rollout_log_probs = [], [], []
    for item in rollouts:
        padding = maximum_response - item.response_ids.shape[1]
        response_ids.append(
            torch.nn.functional.pad(
                item.response_ids, (0, padding), value=int(pad_token_id)
            )
        )
        valid_masks.append(
            torch.nn.functional.pad(item.valid_mask, (0, padding), value=False)
        )
        rollout_log_probs.append(
            torch.nn.functional.pad(
                item.rollout_log_probs, (0, padding), value=float("nan")
            )
        )
    responses = torch.cat(response_ids, dim=0)
    valid = torch.cat(valid_masks, dim=0).bool()
    return RolloutBatch(
        input_ids=torch.cat((prompt_ids, responses), dim=1),
        attention_mask=torch.cat((prompt_attention_mask, valid.long()), dim=1),
        response_ids=responses,
        valid_mask=valid,
        rollout_log_probs=torch.cat(rollout_log_probs, dim=0),
        prompt_width=int(prompt_ids.shape[1]),
    )


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

    benchmark_specs = config.get("evaluation", {}).get("benchmarks", {})
    if settings.benchmark not in benchmark_specs:
        raise ValueError(
            f"one_step_kl_probe benchmark {settings.benchmark!r} is not configured"
        )
    records, schema = benchmark_loader(
        settings.benchmark, benchmark_specs[settings.benchmark]
    )
    selected = select_heldout_records(records, settings.subset_size, settings.seed)
    selected_ids = [str(row["id"]) for row in selected]
    example_hash = _canonical_hash(selected_ids)
    generation = {
        "max_new_tokens": settings.max_new_tokens,
        "temperature": settings.temperature,
        "top_p": settings.top_p,
        "seed": settings.seed,
    }
    manifest_fields = {
        "benchmark": settings.benchmark,
        "benchmark_file": str(schema.get("file", "")),
        "subset_size": settings.subset_size,
        "subset_seed": settings.seed,
        "selected_ids": selected_ids,
        "selected_example_hash": example_hash,
        "student_model": str(config["models"]["student_path"]),
        "teacher_model": str(config["models"]["teacher_path"]),
        "tokenizer": str(getattr(tokenizer, "name_or_path", "unknown")),
        "generation": generation,
    }
    store = HeldoutPrefixStore(Path(output_dir) / settings.artifact_subdir, distributed)
    if resume:
        artifact = store.load(device, expected=manifest_fields)
    else:
        if store.manifest_path.exists():
            raise FileExistsError(
                f"Refusing to replace fixed held-out prefixes: {store.manifest_path}"
            )
        begin, end = contiguous_partition(
            len(selected), distributed.rank, distributed.world_size
        )
        local_records = selected[begin:end]
        if not local_records:
            raise ValueError(
                "Held-out subset must contain at least one example per rank"
            )
        prompts = [
            render_evaluation_prompt(tokenizer, row, config) for row in local_records
        ]
        encoded = _tokenize_evaluation_prompts(tokenizer, prompts, device)
        generate = (
            rollout_engine.generate
            if rollout_engine is not None
            else generate_on_policy
        )
        generated: list[RolloutBatch] = []
        was_training = bool(student.training)
        try:
            for batch_begin in range(
                0, len(local_records), settings.generation_batch_size
            ):
                batch_end = min(
                    batch_begin + settings.generation_batch_size, len(local_records)
                )
                generated.append(
                    generate(
                        student,
                        encoded["input_ids"][batch_begin:batch_end],
                        encoded["attention_mask"][batch_begin:batch_end],
                        max_new_tokens=settings.max_new_tokens,
                        temperature=settings.temperature,
                        top_p=settings.top_p,
                        eos_token_ids=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.pad_token_id,
                        seed=settings.seed,
                        sample_seed_offset=begin + batch_begin,
                    )
                )
        finally:
            student.train(was_training)
        rollout = _concatenate_rollouts(
            encoded["input_ids"],
            encoded["attention_mask"],
            generated,
            pad_token_id=tokenizer.pad_token_id,
        )
        store.save(rollout, manifest_fields)
        artifact = store.load(device, expected=manifest_fields)

    evaluator = FiniteSupportKLEvaluator(
        distributed,
        top_k=settings.top_k,
        temperature=settings.temperature,
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
        logger=logger,
        teacher=teacher,
        rollout=artifact.rollout,
        run_id=str(config.get("experiment", {}).get("run_id", Path(output_dir).name)),
        benchmark=settings.benchmark,
        subset_size=settings.subset_size,
        heldout_example_hash=example_hash,
        heldout_prefix_hash=artifact.prefix_hash,
    )
    metadata = {
        **manifest_fields,
        "enabled": True,
        "interval_steps": settings.interval_steps,
        "top_k": settings.top_k,
        "metric": settings.metric,
        "failure_policy": settings.failure_policy,
        "artifact_subdir": settings.artifact_subdir,
        "prefix_hash": artifact.prefix_hash,
        "valid_prefix_token_count": sum(
            int(item["valid_tokens"]) for item in artifact.manifest["rank_artifacts"]
        ),
    }
    return runtime, metadata
