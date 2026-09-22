from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from b200_experiment.heldout_kl import FiniteSupportKLEvaluator, KLEvaluation
from b200_experiment.one_step_kl_probe import (
    OneStepKLProbeConfig,
    paired_sample_statistics,
)
from b200_experiment.probe_logging import OneStepKLProbeLogger
from b200_experiment.successor_probe import SuccessorProbeBatch, SuccessorStateBuilder


@dataclass(frozen=True)
class PreparedOneStepProbe:
    reference: Any
    kl_before: float
    optimizer_step: int
    rollout_id: int
    ppo_group_index: int
    state_age_steps: int
    successor_batch: SuccessorProbeBatch
    started_at: float


class OneStepKLProbeRuntime:
    def __init__(
        self,
        config: OneStepKLProbeConfig,
        *,
        evaluator: FiniteSupportKLEvaluator,
        builder: SuccessorStateBuilder,
        logger: OneStepKLProbeLogger,
        teacher,
        run_id: str,
        distributed,
    ) -> None:
        self.config = config
        self.evaluator = evaluator
        self.builder = builder
        self.logger = logger
        self.teacher = teacher
        self.run_id = str(run_id)
        self.distributed = distributed
        self.failure_policy = config.failure_policy

    def should_probe(self, optimizer_step: int) -> bool:
        return self.config.should_probe(optimizer_step)

    def prepare_before_step(
        self,
        student,
        optimizer_step: int,
        *,
        rollout,
        objective_valid_mask,
        rollout_id: int,
        ppo_group_index: int,
        state_age_steps: int,
    ) -> PreparedOneStepProbe:
        started = time.perf_counter()
        successor_batch = self.builder.build(
            student,
            rollout,
            objective_valid_mask=objective_valid_mask,
            optimizer_step=optimizer_step,
        )
        reference, kl_before = self.evaluator.build_reference(
            student,
            self.teacher,
            successor_batch.rollout,
            prefix_hash=successor_batch.state_hash,
        )
        return PreparedOneStepProbe(
            reference=reference,
            kl_before=float(kl_before),
            optimizer_step=int(optimizer_step),
            rollout_id=int(rollout_id),
            ppo_group_index=int(ppo_group_index),
            state_age_steps=int(state_age_steps),
            successor_batch=successor_batch,
            started_at=started,
        )

    def score_student(
        self, student, prepared: PreparedOneStepProbe
    ) -> KLEvaluation:
        return self.evaluator.score_student_evaluation(student, prepared.reference)

    def write_pair(
        self,
        prepared: PreparedOneStepProbe,
        *,
        optimizer_step: int,
        rollout_id: int,
        ppo_group_index: int,
        kl_after_uniform: KLEvaluation,
        kl_after_cmt: KLEvaluation,
        uniform_update_loss: float,
        cmt_update_loss: float,
        uniform_branch_time_sec: float,
        cmt_eval_time_sec: float,
    ) -> None:
        if int(optimizer_step) != prepared.optimizer_step:
            raise ValueError("Prepared probe optimizer step does not match write step")
        if int(rollout_id) != prepared.rollout_id:
            raise ValueError("Prepared probe rollout does not match write rollout")
        if int(ppo_group_index) != prepared.ppo_group_index:
            raise ValueError("Prepared probe PPO group does not match write group")
        statistics = paired_sample_statistics(
            prepared.reference.pre_kl_values,
            kl_after_uniform.local_values,
            kl_after_cmt.local_values,
            self.distributed,
        )
        batch = prepared.successor_batch
        self.logger.upsert(
            {
                "run_id": self.run_id,
                "optimizer_step": int(optimizer_step),
                "rollout_id": int(rollout_id),
                "ppo_group_index": int(ppo_group_index),
                "state_age_steps": prepared.state_age_steps,
                "state_source": self.config.state_source,
                "parent_state_count": batch.parent_state_count,
                "successors_per_parent": batch.successors_per_parent,
                "successor_state_count": batch.successor_state_count,
                "sampling_distribution": "student_full_vocab",
                "sampling_seed": batch.sampling_seed,
                "successor_state_hash": batch.state_hash,
                "valid_successor_state_count": int(
                    prepared.reference.global_valid_token_count
                ),
                "top_k": self.config.top_k,
                "metric": self.config.metric,
                "kl_before": prepared.kl_before,
                "kl_after_uniform": float(kl_after_uniform.mean),
                "kl_after_cmt": float(kl_after_cmt.mean),
                **statistics,
                "uniform_update_loss": float(uniform_update_loss),
                "cmt_update_loss": float(cmt_update_loss),
                "probe_time_sec": float(time.perf_counter() - prepared.started_at),
                "uniform_branch_time_sec": float(uniform_branch_time_sec),
                "cmt_eval_time_sec": float(cmt_eval_time_sec),
            }
        )


def prepare_one_step_kl_probe(
    config: dict[str, Any],
    *,
    method: str,
    teacher,
    tokenizer,
    output_dir: str | Path,
    distributed,
) -> tuple[OneStepKLProbeRuntime | None, dict[str, Any] | None]:
    settings = OneStepKLProbeConfig.from_mapping(
        config.get("one_step_kl_probe"), method=method
    )
    if not settings.enabled:
        return None, None
    score_chunk_steps = int(config.get("selector", {}).get("score_chunk_steps", 128))
    evaluator = FiniteSupportKLEvaluator(
        distributed,
        top_k=settings.top_k,
        temperature=1.0,
        score_micro_batch_size=settings.score_micro_batch_size,
        score_chunk_steps=score_chunk_steps,
    )
    builder = SuccessorStateBuilder(
        distributed,
        parent_state_count=settings.parent_state_count,
        successors_per_parent=settings.successors_per_parent,
        seed=settings.seed,
        sampling_temperature=settings.sampling_temperature,
        pad_token_id=int(tokenizer.pad_token_id),
        score_micro_batch_size=settings.score_micro_batch_size,
        score_chunk_steps=score_chunk_steps,
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
    )
    metadata = {
        "enabled": True,
        "state_source": settings.state_source,
        "parent_state_count": settings.parent_state_count,
        "successors_per_parent": settings.successors_per_parent,
        "sampling_distribution": "student_full_vocab",
        "sampling_temperature": settings.sampling_temperature,
        "seed": settings.seed,
        "interval_steps": settings.interval_steps,
        "top_k": settings.top_k,
        "metric": settings.metric,
        "failure_policy": settings.failure_policy,
        "artifact_subdir": settings.artifact_subdir,
    }
    return runtime, metadata
