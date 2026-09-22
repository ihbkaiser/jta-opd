from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch

from b200_experiment.one_step_kl_probe import conditional_reverse_kl
from b200_experiment.scoring import BaseScores, score_original_rollout


@dataclass(frozen=True)
class FiniteSupportKLReference:
    rollout: Any
    candidate_ids: torch.Tensor
    teacher_candidate_log_probs: torch.Tensor
    valid_mask: torch.Tensor
    prefix_hash: str
    global_valid_token_count: int


class FiniteSupportKLEvaluator:
    """Evaluate reverse KL on the student's frozen pre-update Top-K support."""

    def __init__(
        self,
        distributed,
        *,
        top_k: int = 16,
        temperature: float = 1.0,
        score_micro_batch_size: int = 1,
        score_chunk_steps: int = 128,
        score_fn: Callable[..., BaseScores] = score_original_rollout,
    ) -> None:
        if int(top_k) <= 0:
            raise ValueError("Held-out KL top_k must be positive")
        if float(temperature) <= 0:
            raise ValueError("Held-out KL temperature must be positive")
        self.distributed = distributed
        self.top_k = int(top_k)
        self.temperature = float(temperature)
        self.score_micro_batch_size = int(score_micro_batch_size)
        self.score_chunk_steps = int(score_chunk_steps)
        self.score_fn = score_fn

    def _score(
        self,
        model,
        rollout,
        *,
        top_k: int = 0,
        candidate_ids: torch.Tensor | None = None,
    ) -> BaseScores:
        was_training = bool(model.training)
        try:
            return self.score_fn(
                model,
                rollout,
                keep_cache=False,
                score_chunk_steps=self.score_chunk_steps,
                retain_response_logits=False,
                top_k=top_k,
                candidate_ids=candidate_ids,
                temperature=self.temperature,
                micro_batch_size=self.score_micro_batch_size,
                trim_padding=True,
                length_bucketed=True,
            )
        finally:
            model.train(was_training)

    def _global_mean(
        self,
        student_log_probs: torch.Tensor,
        teacher_log_probs: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[float, int]:
        local_total, local_count = conditional_reverse_kl(
            student_log_probs, teacher_log_probs, valid_mask
        )
        global_total = self.distributed.sum_float(float(local_total.item()))
        global_count = self.distributed.sum_int(local_count)
        if global_count <= 0:
            raise ValueError("Held-out KL contains no globally valid prefix tokens")
        return global_total / global_count, global_count

    def build_reference(
        self,
        student,
        teacher,
        rollout,
        *,
        prefix_hash: str,
    ) -> tuple[FiniteSupportKLReference, float]:
        student_scores = self._score(student, rollout, top_k=self.top_k)
        if (
            student_scores.top_k_ids is None
            or student_scores.top_k_log_probs is None
        ):
            raise RuntimeError("Student scorer did not return required Top-K outputs")
        candidate_ids = student_scores.top_k_ids.detach().clone()
        teacher_scores = self._score(
            teacher, rollout, candidate_ids=candidate_ids
        )
        if teacher_scores.candidate_log_probs is None:
            raise RuntimeError(
                "Teacher scorer did not return candidate log-probabilities"
            )
        teacher_log_probs = teacher_scores.candidate_log_probs.detach().clone()
        valid_mask = rollout.valid_mask.detach().clone().bool()
        before, global_count = self._global_mean(
            student_scores.top_k_log_probs,
            teacher_log_probs,
            valid_mask,
        )
        return (
            FiniteSupportKLReference(
                rollout=rollout,
                candidate_ids=candidate_ids,
                teacher_candidate_log_probs=teacher_log_probs,
                valid_mask=valid_mask,
                prefix_hash=str(prefix_hash),
                global_valid_token_count=global_count,
            ),
            before,
        )

    def score_student(
        self, student, reference: FiniteSupportKLReference
    ) -> float:
        scores = self._score(
            student,
            reference.rollout,
            candidate_ids=reference.candidate_ids,
        )
        if scores.candidate_log_probs is None:
            raise RuntimeError(
                "Student scorer did not return candidate log-probabilities"
            )
        value, global_count = self._global_mean(
            scores.candidate_log_probs,
            reference.teacher_candidate_log_probs,
            reference.valid_mask,
        )
        if global_count != reference.global_valid_token_count:
            raise RuntimeError(
                "Held-out KL valid-token population changed after the update"
            )
        return value
