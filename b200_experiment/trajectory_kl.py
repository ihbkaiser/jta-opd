from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from b200_experiment.scoring import score_student_teacher_rollout


@torch.no_grad()
def full_vocab_reverse_kl_values(
    student_log_probs: torch.Tensor, teacher_log_probs: torch.Tensor
) -> torch.Tensor:
    """Return exact ``KL(student || teacher)`` for every leading position."""
    if student_log_probs.shape != teacher_log_probs.shape:
        raise ValueError("Student and teacher full-vocabulary tensors must align")
    if student_log_probs.ndim < 2:
        raise ValueError("Full-vocabulary tensors must include a vocabulary axis")
    student = student_log_probs.detach().double()
    teacher = teacher_log_probs.detach().double()
    if not bool(torch.isfinite(student).all() and torch.isfinite(teacher).all()):
        raise FloatingPointError("Full-vocabulary KL received non-finite values")
    log_p = student - torch.logsumexp(student, dim=-1, keepdim=True)
    log_q = teacher - torch.logsumexp(teacher, dim=-1, keepdim=True)
    values = (log_p.exp() * (log_p - log_q)).sum(dim=-1)
    if not bool(torch.isfinite(values).all()):
        raise FloatingPointError("Full-vocabulary KL produced non-finite values")
    return values


@dataclass(frozen=True)
class TrajectoryKLEvaluation:
    mean: float
    problem_ids: tuple[str, ...]
    per_problem_kl: torch.Tensor
    per_problem_mean_length: torch.Tensor
    per_problem_early_eos_rate: torch.Tensor
    valid_state_count: int
    trajectory_count: int
    mean_length: float
    early_eos_rate: float


class FullVocabularyTrajectoryKLEvaluator:
    """Score exact reverse KL along branch-specific on-policy trajectories."""

    def __init__(
        self,
        distributed,
        *,
        horizon: int,
        num_rollouts_per_problem: int,
        temperature: float = 1.0,
        score_micro_batch_size: int = 1,
        score_chunk_steps: int = 128,
        score_fn: Callable = score_student_teacher_rollout,
    ) -> None:
        if int(horizon) <= 0:
            raise ValueError("Trajectory KL horizon must be positive")
        if int(num_rollouts_per_problem) <= 0:
            raise ValueError("Trajectory rollout count must be positive")
        if float(temperature) <= 0.0:
            raise ValueError("Trajectory KL temperature must be positive")
        self.distributed = distributed
        self.horizon = int(horizon)
        self.num_rollouts_per_problem = int(num_rollouts_per_problem)
        self.temperature = float(temperature)
        self.score_micro_batch_size = int(score_micro_batch_size)
        self.score_chunk_steps = int(score_chunk_steps)
        self.score_fn = score_fn

    @torch.no_grad()
    def evaluate(
        self,
        student,
        teacher,
        rollout,
        *,
        problem_ids: tuple[str, ...],
    ) -> TrajectoryKLEvaluation:
        expected = len(problem_ids) * self.num_rollouts_per_problem
        if int(rollout.valid_mask.shape[0]) != expected:
            raise ValueError(
                "Trajectory KL rollout population does not match held-out roots"
            )
        student_was_training = bool(getattr(student, "training", False))
        teacher_was_training = bool(getattr(teacher, "training", False))
        try:
            student_scores, _teacher_scores = self.score_fn(
                student,
                teacher,
                rollout,
                score_chunk_steps=self.score_chunk_steps,
                top_k=1,
                student_temperature=self.temperature,
                teacher_temperature=self.temperature,
                micro_batch_size=self.score_micro_batch_size,
                trim_padding=True,
                length_bucketed=True,
                compute_full_vocab_metrics=True,
            )
        finally:
            if hasattr(student, "train"):
                student.train(student_was_training)
            if hasattr(teacher, "train"):
                teacher.train(teacher_was_training)
        log_ratio_mean = student_scores.full_log_ratio_mean
        if log_ratio_mean is None:
            raise RuntimeError("Joint scorer did not return full-vocabulary metrics")
        valid = rollout.valid_mask.bool()
        if log_ratio_mean.shape != valid.shape:
            raise ValueError("Trajectory KL values do not align with valid states")
        kl = -log_ratio_mean.detach().double()
        if not bool(torch.isfinite(kl[valid]).all()):
            raise FloatingPointError("Trajectory KL contains non-finite values")
        minimum = float(kl[valid].min().item()) if bool(valid.any()) else 0.0
        if minimum < -1e-5:
            raise FloatingPointError(
                f"Full-vocabulary reverse KL is unexpectedly negative: {minimum}"
            )
        kl = kl.clamp_min(0.0)
        per_trajectory = (kl * valid).sum(dim=-1) / float(self.horizon)
        lengths = valid.long().sum(dim=-1).double()
        early = lengths.lt(float(self.horizon)).double()
        roots = len(problem_ids)
        per_problem_kl = per_trajectory.reshape(
            roots, self.num_rollouts_per_problem
        ).mean(dim=1)
        per_problem_length = lengths.reshape(
            roots, self.num_rollouts_per_problem
        ).mean(dim=1)
        per_problem_early = early.reshape(
            roots, self.num_rollouts_per_problem
        ).mean(dim=1)

        global_root_count = self.distributed.sum_int(roots)
        global_trajectory_count = self.distributed.sum_int(expected)
        global_valid_count = self.distributed.sum_int(int(valid.sum().item()))
        if global_root_count <= 0 or global_trajectory_count <= 0:
            raise ValueError("Trajectory KL has no global held-out population")
        global_kl = self.distributed.sum_float(
            float(per_problem_kl.sum().item())
        ) / global_root_count
        global_length = self.distributed.sum_float(
            float(lengths.sum().item())
        ) / global_trajectory_count
        global_early = self.distributed.sum_float(
            float(early.sum().item())
        ) / global_trajectory_count
        return TrajectoryKLEvaluation(
            mean=global_kl,
            problem_ids=tuple(problem_ids),
            per_problem_kl=per_problem_kl.cpu(),
            per_problem_mean_length=per_problem_length.cpu(),
            per_problem_early_eos_rate=per_problem_early.cpu(),
            valid_state_count=global_valid_count,
            trajectory_count=global_trajectory_count,
            mean_length=global_length,
            early_eos_rate=global_early,
        )
