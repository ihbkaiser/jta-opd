from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import torch

@dataclass(frozen=True)
class OneStepKLProbeConfig:
    enabled: bool = False
    state_source: str = "on_policy_heldout_roots"
    benchmark: str = "Competition-MATH"
    subset_size: int = 64
    num_rollouts_per_problem: int = 2
    horizon: int = 64
    seed: int = 20260923
    interval_steps: int = 50
    metric: str = "full_vocab_reverse_kl"
    failure_policy: str = "error"
    artifact_subdir: str = "one_step_trajectory_kl_probe"
    sampling_temperature: float = 1.0
    sampling_top_p: float = 1.0
    generation_batch_size: int = 8
    score_micro_batch_size: int = 1

    @classmethod
    def from_mapping(
        cls, settings: Mapping[str, Any] | None, *, method: str
    ) -> "OneStepKLProbeConfig":
        values = dict(settings or {})
        config = cls(
            enabled=bool(values.get("enabled", False)),
            state_source=str(
                values.get("state_source", "on_policy_heldout_roots")
            ),
            benchmark=str(values.get("benchmark", "Competition-MATH")),
            subset_size=int(values.get("subset_size", 64)),
            num_rollouts_per_problem=int(
                values.get("num_rollouts_per_problem", 2)
            ),
            horizon=int(values.get("horizon", 64)),
            seed=int(values.get("seed", 20260923)),
            interval_steps=int(values.get("interval_steps", 50)),
            metric=str(values.get("metric", "full_vocab_reverse_kl")),
            failure_policy=str(values.get("failure_policy", "error")),
            artifact_subdir=str(
                values.get("artifact_subdir", "one_step_trajectory_kl_probe")
            ),
            sampling_temperature=float(values.get("sampling_temperature", 1.0)),
            sampling_top_p=float(values.get("sampling_top_p", 1.0)),
            generation_batch_size=int(values.get("generation_batch_size", 8)),
            score_micro_batch_size=int(values.get("score_micro_batch_size", 1)),
        )
        config.validate(method=method)
        return config

    def validate(self, *, method: str) -> None:
        if self.enabled and str(method).strip().lower() != "cmt":
            raise ValueError("one_step_kl_probe is only supported for method=cmt")
        if self.state_source != "on_policy_heldout_roots":
            raise ValueError(
                "one_step_kl_probe.state_source must be "
                "'on_policy_heldout_roots'"
            )
        if not self.benchmark.strip():
            raise ValueError("one_step_kl_probe.benchmark cannot be empty")
        if self.subset_size <= 0:
            raise ValueError("one_step_kl_probe.subset_size must be positive")
        if self.num_rollouts_per_problem <= 0:
            raise ValueError(
                "one_step_kl_probe.num_rollouts_per_problem must be positive"
            )
        if self.horizon <= 0:
            raise ValueError("one_step_kl_probe.horizon must be positive")
        if self.interval_steps <= 0:
            raise ValueError("one_step_kl_probe.interval_steps must be positive")
        if self.metric != "full_vocab_reverse_kl":
            raise ValueError(
                "one_step_kl_probe.metric must be 'full_vocab_reverse_kl'"
            )
        if self.failure_policy not in {"error", "warn"}:
            raise ValueError(
                "one_step_kl_probe.failure_policy must be 'error' or 'warn'"
            )
        if not self.artifact_subdir.strip():
            raise ValueError("one_step_kl_probe.artifact_subdir cannot be empty")
        if (
            not math.isfinite(self.sampling_temperature)
            or self.sampling_temperature <= 0.0
        ):
            raise ValueError(
                "one_step_kl_probe.sampling_temperature must be finite and positive"
            )
        if self.sampling_top_p != 1.0:
            raise ValueError("one_step_kl_probe requires sampling_top_p=1.0")
        if self.generation_batch_size <= 0:
            raise ValueError(
                "one_step_kl_probe.generation_batch_size must be positive"
            )
        if self.score_micro_batch_size <= 0:
            raise ValueError(
                "one_step_kl_probe.score_micro_batch_size must be positive"
            )

    def should_probe(self, optimizer_step: int) -> bool:
        step = int(optimizer_step)
        return self.enabled and step > 0 and step % self.interval_steps == 0


@torch.no_grad()
def conditional_reverse_kl_values(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    if student_log_probs.shape != teacher_log_probs.shape:
        raise ValueError("Student and teacher candidate tensors must share one shape")
    if student_log_probs.ndim != 3:
        raise ValueError("Candidate log-probabilities must have shape [batch, time, K]")
    if valid_mask.shape != student_log_probs.shape[:2]:
        raise ValueError("Probe valid mask must align with candidate tensors")
    student = student_log_probs.detach().float()
    teacher = teacher_log_probs.detach().float()
    if not bool(torch.isfinite(student).all() and torch.isfinite(teacher).all()):
        raise FloatingPointError("Probe KL received non-finite log-probabilities")
    student_conditional = student - torch.logsumexp(student, dim=-1, keepdim=True)
    teacher_conditional = teacher - torch.logsumexp(teacher, dim=-1, keepdim=True)
    per_token = (
        student_conditional.exp() * (student_conditional - teacher_conditional)
    ).sum(dim=-1)
    selected = per_token[valid_mask.bool()]
    if selected.numel() == 0:
        raise ValueError("Probe KL reference contains no valid response tokens")
    if not bool(torch.isfinite(selected).all()):
        raise FloatingPointError("Probe KL produced non-finite values")
    return selected.double()


@torch.no_grad()
def conditional_reverse_kl(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    valid_mask: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    selected = conditional_reverse_kl_values(
        student_log_probs, teacher_log_probs, valid_mask
    )
    return selected.sum(), int(selected.numel())


def paired_improvement(
    kl_before: float, kl_after_uniform: float, kl_after_cmt: float
) -> dict[str, float]:
    values = tuple(float(value) for value in (kl_before, kl_after_uniform, kl_after_cmt))
    if not all(math.isfinite(value) for value in values):
        raise FloatingPointError("Paired KL improvement requires finite values")
    before, after_uniform, after_cmt = values
    delta_uniform = before - after_uniform
    delta_cmt = before - after_cmt
    return {
        "delta_uniform": delta_uniform,
        "delta_cmt": delta_cmt,
        "paired_gap": delta_cmt - delta_uniform,
    }


def paired_sample_statistics(
    kl_before: torch.Tensor,
    kl_after_uniform: torch.Tensor,
    kl_after_cmt: torch.Tensor,
    distributed,
) -> dict[str, float | int]:
    before = kl_before.detach().double().reshape(-1)
    uniform = kl_after_uniform.detach().double().reshape(-1)
    cmt = kl_after_cmt.detach().double().reshape(-1)
    if before.shape != uniform.shape or before.shape != cmt.shape:
        raise ValueError("Paired KL sample tensors must share one shape")
    if before.numel() == 0:
        raise ValueError("Paired KL statistics require at least one successor state")
    if not bool(
        torch.isfinite(before).all()
        and torch.isfinite(uniform).all()
        and torch.isfinite(cmt).all()
    ):
        raise FloatingPointError("Paired KL sample tensors must be finite")

    def summarize(values: torch.Tensor) -> tuple[float, float, int]:
        count = distributed.sum_int(int(values.numel()))
        total = distributed.sum_float(float(values.sum().item()))
        total_squared = distributed.sum_float(float(values.square().sum().item()))
        if count <= 0:
            raise ValueError("Paired KL statistics have no global samples")
        mean = total / count
        if count == 1:
            return mean, 0.0, count
        centered_sum_squares = max(0.0, total_squared - total * total / count)
        standard_error = math.sqrt(centered_sum_squares / (count * (count - 1)))
        return mean, standard_error, count

    delta_uniform = before - uniform
    delta_cmt = before - cmt
    paired_gap = delta_cmt - delta_uniform
    uniform_mean, uniform_se, count = summarize(delta_uniform)
    cmt_mean, cmt_se, cmt_count = summarize(delta_cmt)
    gap_mean, gap_se, gap_count = summarize(paired_gap)
    if cmt_count != count or gap_count != count:
        raise RuntimeError("Paired KL statistics changed sample population")
    return {
        "sample_count": count,
        "delta_uniform": uniform_mean,
        "delta_uniform_standard_error": uniform_se,
        "delta_cmt": cmt_mean,
        "delta_cmt_standard_error": cmt_se,
        "paired_gap": gap_mean,
        "paired_gap_standard_error": gap_se,
    }
