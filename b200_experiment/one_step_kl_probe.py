from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import torch

from .opd_core import OPD_LOSS_TOP_K


@dataclass(frozen=True)
class OneStepKLProbeConfig:
    enabled: bool = False
    benchmark: str = "Competition-MATH"
    subset_size: int = 64
    seed: int = 20260922
    interval_steps: int = 1
    top_k: int = OPD_LOSS_TOP_K
    metric: str = "conditional_reverse_kl"
    failure_policy: str = "error"
    artifact_subdir: str = "one_step_kl_probe"
    max_new_tokens: int = 512
    temperature: float = 1.0
    top_p: float = 1.0
    generation_batch_size: int = 8
    score_micro_batch_size: int = 1

    @classmethod
    def from_mapping(
        cls, settings: Mapping[str, Any] | None, *, method: str
    ) -> "OneStepKLProbeConfig":
        values = dict(settings or {})
        config = cls(
            enabled=bool(values.get("enabled", False)),
            benchmark=str(values.get("benchmark", "Competition-MATH")),
            subset_size=int(values.get("subset_size", 64)),
            seed=int(values.get("seed", 20260922)),
            interval_steps=int(values.get("interval_steps", 1)),
            top_k=int(values.get("top_k", OPD_LOSS_TOP_K)),
            metric=str(values.get("metric", "conditional_reverse_kl")),
            failure_policy=str(values.get("failure_policy", "error")),
            artifact_subdir=str(values.get("artifact_subdir", "one_step_kl_probe")),
            max_new_tokens=int(values.get("max_new_tokens", 512)),
            temperature=float(values.get("temperature", 1.0)),
            top_p=float(values.get("top_p", 1.0)),
            generation_batch_size=int(values.get("generation_batch_size", 8)),
            score_micro_batch_size=int(values.get("score_micro_batch_size", 1)),
        )
        config.validate(method=method)
        return config

    def validate(self, *, method: str) -> None:
        if self.enabled and str(method).strip().lower() != "cmt":
            raise ValueError("one_step_kl_probe is only supported for method=cmt")
        if self.subset_size <= 0:
            raise ValueError("one_step_kl_probe.subset_size must be positive")
        if self.interval_steps <= 0:
            raise ValueError("one_step_kl_probe.interval_steps must be positive")
        if self.top_k != OPD_LOSS_TOP_K:
            raise ValueError(
                f"one_step_kl_probe requires top_k={OPD_LOSS_TOP_K}"
            )
        if self.metric != "conditional_reverse_kl":
            raise ValueError(
                "one_step_kl_probe.metric must be 'conditional_reverse_kl'"
            )
        if self.failure_policy not in {"error", "warn"}:
            raise ValueError(
                "one_step_kl_probe.failure_policy must be 'error' or 'warn'"
            )
        if not self.benchmark.strip():
            raise ValueError("one_step_kl_probe.benchmark cannot be empty")
        if not self.artifact_subdir.strip():
            raise ValueError("one_step_kl_probe.artifact_subdir cannot be empty")
        if self.max_new_tokens <= 0:
            raise ValueError("one_step_kl_probe.max_new_tokens must be positive")
        if not math.isfinite(self.temperature) or self.temperature <= 0.0:
            raise ValueError("one_step_kl_probe.temperature must be finite and positive")
        if not math.isfinite(self.top_p) or not 0.0 < self.top_p <= 1.0:
            raise ValueError("one_step_kl_probe.top_p must lie in (0, 1]")
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
def conditional_reverse_kl(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    valid_mask: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    if student_log_probs.shape != teacher_log_probs.shape:
        raise ValueError("Student and teacher candidate tensors must share one shape")
    if student_log_probs.ndim != 3:
        raise ValueError("Candidate log-probabilities must have shape [batch, time, K]")
    if valid_mask.shape != student_log_probs.shape[:2]:
        raise ValueError("Held-out valid mask must align with candidate tensors")
    student = student_log_probs.detach().float()
    teacher = teacher_log_probs.detach().float()
    if not bool(torch.isfinite(student).all() and torch.isfinite(teacher).all()):
        raise FloatingPointError("Held-out KL received non-finite log-probabilities")
    student_conditional = student - torch.logsumexp(student, dim=-1, keepdim=True)
    teacher_conditional = teacher - torch.logsumexp(teacher, dim=-1, keepdim=True)
    per_token = (
        student_conditional.exp() * (student_conditional - teacher_conditional)
    ).sum(dim=-1)
    selected = per_token[valid_mask.bool()]
    if selected.numel() == 0:
        raise ValueError("Held-out KL reference contains no valid response tokens")
    if not bool(torch.isfinite(selected).all()):
        raise FloatingPointError("Held-out KL produced non-finite values")
    return selected.double().sum(), int(selected.numel())


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
