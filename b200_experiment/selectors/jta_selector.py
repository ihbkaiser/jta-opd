"""Joint Token Allocation for OPD.

The production embedding is a compact context-aware TensorSketch of the
Top-K OPD coefficient vector and the detached hidden state feeding the same
candidate logits.  It is a structured proxy, not a full-parameter or exact
LM-head gradient: the dense softmax tail is intentionally omitted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

from ..opd_core import TopKOPDReference


@dataclass
class JTAOutput:
    """Detached JTA multipliers, probabilities, and bounded diagnostics."""

    weights: torch.Tensor
    probabilities: torch.Tensor
    diagnostics: dict[str, Any]


class _DeterministicCountSketch:
    """One deterministic CountSketch map for integer coordinates."""

    _BUCKET_MULTIPLIER = 1_103_515_245
    _SIGN_MULTIPLIER = 2_654_435_761
    _PRIME = 2_147_483_647

    def __init__(self, sketch_dim: int, sketch_seed: int = 42) -> None:
        if int(sketch_dim) <= 0:
            raise ValueError("sketch_dim must be positive")
        self.sketch_dim = int(sketch_dim)
        self.sketch_seed = int(sketch_seed)

    def hash(self, coordinates: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        indices = coordinates.to(dtype=torch.long)
        if bool((indices < 0).any()):
            raise ValueError("CountSketch coordinates must be non-negative")
        seed = int(self.sketch_seed)
        bucket_hash = torch.remainder(
            indices * self._BUCKET_MULTIPLIER + (seed + 1) * 12_345,
            self._PRIME,
        )
        sign_hash = torch.remainder(
            indices * self._SIGN_MULTIPLIER + (seed + 1) * 67_867,
            self._PRIME,
        )
        buckets = torch.remainder(bucket_hash, self.sketch_dim).long()
        signs = torch.where(
            torch.remainder(sign_hash, 2).eq(0),
            torch.ones_like(sign_hash, dtype=torch.float32),
            -torch.ones_like(sign_hash, dtype=torch.float32),
        )
        return buckets, signs

    def reduce_dense(self, values: torch.Tensor, *, chunk_size: int) -> torch.Tensor:
        if values.ndim < 1:
            raise ValueError("dense CountSketch input must have at least one dimension")
        if int(chunk_size) <= 0:
            raise ValueError("CountSketch chunk_size must be positive")
        values = values.to(dtype=torch.float32)
        flat = values.reshape(-1, values.shape[-1])
        coordinates = torch.arange(values.shape[-1], device=values.device)
        buckets, signs = self.hash(coordinates)
        result = torch.zeros(
            flat.shape[0], self.sketch_dim, dtype=torch.float32, device=values.device
        )
        for begin in range(0, flat.shape[0], int(chunk_size)):
            end = min(begin + int(chunk_size), flat.shape[0])
            result[begin:end].scatter_add_(
                1,
                buckets.unsqueeze(0).expand(end - begin, -1),
                flat[begin:end] * signs.unsqueeze(0),
            )
        return result.reshape(*values.shape[:-1], self.sketch_dim)


class CountSketchOperator:
    """Deterministic token-ID CountSketch ablation with FP32 accumulation."""

    def __init__(
        self,
        sketch_dim: int,
        sketch_seed: int = 42,
        token_chunk_size: int = 4096,
    ) -> None:
        if int(sketch_dim) <= 0:
            raise ValueError("sketch_dim must be positive")
        if int(token_chunk_size) <= 0:
            raise ValueError("token_chunk_size must be positive")
        self.sketch_dim = int(sketch_dim)
        self.sketch_seed = int(sketch_seed)
        self.token_chunk_size = int(token_chunk_size)
        self.vocab = _DeterministicCountSketch(self.sketch_dim, self.sketch_seed)

    def hash(self, token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.vocab.hash(token_ids)

    @staticmethod
    def _broadcast_mask(
        value: torch.Tensor | None, target: torch.Tensor, *, name: str
    ) -> torch.Tensor:
        if value is None:
            return torch.ones_like(target, dtype=torch.bool)
        mask = value.to(device=target.device, dtype=torch.bool)
        try:
            return torch.broadcast_to(mask, target.shape)
        except RuntimeError as error:
            raise ValueError(f"{name} must broadcast to candidate shape") from error

    @staticmethod
    def _broadcast_weights(
        value: torch.Tensor | None, target: torch.Tensor, *, name: str
    ) -> torch.Tensor:
        if value is None:
            return torch.ones_like(target, dtype=torch.float32)
        weights = value.to(device=target.device, dtype=torch.float32)
        try:
            return torch.broadcast_to(weights, target.shape)
        except RuntimeError as error:
            raise ValueError(f"{name} must broadcast to candidate shape") from error

    def reduce(
        self,
        coefficients: torch.Tensor,
        candidate_ids: torch.Tensor,
        token_weights: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Reduce sparse candidate coefficients with duplicate accumulation."""
        if candidate_ids.ndim < 1 or coefficients.shape != candidate_ids.shape:
            raise ValueError("coefficients and candidate_ids must share shape [..., K]")
        if candidate_ids.shape[-1] <= 0:
            raise ValueError("candidate support cannot be empty")
        ids = candidate_ids.to(dtype=torch.long)
        values = coefficients.to(dtype=torch.float32, device=ids.device)
        weights = self._broadcast_weights(token_weights, ids, name="token_weights")
        valid = self._broadcast_mask(mask, ids, name="mask")
        flat_ids = ids.reshape(-1, ids.shape[-1])
        flat_values = values.reshape_as(flat_ids).float()
        flat_weights = weights.reshape_as(flat_ids).float()
        flat_valid = valid.reshape_as(flat_ids)
        result = torch.zeros(
            flat_ids.shape[0], self.sketch_dim, dtype=torch.float32, device=ids.device
        )
        for begin in range(0, flat_ids.shape[0], self.token_chunk_size):
            end = min(begin + self.token_chunk_size, flat_ids.shape[0])
            ids_chunk = torch.where(
                flat_valid[begin:end], flat_ids[begin:end], torch.zeros_like(flat_ids[begin:end])
            )
            buckets, signs = self.vocab.hash(ids_chunk)
            contribution = flat_values[begin:end] * flat_weights[begin:end]
            contribution = contribution * flat_valid[begin:end].float() * signs
            result[begin:end].scatter_add_(1, buckets, contribution)
        return result.reshape(*candidate_ids.shape[:-1], self.sketch_dim)

    def dot(
        self,
        coefficients: torch.Tensor,
        candidate_ids: torch.Tensor,
        vectors: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        embeddings = self.reduce(coefficients, candidate_ids, mask=mask)
        vector = vectors.to(device=embeddings.device, dtype=torch.float32)
        if vector.shape == (self.sketch_dim,):
            return (embeddings * vector).sum(dim=-1)
        if vector.shape != embeddings.shape:
            raise ValueError("vectors must be [sketch_dim] or match reduced shape")
        return (embeddings * vector).sum(dim=-1)

    def squared_norm(
        self,
        coefficients: torch.Tensor,
        candidate_ids: torch.Tensor,
        token_weights: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        embeddings = self.reduce(
            coefficients,
            candidate_ids,
            token_weights=token_weights,
            mask=mask,
        )
        return embeddings.square().sum(dim=-1)


class TensorSketchOperator:
    """TensorSketch of sparse vocabulary coefficients and hidden context."""

    def __init__(
        self,
        sketch_dim: int,
        vocab_hash_seed: int = 42,
        hidden_hash_seed: int = 1729,
        token_chunk_size: int = 4096,
    ) -> None:
        if int(sketch_dim) <= 0:
            raise ValueError("sketch_dim must be positive")
        if int(token_chunk_size) <= 0:
            raise ValueError("token_chunk_size must be positive")
        self.sketch_dim = int(sketch_dim)
        self.vocab_hash_seed = int(vocab_hash_seed)
        self.hidden_hash_seed = int(hidden_hash_seed)
        self.token_chunk_size = int(token_chunk_size)
        self.vocab = _DeterministicCountSketch(self.sketch_dim, self.vocab_hash_seed)
        self.hidden = _DeterministicCountSketch(self.sketch_dim, self.hidden_hash_seed)

    def reduce_vocab(
        self,
        coefficients: torch.Tensor,
        candidate_ids: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if coefficients.shape != candidate_ids.shape:
            raise ValueError("coefficients and candidate_ids must share shape [..., K]")
        ids = candidate_ids.to(dtype=torch.long)
        values = coefficients.to(device=ids.device, dtype=torch.float32)
        valid = torch.ones_like(ids, dtype=torch.bool)
        if mask is not None:
            valid = torch.broadcast_to(mask.to(device=ids.device, dtype=torch.bool), ids.shape)
            values = values * valid.float()
        flat_ids = ids.reshape(-1, ids.shape[-1])
        flat_values = values.reshape_as(flat_ids)
        flat_valid = valid.reshape_as(flat_ids)
        result = torch.zeros(
            flat_ids.shape[0], self.sketch_dim, dtype=torch.float32, device=ids.device
        )
        for begin in range(0, flat_ids.shape[0], self.token_chunk_size):
            end = min(begin + self.token_chunk_size, flat_ids.shape[0])
            ids_chunk = torch.where(
                flat_valid[begin:end], flat_ids[begin:end], torch.zeros_like(flat_ids[begin:end])
            )
            buckets, signs = self.vocab.hash(ids_chunk)
            result[begin:end].scatter_add_(
                1, buckets, flat_values[begin:end] * signs
            )
        return result.reshape(*ids.shape[:-1], self.sketch_dim)

    def reduce_hidden(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.hidden.reduce_dense(
            hidden_states.to(dtype=torch.float32), chunk_size=self.token_chunk_size
        )

    def convolve(
        self, vocab_sketch: torch.Tensor, hidden_sketch: torch.Tensor
    ) -> torch.Tensor:
        if vocab_sketch.shape != hidden_sketch.shape:
            raise ValueError("vocabulary and hidden sketches must have identical shapes")
        if vocab_sketch.shape[-1] != self.sketch_dim:
            raise ValueError("sketch vectors have the wrong dimension")
        return torch.fft.ifft(
            torch.fft.fft(vocab_sketch.float(), dim=-1)
            * torch.fft.fft(hidden_sketch.float(), dim=-1),
            dim=-1,
        ).real.float()

    def reduce(
        self,
        coefficients: torch.Tensor,
        candidate_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if hidden_states.shape[:-1] != candidate_ids.shape[:-1]:
            raise ValueError("hidden states must align with candidate leading dimensions")
        return self.convolve(
            self.reduce_vocab(coefficients, candidate_ids, mask=mask),
            self.reduce_hidden(hidden_states),
        )

    def coefficient_norm(
        self,
        coefficients: torch.Tensor,
        candidate_ids: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Exact L2 norm after coalescing duplicate sparse vocabulary IDs."""
        if coefficients.shape != candidate_ids.shape:
            raise ValueError("coefficients and candidate_ids must share shape [..., K]")
        ids = candidate_ids.to(dtype=torch.long)
        values = coefficients.to(device=ids.device, dtype=torch.float32)
        valid = torch.ones_like(ids, dtype=torch.bool)
        if mask is not None:
            valid = torch.broadcast_to(mask.to(device=ids.device, dtype=torch.bool), ids.shape)
            values = values * valid.float()
        safe_ids = torch.where(valid, ids, torch.zeros_like(ids))
        sorted_ids, order = torch.sort(safe_ids, dim=-1)
        sorted_values = torch.gather(values, -1, order)
        new_group = torch.ones_like(sorted_ids, dtype=torch.bool)
        if sorted_ids.shape[-1] > 1:
            new_group[..., 1:] = sorted_ids[..., 1:] != sorted_ids[..., :-1]
        group = new_group.cumsum(dim=-1) - 1
        grouped = torch.zeros_like(sorted_values)
        grouped.scatter_add_(-1, group, sorted_values)
        return grouped.square().sum(dim=-1).sqrt()

    def hidden_norm(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states.float().square().sum(dim=-1).sqrt()


def _validate_probability_inputs(
    scores: torch.Tensor, prior: torch.Tensor, epsilon: float, tolerance: float
) -> tuple[torch.Tensor, torch.Tensor, float]:
    if scores.ndim != 1 or prior.ndim != 1 or scores.shape != prior.shape:
        raise ValueError("scores and prior must be non-empty one-dimensional tensors")
    if scores.numel() == 0:
        raise ValueError("KL oracle needs at least one candidate")
    if epsilon < 0.0:
        raise ValueError("epsilon must be non-negative")
    if tolerance < 0.0:
        raise ValueError("constraint tolerance must be non-negative")
    scores = scores.detach().float()
    prior = prior.detach().float()
    if not torch.isfinite(scores).all() or not torch.isfinite(prior).all():
        raise FloatingPointError("KL oracle received non-finite scores or prior")
    if bool((prior <= 0).any()):
        raise ValueError("KL oracle prior must be strictly positive")
    prior = prior / prior.sum().clamp_min(torch.finfo(prior.dtype).tiny)
    return scores, prior, float(epsilon)


def _kl_oracle_with_stats(
    scores: torch.Tensor,
    prior: torch.Tensor,
    epsilon: float,
    *,
    iterations: int = 50,
    tolerance: float = 1e-6,
) -> tuple[torch.Tensor, float, float]:
    scores, prior, epsilon = _validate_probability_inputs(
        scores, prior, epsilon, tolerance
    )
    mask = torch.ones_like(scores, dtype=torch.bool)
    probability, beta, achieved = _kl_oracle_batched_with_stats(
        scores.unsqueeze(0),
        prior.unsqueeze(0),
        mask.unsqueeze(0),
        epsilon,
        iterations=iterations,
        tolerance=tolerance,
    )
    return probability[0], float(beta[0].item()), float(achieved[0].item())


def _kl_oracle_batched_with_stats(
    scores: torch.Tensor,
    prior: torch.Tensor,
    mask: torch.Tensor,
    epsilon: float,
    *,
    iterations: int = 50,
    tolerance: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Solve independent KL linear oracles over a padded prompt batch."""
    if scores.ndim != 2 or prior.shape != scores.shape or mask.shape != scores.shape:
        raise ValueError("batched KL oracle inputs must have shape [P, M]")
    if int(iterations) <= 0:
        raise ValueError("KL bisection iterations must be positive")
    if epsilon < 0.0 or tolerance < 0.0:
        raise ValueError("epsilon and tolerance must be non-negative")
    scores = scores.detach().float()
    prior = prior.detach().float()
    mask = mask.detach().bool()
    if bool((mask.sum(dim=-1) <= 0).any()):
        raise ValueError("Every batched KL oracle row needs one candidate")
    if not bool(torch.isfinite(scores[mask]).all()):
        raise FloatingPointError("KL oracle received non-finite scores")
    if bool((prior[mask] <= 0).any()) or not bool(torch.isfinite(prior[mask]).all()):
        raise ValueError("KL oracle prior must be finite and strictly positive")

    tiny = torch.finfo(scores.dtype).tiny
    prior = torch.where(mask, prior, torch.zeros_like(prior))
    prior = prior / prior.sum(dim=-1, keepdim=True).clamp_min(tiny)
    safe_log_prior = torch.where(
        mask, prior.clamp_min(tiny).log(), torch.zeros_like(prior)
    )
    negative_inf = torch.full_like(scores, -torch.inf)
    maximum = scores.masked_fill(~mask, -torch.inf).amax(dim=-1)
    minimum = scores.masked_fill(~mask, torch.inf).amin(dim=-1)
    centered = torch.where(mask, scores - maximum.unsqueeze(-1), torch.zeros_like(scores))
    best = mask & scores.eq(maximum.unsqueeze(-1))
    max_kl = -(
        prior.masked_fill(~best, 0.0).sum(dim=-1).clamp_min(tiny).log()
    )
    # Leave a tiny FP32/bisection safety margin.  This matters for deliberately
    # short smoke-test bisections (e.g. 10 iterations) because FW convexity
    # preserves feasibility only up to the oracle's numerical boundary error.
    boundary_safety = min(1.0e-4, max(float(tolerance), float(epsilon) * 1.0e-3))
    target = torch.minimum(
        torch.full_like(max_kl, max(0.0, float(epsilon) - boundary_safety)), max_kl
    )
    trivial = (mask.sum(dim=-1) <= 1) | ((maximum - minimum) <= 1e-12)
    large_epsilon = (~trivial) & (target >= max_kl - float(tolerance))
    active = (~trivial) & (~large_epsilon) & (float(epsilon) > float(tolerance))

    def distribution(beta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = torch.where(
            mask,
            safe_log_prior + centered * beta.unsqueeze(-1),
            negative_inf,
        )
        log_probability = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
        probability = torch.where(mask, log_probability.exp(), torch.zeros_like(logits))
        safe_log_probability = torch.where(
            mask, log_probability, torch.zeros_like(log_probability)
        )
        kl = (
            probability * (safe_log_probability - safe_log_prior)
        ).sum(dim=-1)
        return probability, kl

    low = torch.zeros_like(max_kl)
    high = torch.ones_like(max_kl)
    # A fixed, vectorized bracket search avoids a host-side while condition.
    for _ in range(64):
        _, high_kl = distribution(high)
        grow = active & (high_kl < target)
        high = torch.where(grow, high.mul(2.0).clamp_max(1e12), high)
        if not bool(grow.any()):
            break
    for _ in range(int(iterations)):
        midpoint = (low + high) * 0.5
        _, midpoint_kl = distribution(midpoint)
        below = active & (midpoint_kl < target)
        low = torch.where(below, midpoint, low)
        high = torch.where(below, high, midpoint)
    beta = (low + high) * 0.5
    probability, achieved_kl = distribution(beta)
    # The midpoint is usually the most accurate root estimate, but with a
    # deliberately short bisection it can lie just beyond the feasible side
    # of the KL boundary.  Keep the final feasible endpoint in that case so
    # the returned simplex remains a valid constraint-preserving oracle.
    low_probability, low_kl = distribution(low)
    use_feasible_endpoint = active & (achieved_kl > target)
    probability = torch.where(
        use_feasible_endpoint.unsqueeze(-1), low_probability, probability
    )
    achieved_kl = torch.where(use_feasible_endpoint, low_kl, achieved_kl)
    beta = torch.where(use_feasible_endpoint, low, beta)
    best_probability = torch.where(best, prior, torch.zeros_like(prior))
    best_probability = best_probability / best_probability.sum(
        dim=-1, keepdim=True
    ).clamp_min(tiny)
    probability = torch.where(
        active.unsqueeze(-1), probability, prior
    )
    probability = torch.where(
        large_epsilon.unsqueeze(-1), best_probability, probability
    )
    probability = torch.where(mask, probability, torch.zeros_like(probability))
    beta = torch.where(
        large_epsilon,
        torch.full_like(beta, torch.inf),
        torch.where(active, beta, torch.zeros_like(beta)),
    )
    achieved_kl = torch.where(
        large_epsilon,
        max_kl,
        torch.where(active, achieved_kl, torch.zeros_like(achieved_kl)),
    )
    return probability, beta, achieved_kl


@torch.no_grad()
def kl_ball_linear_oracle(
    scores: torch.Tensor,
    prior: torch.Tensor,
    epsilon: float,
    *,
    mask: torch.Tensor | None = None,
    iterations: int = 50,
    tolerance: float = 1e-6,
) -> torch.Tensor:
    """Maximize a linear score over a simplex KL ball around ``prior``."""
    if scores.ndim == 2:
        if mask is None:
            mask = prior > 0
        probability, _, _ = _kl_oracle_batched_with_stats(
            scores,
            prior,
            mask,
            epsilon,
            iterations=iterations,
            tolerance=tolerance,
        )
        return probability
    if mask is not None:
        raise ValueError("mask is only supported for batched KL oracle inputs")
    probability, _, _ = _kl_oracle_with_stats(
        scores,
        prior,
        epsilon,
        iterations=iterations,
        tolerance=tolerance,
    )
    return probability


def _solve_prompts_batched(
    embeddings: torch.Tensor,
    utilities: torch.Tensor,
    epsilon: float,
    *,
    mask: torch.Tensor,
    fw_max_iterations: int,
    fw_gap_tolerance: float,
    kl_bisection_iterations: int,
    constraint_tolerance: float,
) -> dict[str, torch.Tensor | list[torch.Tensor]]:
    """Run all prompt-local FW problems in one padded tensor batch."""
    if embeddings.ndim != 3 or utilities.ndim != 2 or mask.shape != utilities.shape:
        raise ValueError("batched JTA solver expects embeddings [P, M, D]")
    if embeddings.shape[:2] != utilities.shape:
        raise ValueError("JTA embeddings and utilities must align")
    mask = mask.bool()
    tiny = torch.finfo(embeddings.dtype).tiny
    counts = mask.sum(dim=-1).clamp_min(1).to(dtype=embeddings.dtype)
    prior = mask.to(dtype=embeddings.dtype) / counts.unsqueeze(-1)

    def objective(probability: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sketch = torch.bmm(probability.unsqueeze(1), embeddings).squeeze(1)
        linear = (probability * utilities).sum(dim=-1)
        quadratic = 0.5 * sketch.square().sum(dim=-1)
        return linear - quadratic, linear, quadratic

    probability = prior.clone()
    current, _, _ = objective(probability)
    initial = current.detach().clone()
    history = [current.detach().clone()]
    iterations = torch.zeros_like(counts, dtype=torch.int64)
    gamma_sum = torch.zeros_like(current)
    last_gamma = torch.zeros_like(current)
    done = mask.sum(dim=-1) <= 1
    final_gap = torch.where(
        done, torch.zeros_like(current), torch.full_like(current, torch.inf)
    )
    for _ in range(max(1, int(fw_max_iterations))):
        if not bool((~done).any()):
            break
        sketch = torch.bmm(probability.unsqueeze(1), embeddings).squeeze(1)
        marginal = utilities - torch.bmm(
            embeddings, sketch.unsqueeze(-1)
        ).squeeze(-1)
        oracle, _, _ = _kl_oracle_batched_with_stats(
            marginal,
            prior,
            mask,
            epsilon,
            iterations=kl_bisection_iterations,
            tolerance=constraint_tolerance,
        )
        active = ~done
        direction = torch.where(
            active.unsqueeze(-1), oracle - probability, torch.zeros_like(probability)
        )
        gap = (marginal * direction).sum(dim=-1).clamp_min(0.0)
        scale = torch.maximum(current.abs(), torch.ones_like(current))
        newly_done = active & (gap <= float(fw_gap_tolerance) * scale)
        update = active & ~newly_done
        final_gap = torch.where(active, gap, final_gap)

        direction_sketch = torch.bmm(direction.unsqueeze(1), embeddings).squeeze(1)
        denominator = direction_sketch.square().sum(dim=-1)
        gamma = torch.where(
            denominator > 0.0,
            (gap / denominator).clamp(0.0, 1.0),
            torch.where(gap > 0.0, torch.ones_like(gap), torch.zeros_like(gap)),
        )
        gamma = torch.where(update, gamma, torch.zeros_like(gamma))
        candidate = probability + gamma.unsqueeze(-1) * direction
        candidate = candidate.clamp_min(0.0)
        candidate = candidate * mask.to(dtype=candidate.dtype)
        candidate = candidate / candidate.sum(dim=-1, keepdim=True).clamp_min(tiny)
        updated, _, _ = objective(candidate)
        accept = update & (updated + 1e-7 >= current)
        probability = torch.where(accept.unsqueeze(-1), candidate, probability)
        current = torch.where(accept, updated, current)
        gamma = torch.where(accept, gamma, torch.zeros_like(gamma))
        iterations = iterations + update.to(dtype=iterations.dtype)
        gamma_sum = gamma_sum + gamma
        last_gamma = torch.where(accept, gamma, last_gamma)
        done = done | newly_done
        history.append(current.detach().clone())

    final_objective, linear, quadratic = objective(probability)
    final_sketch = torch.bmm(probability.unsqueeze(1), embeddings).squeeze(1)
    final_marginal = utilities - torch.bmm(
        embeddings, final_sketch.unsqueeze(-1)
    ).squeeze(-1)
    achieved_kl = (
        probability
        * torch.where(
            mask,
            (probability.clamp_min(tiny) / prior.clamp_min(tiny)).log(),
            torch.zeros_like(probability),
        )
    ).sum(dim=-1)
    max_score = final_marginal.masked_fill(~mask, -torch.inf).amax(dim=-1)
    best = mask & final_marginal.eq(max_score.unsqueeze(-1))
    max_kl = -(
        prior.masked_fill(~best, 0.0).sum(dim=-1).clamp_min(tiny).log()
    )
    scale = torch.maximum(final_objective.abs(), torch.ones_like(final_objective))
    converged = done | (final_gap <= float(fw_gap_tolerance) * scale)
    return {
        "probabilities": probability,
        "redundancy": torch.bmm(
            embeddings, torch.bmm(probability.unsqueeze(1), embeddings).transpose(1, 2)
        ).squeeze(-1),
        "marginal": final_marginal,
        "objective_history": history,
        "initial_objective": initial,
        "final_objective": final_objective,
        "linear_term": linear,
        "quadratic_term": quadratic,
        "iterations": iterations,
        "final_gap": final_gap,
        "converged": converged,
        "last_gamma": last_gamma,
        "mean_gamma": gamma_sum / iterations.clamp_min(1).to(gamma_sum.dtype),
        "achieved_kl": achieved_kl,
        "kl_boundary": (epsilon > 0.0)
        & (max_kl > 0.0)
        & ((achieved_kl - float(epsilon)).abs() <= 1e-4),
    }


class JTASelector:
    """Prompt-local Joint Token Allocation for the frozen OPD reference."""

    def __init__(
        self,
        *,
        epsilon: float = 0.1,
        sketch_dim: int = 512,
        sketch_seed: int = 42,
        hidden_sketch_seed: int = 1729,
        embedding_backend: str = "topk_context_tensorsketch",
        fw_max_iterations: int = 20,
        fw_gap_tolerance: float = 1.0e-5,
        kl_bisection_iterations: int = 50,
        constraint_tolerance: float = 1.0e-6,
        token_chunk_size: int = 4096,
    ) -> None:
        if float(epsilon) < 0.0:
            raise ValueError("JTA epsilon must be non-negative")
        if int(fw_max_iterations) <= 0:
            raise ValueError("JTA fw_max_iterations must be positive")
        if float(fw_gap_tolerance) < 0.0:
            raise ValueError("JTA fw_gap_tolerance must be non-negative")
        if int(kl_bisection_iterations) <= 0:
            raise ValueError("JTA kl_bisection_iterations must be positive")
        if float(constraint_tolerance) < 0.0:
            raise ValueError("JTA constraint_tolerance must be non-negative")
        if embedding_backend not in {
            "topk_context_tensorsketch",
            "topk_logprob_countsketch",
        }:
            raise ValueError(
                "Unsupported JTA embedding_backend; choose "
                "'topk_context_tensorsketch' or 'topk_logprob_countsketch'"
            )
        self.epsilon = float(epsilon)
        self.sketch_dim = int(sketch_dim)
        self.sketch_seed = int(sketch_seed)
        self.hidden_sketch_seed = int(hidden_sketch_seed)
        self.embedding_backend = str(embedding_backend)
        self.fw_max_iterations = int(fw_max_iterations)
        self.fw_gap_tolerance = float(fw_gap_tolerance)
        self.kl_bisection_iterations = int(kl_bisection_iterations)
        self.constraint_tolerance = float(constraint_tolerance)
        self.operator = CountSketchOperator(
            self.sketch_dim,
            self.sketch_seed,
            token_chunk_size,
        )
        self.tensor_operator = TensorSketchOperator(
            self.sketch_dim,
            vocab_hash_seed=self.sketch_seed,
            hidden_hash_seed=self.hidden_sketch_seed,
            token_chunk_size=token_chunk_size,
        )

    @staticmethod
    def _distributed_sum_tensor(distributed: Any, value: torch.Tensor) -> torch.Tensor:
        if distributed is None or not bool(getattr(distributed, "enabled", False)):
            return value.detach().clone()
        return distributed.sum_tensor(value)

    @staticmethod
    def _distributed_sum_int(distributed: Any, value: int) -> int:
        if distributed is None or not bool(getattr(distributed, "enabled", False)):
            return int(value)
        return int(distributed.sum_int(int(value)))

    def allocate(
        self,
        reference: TopKOPDReference,
        valid_mask: torch.Tensor,
        active_prompt_mask: torch.Tensor,
        num_responses: int,
        distributed: Any = None,
        *,
        hidden_states: torch.Tensor | None = None,
        hidden_sketches: torch.Tensor | None = None,
        hidden_state_norm: torch.Tensor | None = None,
    ) -> JTAOutput:
        """Allocate detached prompt-normalized multipliers for one rollout batch.

        The context backend accepts either detached hidden states (unit-test and
        small-batch API) or their streamed ``CS_hidden`` representation from
        scoring.  Production training retains only the latter.
        """
        ids = reference.candidate_ids
        advantages = reference.advantages
        if ids.ndim != 3 or advantages.shape != ids.shape:
            raise ValueError("JTA reference tensors must have shape [B, T, K]")
        if valid_mask.ndim != 2 or valid_mask.shape != ids.shape[:2]:
            raise ValueError("JTA valid_mask must have shape [B, T]")
        responses = int(num_responses)
        if responses <= 0:
            raise ValueError("num_responses must be positive")
        batch, time, _ = ids.shape
        if batch % responses != 0:
            raise ValueError("JTA batch must equal local_prompt_slots * num_responses")
        local_prompts = batch // responses
        active = active_prompt_mask.to(device=ids.device, dtype=torch.bool).reshape(-1)
        if active.numel() != local_prompts:
            raise ValueError("active_prompt_mask must have one value per local prompt")
        objective_valid = valid_mask.to(device=ids.device, dtype=torch.bool)
        support = reference.support_mask
        if support is None:
            support = torch.ones_like(advantages, dtype=torch.bool)
        elif support.shape != advantages.shape:
            raise ValueError("JTA reference support_mask must align with advantages")
        support = support.to(device=ids.device, dtype=torch.bool)
        delta = -advantages.detach().float()
        delta = torch.where(support, delta, torch.zeros_like(delta))

        context_backend = self.embedding_backend == "topk_context_tensorsketch"
        if context_backend:
            if hidden_states is not None and hidden_sketches is not None:
                raise ValueError("Pass hidden_states or hidden_sketches, not both")
            if hidden_states is not None:
                hidden_states = hidden_states.detach().to(device=ids.device)
                if hidden_states.shape[:2] != ids.shape[:2]:
                    raise ValueError("hidden_states must align with [B, T]")
                hidden_sketches = self.tensor_operator.reduce_hidden(hidden_states)
                hidden_state_norm = self.tensor_operator.hidden_norm(hidden_states)
            if hidden_sketches is None:
                raise ValueError(
                    "topk_context_tensorsketch requires detached hidden_states "
                    "or streamed hidden_sketches"
                )
            hidden_sketches = hidden_sketches.detach().to(
                device=ids.device, dtype=torch.float32
            )
            if hidden_sketches.shape != (*ids.shape[:2], self.sketch_dim):
                raise ValueError("hidden_sketches must have shape [B, T, sketch_dim]")
            hidden_sketches = torch.where(
                objective_valid.unsqueeze(-1), hidden_sketches, torch.zeros_like(hidden_sketches)
            )
            if hidden_state_norm is None:
                hidden_state_norm = hidden_sketches.square().sum(dim=-1).sqrt()
            else:
                hidden_state_norm = hidden_state_norm.detach().to(
                    device=ids.device, dtype=torch.float32
                )
                if hidden_state_norm.shape != objective_valid.shape:
                    raise ValueError("hidden_state_norm must have shape [B, T]")
                hidden_state_norm = torch.where(
                    objective_valid, hidden_state_norm, torch.zeros_like(hidden_state_norm)
                )
        elif hidden_states is not None or hidden_sketches is not None:
            raise ValueError(
                "hidden states are only accepted by topk_context_tensorsketch"
            )

        output_weights = torch.zeros(batch, time, dtype=torch.float32, device=ids.device)
        output_probabilities = torch.zeros_like(output_weights)
        utility_output = torch.zeros_like(output_weights)
        redundancy_output = torch.zeros_like(output_weights)
        marginal_output = torch.zeros_like(output_weights)
        embedding_norm_output = torch.zeros_like(output_weights)
        coefficient_norm_output = torch.zeros_like(output_weights)
        hidden_state_norm_output = torch.zeros_like(output_weights)
        prompt_references = torch.zeros(
            local_prompts, self.sketch_dim, dtype=torch.float32, device=ids.device
        )
        prompt_counts = objective_valid.reshape(local_prompts, responses, time).sum(
            dim=(1, 2)
        )
        prompt_counts_cpu = [int(value) for value in prompt_counts.detach().cpu().tolist()]
        active_cpu = [bool(value) for value in active.detach().cpu().tolist()]
        if any(
            count > 0 and not is_active
            for count, is_active in zip(prompt_counts_cpu, active_cpu)
        ):
            raise ValueError("Inactive JTA prompt contains objective-valid tokens")
        active_prompt_ids = [
            prompt for prompt, is_active in enumerate(active_cpu) if is_active
        ]
        local_total = torch.zeros(self.sketch_dim, dtype=torch.float32, device=ids.device)
        local_tokens = sum(prompt_counts_cpu[prompt] for prompt in active_prompt_ids)
        local_active_prompts = len(active_prompt_ids)

        def collect_prompt(
            prompt: int,
        ) -> tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]:
            row_begin, row_end = prompt * responses, (prompt + 1) * responses
            group_valid = objective_valid[row_begin:row_end].reshape(-1)
            count = prompt_counts_cpu[prompt]
            if count <= 0:
                raise ValueError("Every active JTA prompt must have at least one valid token")
            flat_ids = ids[row_begin:row_end].reshape(-1, ids.shape[-1])
            flat_delta = delta[row_begin:row_end].reshape(-1, ids.shape[-1])
            flat_support = support[row_begin:row_end].reshape(-1, ids.shape[-1])
            if context_backend:
                vocab_sketch = self.tensor_operator.reduce_vocab(
                    flat_delta[group_valid],
                    flat_ids[group_valid],
                    mask=flat_support[group_valid],
                )
                embeddings = self.tensor_operator.convolve(
                    vocab_sketch,
                    hidden_sketches[row_begin:row_end]
                    .reshape(-1, self.sketch_dim)[group_valid],
                )
                coefficient_norm = self.tensor_operator.coefficient_norm(
                    flat_delta[group_valid],
                    flat_ids[group_valid],
                    mask=flat_support[group_valid],
                )
            else:
                embeddings = self.operator.reduce(
                    flat_delta[group_valid],
                    flat_ids[group_valid],
                    mask=flat_support[group_valid],
                ).float()
                coefficient_norm = embeddings.square().sum(dim=-1).sqrt()
            flat_positions = torch.arange(
                row_begin * time,
                row_end * time,
                device=ids.device,
            ).reshape(responses, time).reshape(-1)[group_valid]
            return embeddings, flat_positions.long(), count, coefficient_norm

        for prompt in active_prompt_ids:
            embeddings, _, _, _ = collect_prompt(prompt)
            prompt_sum = embeddings.sum(dim=0)
            local_total = local_total + prompt_sum

        global_total = self._distributed_sum_tensor(distributed, local_total)
        global_tokens = self._distributed_sum_int(distributed, local_tokens)
        global_prompts = self._distributed_sum_int(distributed, local_active_prompts)
        fallback = global_prompts < 2 or global_tokens <= 0
        prompt_summaries: list[dict[str, float | int | bool]] = []
        objective_histories: list[list[float]] = []
        prompt_chunk_size = 8
        for chunk_begin in range(0, len(active_prompt_ids), prompt_chunk_size):
            chunk_prompts = active_prompt_ids[
                chunk_begin : chunk_begin + prompt_chunk_size
            ]
            chunk_embeddings: list[torch.Tensor] = []
            chunk_positions: list[torch.Tensor] = []
            chunk_counts: list[int] = []
            chunk_utilities: list[torch.Tensor] = []
            chunk_coefficient_norms: list[torch.Tensor] = []
            for prompt in chunk_prompts:
                embeddings, flat_positions, count, coefficient_norm = collect_prompt(prompt)
                prompt_sum = embeddings.sum(dim=0)
                reference_vector = torch.zeros_like(global_total)
                if not fallback:
                    denominator = global_tokens - count
                    if denominator <= 0:
                        raise AssertionError(
                            "JTA LOPO reference denominator must be positive"
                        )
                    reference_vector = (global_total - prompt_sum) / float(denominator)
                utility = (
                    embeddings @ reference_vector
                    if not fallback
                    else torch.zeros(count, dtype=torch.float32, device=ids.device)
                )
                prompt_references[prompt] = reference_vector
                chunk_embeddings.append(embeddings)
                chunk_positions.append(flat_positions)
                chunk_counts.append(count)
                chunk_utilities.append(utility)
                chunk_coefficient_norms.append(coefficient_norm)

            max_count = max(chunk_counts)
            chunk_size = len(chunk_prompts)
            padded_embeddings = torch.zeros(
                chunk_size,
                max_count,
                self.sketch_dim,
                dtype=torch.float32,
                device=ids.device,
            )
            chunk_mask = torch.zeros(
                chunk_size, max_count, dtype=torch.bool, device=ids.device
            )
            padded_utilities = torch.zeros_like(chunk_mask, dtype=torch.float32)
            for index, (embeddings, utility, count) in enumerate(
                zip(chunk_embeddings, chunk_utilities, chunk_counts)
            ):
                padded_embeddings[index, :count] = embeddings
                chunk_mask[index, :count] = True
                padded_utilities[index, :count] = utility

            if fallback:
                solved = None
            else:
                solved = _solve_prompts_batched(
                    padded_embeddings,
                    padded_utilities,
                    epsilon=self.epsilon,
                    mask=chunk_mask,
                    fw_max_iterations=self.fw_max_iterations,
                    fw_gap_tolerance=self.fw_gap_tolerance,
                    kl_bisection_iterations=self.kl_bisection_iterations,
                    constraint_tolerance=self.constraint_tolerance,
                )

            for index, prompt in enumerate(chunk_prompts):
                embeddings = chunk_embeddings[index]
                flat_positions = chunk_positions[index]
                count = chunk_counts[index]
                coefficient_norm = chunk_coefficient_norms[index]
                if solved is None:
                    probabilities = torch.full(
                        (count,), 1.0 / count, dtype=torch.float32, device=ids.device
                    )
                    utility = torch.zeros_like(probabilities)
                    redundancy = torch.zeros_like(probabilities)
                    marginal = torch.zeros_like(probabilities)
                    summary = {
                        "prompt": prompt,
                        "token_count": count,
                        "initial_objective": 0.0,
                        "final_objective": 0.0,
                        "objective_improvement": 0.0,
                        "linear_term": 0.0,
                        "quadratic_term": 0.0,
                        "fw_iterations": 0,
                        "final_gap": 0.0,
                        "converged": True,
                        "last_gamma": 0.0,
                        "mean_gamma": 0.0,
                        "achieved_kl": 0.0,
                        "kl_boundary": False,
                        "ess_ratio": 1.0,
                        "max_to_mean_weight_ratio": 1.0,
                        "matching_residual": 0.0,
                    }
                    objective_history = [0.0]
                else:
                    probabilities = solved["probabilities"][index, :count]
                    utility = padded_utilities[index, :count]
                    redundancy = solved["redundancy"][index, :count]
                    marginal = solved["marginal"][index, :count]
                    histories = solved["objective_history"]
                    objective_history = [
                        float(history[index].detach().cpu()) for history in histories
                    ]
                    initial_objective = float(
                        solved["initial_objective"][index].detach().cpu()
                    )
                    final_objective = float(
                        solved["final_objective"][index].detach().cpu()
                    )
                    summary = {
                        "prompt": prompt,
                        "token_count": count,
                        "initial_objective": initial_objective,
                        "final_objective": final_objective,
                        "objective_improvement": final_objective - initial_objective,
                        "linear_term": float(
                            solved["linear_term"][index].detach().cpu()
                        ),
                        "quadratic_term": float(
                            solved["quadratic_term"][index].detach().cpu()
                        ),
                        "fw_iterations": int(
                            solved["iterations"][index].detach().cpu()
                        ),
                        "final_gap": float(solved["final_gap"][index].detach().cpu()),
                        "converged": bool(
                            solved["converged"][index].detach().cpu()
                        ),
                        "last_gamma": float(
                            solved["last_gamma"][index].detach().cpu()
                        ),
                        "mean_gamma": float(
                            solved["mean_gamma"][index].detach().cpu()
                        ),
                        "achieved_kl": float(
                            solved["achieved_kl"][index].detach().cpu()
                        ),
                        "kl_boundary": bool(
                            solved["kl_boundary"][index].detach().cpu()
                        ),
                        "ess_ratio": float(
                            1.0
                            / probabilities.square().sum().clamp_min(1e-12)
                            / count
                        ),
                        "max_to_mean_weight_ratio": float(
                            probabilities.max() * count
                        ),
                        "matching_residual": float(
                            solved["final_gap"][index].detach().cpu()
                        ),
                    }
                prompt_summaries.append(summary)
                objective_histories.append(objective_history)
                output_probabilities.view(-1).index_copy_(0, flat_positions, probabilities)
                output_weights.view(-1).index_copy_(
                    0, flat_positions, probabilities * count
                )
                utility_output.view(-1).index_copy_(0, flat_positions, utility)
                redundancy_output.view(-1).index_copy_(0, flat_positions, redundancy)
                marginal_output.view(-1).index_copy_(0, flat_positions, marginal)
                embedding_norm_output.view(-1).index_copy_(
                    0, flat_positions, embeddings.square().sum(dim=-1).sqrt()
                )
                coefficient_norm_output.view(-1).index_copy_(
                    0, flat_positions, coefficient_norm
                )
                if context_backend:
                    row_begin, row_end = prompt * responses, (prompt + 1) * responses
                    hidden_state_norm_output.view(-1).index_copy_(
                        0,
                        flat_positions,
                        hidden_state_norm[row_begin:row_end]
                        .reshape(-1)[
                            objective_valid[row_begin:row_end].reshape(-1)
                        ],
                    )

        if not torch.isfinite(embedding_norm_output[objective_valid]).all():
            raise FloatingPointError("JTA embeddings are non-finite")
        if context_backend and not torch.isfinite(hidden_state_norm_output[objective_valid]).all():
            raise FloatingPointError("JTA hidden-state norms are non-finite")
        for prompt in active_prompt_ids:
            row_begin, row_end = prompt * responses, (prompt + 1) * responses
            prompt_valid = objective_valid[row_begin:row_end]
            prompt_probabilities = output_probabilities[row_begin:row_end][prompt_valid]
            prompt_weights = output_weights[row_begin:row_end][prompt_valid]
            if not torch.isclose(
                prompt_probabilities.sum(),
                torch.ones((), dtype=torch.float32, device=ids.device),
                atol=1e-5,
            ):
                raise AssertionError("JTA prompt probability simplex is violated")
            expected_mass = torch.tensor(
                float(prompt_probabilities.numel()), dtype=torch.float32, device=ids.device
            )
            if not torch.isclose(prompt_weights.sum(), expected_mass, atol=1e-4):
                raise AssertionError("JTA prompt multiplier budget is violated")
            prompt_kl = (
                prompt_probabilities
                * (prompt_probabilities.clamp_min(1e-12) * prompt_probabilities.numel()).log()
            ).sum()
            if float(prompt_kl) > self.epsilon + 1e-4:
                raise AssertionError("JTA prompt KL constraint is violated")
        if bool((output_weights[objective_valid] < 0).any()):
            raise AssertionError("JTA produced negative valid multipliers")
        if not torch.isfinite(output_weights).all():
            raise FloatingPointError("JTA multipliers are non-finite")
        with torch.inference_mode(False):
            weights = output_weights.detach().clone()
            probabilities = output_probabilities.detach().clone()
            diagnostics: dict[str, Any] = {
                "utility": utility_output.detach().clone(),
                "redundancy": redundancy_output.detach().clone(),
                "marginal_score": marginal_output.detach().clone(),
                "embedding_norm": embedding_norm_output.detach().clone(),
                "tensorsketch_embedding_norm": embedding_norm_output.detach().clone(),
                "coefficient_norm": coefficient_norm_output.detach().clone(),
                "hidden_state_norm": hidden_state_norm_output.detach().clone(),
                "w_probability": probabilities,
                "w": weights,
                "reference": prompt_references.detach().clone(),
                "reference_norm": prompt_references.square().sum(dim=-1).sqrt(),
                "prompt_token_counts": prompt_counts.detach().clone(),
                "prompt_summaries": prompt_summaries,
                "objective_history": objective_histories,
                "reference_scope": "active_batch_leave_one_prompt_out",
                "allocation_scope": "independent_within_prompt",
                "candidate_scope": "all_rollouts_of_prompt",
                "prior": "uniform_valid_tokens_within_prompt",
                "embedding_backend": self.embedding_backend,
                "sketch_dim": self.sketch_dim,
                "sketch_seed": self.sketch_seed,
                "vocab_hash_seed": self.sketch_seed,
                "hidden_hash_seed": self.hidden_sketch_seed,
                "fixed_lambda": 1.0,
                "reference_fallback": bool(fallback),
                "active_prompt_count": global_prompts,
                "valid_token_count": global_tokens,
                "all_response_tokens_supervised": True,
            }
        return JTAOutput(weights, probabilities, diagnostics)


__all__ = [
    "CountSketchOperator",
    "JTAOutput",
    "JTASelector",
    "TensorSketchOperator",
    "kl_ball_linear_oracle",
]
