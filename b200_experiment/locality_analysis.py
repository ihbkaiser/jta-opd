from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch


def _detached_float(value: torch.Tensor) -> torch.Tensor:
    return value.detach().float()


@torch.inference_mode()
def reverse_kl_on_fixed_support(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    support_mask: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return ``KL(student || teacher)`` after conditioning on fixed candidates."""
    if student_log_probs.shape != teacher_log_probs.shape:
        raise ValueError("Student and teacher support tensors must have the same shape")
    if support_mask.shape != student_log_probs.shape:
        raise ValueError("support_mask must align with candidate log-probabilities")
    if student_log_probs.ndim < 2:
        raise ValueError("Candidate log-probabilities require a support dimension")
    support = support_mask.detach().bool()
    state_valid = (
        torch.ones(support.shape[:-1], dtype=torch.bool, device=support.device)
        if valid_mask is None
        else valid_mask.detach().bool()
    )
    if state_valid.shape != support.shape[:-1]:
        raise ValueError("valid_mask must align with state dimensions")
    empty = support.sum(dim=-1).eq(0)
    if bool((empty & state_valid).any()):
        raise ValueError("Every valid state must retain at least one support candidate")
    # Padding rows have no support by construction. Give them a temporary
    # singleton so normalization stays finite, then zero them below.
    if bool(empty.any()):
        support = support.clone()
        support[..., 0] |= empty
    student = _detached_float(student_log_probs).masked_fill(~support, -torch.inf)
    teacher = _detached_float(teacher_log_probs).masked_fill(~support, -torch.inf)
    student = student - torch.logsumexp(student, dim=-1, keepdim=True)
    teacher = teacher - torch.logsumexp(teacher, dim=-1, keepdim=True)
    probabilities = torch.where(support, student.exp(), torch.zeros_like(student))
    terms = torch.where(
        support,
        probabilities * (student - teacher),
        torch.zeros_like(probabilities),
    )
    result = terms.sum(dim=-1)
    if valid_mask is not None:
        result = torch.where(state_valid, result, torch.zeros_like(result))
    return result.detach()


@torch.inference_mode()
def realized_self_gain(
    reverse_kl_pre: torch.Tensor, reverse_kl_post: torch.Tensor
) -> torch.Tensor:
    if reverse_kl_pre.shape != reverse_kl_post.shape:
        raise ValueError("Pre/post reverse-KL tensors must have the same shape")
    return (_detached_float(reverse_kl_pre) - _detached_float(reverse_kl_post)).detach()


@torch.inference_mode()
def assign_global_quantile_bins(
    values: torch.Tensor, bins: int = 10
) -> tuple[torch.Tensor, torch.Tensor]:
    """Assign values using global quantile edges while keeping exact ties together."""
    flat = _detached_float(values).reshape(-1)
    if flat.numel() == 0:
        raise ValueError("Cannot bin an empty tensor")
    if not bool(torch.isfinite(flat).all()):
        raise ValueError("Quantile values must be finite")
    count = int(bins)
    if count <= 0:
        raise ValueError("bins must be positive")
    quantiles = torch.linspace(0.0, 1.0, count + 1, device=flat.device)
    edges = torch.quantile(flat, quantiles)
    assignments = torch.bucketize(flat, edges[1:-1], right=False).clamp_max(count - 1)
    return assignments.reshape(values.shape).detach(), edges.detach()


@torch.inference_mode()
def finite_horizon_successor_gain(
    realized_gain: torch.Tensor,
    valid_mask: torch.Tensor,
    horizons: Sequence[int],
    *,
    gamma: float = 1.0,
) -> dict[int, torch.Tensor]:
    """Sum future (never current) realized gains along each valid trajectory."""
    if realized_gain.ndim != 2 or valid_mask.shape != realized_gain.shape:
        raise ValueError("realized_gain and valid_mask must be aligned [batch, time]")
    requested = [int(horizon) for horizon in horizons]
    if not requested or any(horizon <= 0 for horizon in requested):
        raise ValueError("horizons must contain positive integers")
    if requested != sorted(set(requested)):
        raise ValueError("horizons must be strictly increasing")
    discount = float(gamma)
    if not math.isfinite(discount) or discount < 0.0:
        raise ValueError("gamma must be finite and non-negative")
    gains = _detached_float(realized_gain)
    valid = valid_mask.detach().bool()
    width = gains.shape[1]
    outputs: dict[int, torch.Tensor] = {}
    if discount == 1.0:
        masked = torch.where(valid, gains, torch.zeros_like(gains))
        prefix = torch.cat(
            (torch.zeros_like(masked[:, :1]), masked.cumsum(dim=1)), dim=1
        )
        positions = torch.arange(width, device=gains.device)
        lengths = valid.long().sum(dim=1)
        for horizon in requested:
            starts = (positions + 1).unsqueeze(0).expand(gains.shape[0], -1)
            ends = torch.minimum(starts + horizon, lengths.unsqueeze(1))
            starts = torch.minimum(starts, lengths.unsqueeze(1))
            values = prefix.gather(1, ends) - prefix.gather(1, starts)
            outputs[horizon] = torch.where(valid, values, torch.zeros_like(values)).detach()
        return outputs
    for horizon in requested:
        values = torch.zeros_like(gains)
        factor = 1.0
        for offset in range(1, horizon + 1):
            if offset >= width:
                break
            destination = valid[:, :-offset] & valid[:, offset:]
            values[:, :-offset] += torch.where(
                destination,
                gains[:, offset:] * factor,
                torch.zeros_like(gains[:, offset:]),
            )
            factor *= discount
        outputs[horizon] = torch.where(valid, values, torch.zeros_like(values)).detach()
    return outputs


def _average_ranks(values: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(values, stable=True)
    sorted_values = values.index_select(0, order)
    ranks = torch.empty_like(values, dtype=torch.float64)
    begin = 0
    while begin < values.numel():
        end = begin + 1
        while end < values.numel() and bool(sorted_values[end] == sorted_values[begin]):
            end += 1
        ranks.index_fill_(0, order[begin:end], 0.5 * (begin + end - 1))
        begin = end
    return ranks


def _pearson(left: torch.Tensor, right: torch.Tensor) -> float:
    left64 = left.detach().double().reshape(-1)
    right64 = right.detach().double().reshape(-1)
    if left64.numel() < 2:
        return float("nan")
    left64 = left64 - left64.mean()
    right64 = right64 - right64.mean()
    denominator = torch.sqrt(left64.square().sum() * right64.square().sum())
    if float(denominator) == 0.0:
        return float("nan")
    return float((left64 * right64).sum() / denominator)


@torch.inference_mode()
def correlation_summary(
    local_gain: torch.Tensor, realized_gain: torch.Tensor
) -> dict[str, float]:
    local = _detached_float(local_gain).reshape(-1)
    realized = _detached_float(realized_gain).reshape(-1)
    if local.shape != realized.shape:
        raise ValueError("Correlation inputs must have the same number of states")
    finite = torch.isfinite(local) & torch.isfinite(realized)
    local, realized = local[finite], realized[finite]
    return {
        "pearson_g_realized_gain": _pearson(local, realized),
        "spearman_g_realized_gain": _pearson(
            _average_ranks(local), _average_ranks(realized)
        ),
    }


def _quantile(values: torch.Tensor, probability: float) -> float:
    return float(torch.quantile(values.double(), float(probability)))


@torch.inference_mode()
def decile_summary(
    local_gain: torch.Tensor,
    reverse_kl_pre: torch.Tensor,
    reverse_kl_post: torch.Tensor,
    realized_gain: torch.Tensor,
    assignments: torch.Tensor | None = None,
    *,
    bins: int = 10,
) -> dict[str, Any]:
    local = _detached_float(local_gain).reshape(-1)
    pre = _detached_float(reverse_kl_pre).reshape(-1)
    post = _detached_float(reverse_kl_post).reshape(-1)
    realized = _detached_float(realized_gain).reshape(-1)
    if not (local.shape == pre.shape == post.shape == realized.shape):
        raise ValueError("Decile inputs must align")
    if assignments is None:
        assignments, _ = assign_global_quantile_bins(local, bins=bins)
    assigned = assignments.detach().long().reshape(-1)
    if assigned.shape != local.shape:
        raise ValueError("assignments must align with decile inputs")
    rows: list[dict[str, Any]] = []
    means: list[tuple[int, float]] = []
    for index in range(int(bins)):
        mask = assigned.eq(index)
        values = realized[mask]
        if values.numel() == 0:
            rows.append({"decile": index + 1, "count": 0})
            continue
        gain_values = local[mask]
        row = {
            "decile": index + 1,
            "count": int(values.numel()),
            "g_mean": float(gain_values.double().mean()),
            "g_median": float(gain_values.double().median()),
            "kl_pre_mean": float(pre[mask].double().mean()),
            "kl_post_mean": float(post[mask].double().mean()),
            "realized_gain_mean": float(values.double().mean()),
            "realized_gain_median": float(values.double().median()),
            "realized_gain_q10": _quantile(values, 0.1),
            "realized_gain_q90": _quantile(values, 0.9),
            "positive_fraction": float(values.gt(0).double().mean()),
        }
        rows.append(row)
        means.append((index + 1, row["realized_gain_mean"]))
    if len(means) >= 2:
        x = torch.tensor([item[0] for item in means], dtype=torch.float64)
        y = torch.tensor([item[1] for item in means], dtype=torch.float64)
        slope = float(((x - x.mean()) * (y - y.mean())).sum() / (x - x.mean()).square().sum())
        monotonic = sum(
            float(means[index + 1][1] >= means[index][1])
            for index in range(len(means) - 1)
        ) / (len(means) - 1)
        end_difference = means[-1][1] - means[0][1]
        start = means[0][1]
        end_ratio = end_difference if start == 0.0 else means[-1][1] / start
    else:
        slope = monotonic = end_difference = end_ratio = float("nan")
    return {
        "deciles": rows,
        "decile_mean_slope": slope,
        "adjacent_monotonic_fraction": monotonic,
        "q10_to_q1_gain_difference": end_difference,
        "q10_to_q1_gain_ratio": end_ratio,
    }


@torch.inference_mode()
def conditional_future_summary(
    local_gain: torch.Tensor,
    future_gains: Mapping[int, torch.Tensor],
    *,
    conditioning_quantiles: tuple[float, float] = (0.4, 0.6),
    main_horizon: int = 32,
    minimum_count: int = 32,
) -> dict[str, Any]:
    local = _detached_float(local_gain).reshape(-1)
    low, high = map(float, conditioning_quantiles)
    if not 0.0 <= low < high <= 1.0:
        raise ValueError("conditioning quantiles must satisfy 0 <= low < high <= 1")
    if int(main_horizon) not in future_gains:
        raise ValueError("main_horizon must be present in future_gains")

    def band_mask(bounds: tuple[float, float]) -> torch.Tensor:
        lower, upper = torch.quantile(
            local.double(), torch.tensor(bounds, device=local.device, dtype=torch.float64)
        )
        return local.ge(lower) & local.le(upper)

    band = (low, high)
    mask = band_mask(band)
    if int(mask.sum()) < int(minimum_count) and band == (0.4, 0.6):
        band = (0.35, 0.65)
        mask = band_mask(band)
    summary: dict[str, Any] = {
        "conditioning_band": [band[0], band[1]],
        "count": int(mask.sum()),
    }
    for horizon, tensor in sorted(future_gains.items()):
        values = _detached_float(tensor).reshape(-1)
        if values.shape != local.shape:
            raise ValueError("Every future-gain tensor must align with local_gain")
        selected = values[mask]
        prefix = f"horizon_{int(horizon)}"
        if selected.numel() == 0:
            for field in ("mean", "std", "p10", "p25", "p50", "p75", "p90", "iqr", "p90_minus_p10", "fraction_positive", "fraction_negative"):
                summary[f"{prefix}_{field}"] = float("nan")
            continue
        quantiles = torch.quantile(
            selected.double(),
            torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9], dtype=torch.float64, device=selected.device),
        )
        p10, p25, p50, p75, p90 = map(float, quantiles)
        summary.update(
            {
                f"{prefix}_mean": float(selected.double().mean()),
                f"{prefix}_std": float(selected.double().std(unbiased=False)),
                f"{prefix}_p10": p10,
                f"{prefix}_p25": p25,
                f"{prefix}_p50": p50,
                f"{prefix}_p75": p75,
                f"{prefix}_p90": p90,
                f"{prefix}_iqr": p75 - p25,
                f"{prefix}_p90_minus_p10": p90 - p10,
                f"{prefix}_fraction_positive": float(selected.gt(0).double().mean()),
                f"{prefix}_fraction_negative": float(selected.lt(0).double().mean()),
            }
        )
    return summary


@torch.inference_mode()
def build_matched_pairs(
    local_gain: torch.Tensor,
    future_gain: torch.Tensor,
    metadata: Sequence[Mapping[str, Any]],
    *,
    candidate_ids: torch.Tensor | None = None,
    max_pairs: int = 8,
    percentile_bin_width: float = 2.0,
    normalized_g_tolerance: float = 0.02,
) -> list[dict[str, Any]]:
    """Select deterministic low/high-future examples inside narrow gain ranks."""
    local = _detached_float(local_gain).reshape(-1).cpu()
    future = _detached_float(future_gain).reshape(-1).cpu()
    if local.shape != future.shape or len(metadata) != local.numel():
        raise ValueError("Matched-pair inputs must contain one aligned row per state")
    if max_pairs <= 0 or local.numel() < 2:
        return []
    width = float(percentile_bin_width)
    if not 0.0 < width <= 100.0:
        raise ValueError("percentile_bin_width must be in (0, 100]")
    order = torch.argsort(local, stable=True)
    percentile = torch.empty(local.numel(), dtype=torch.float64)
    percentile[order] = (
        torch.arange(local.numel(), dtype=torch.float64) + 0.5
    ) * (100.0 / local.numel())
    bin_ids = torch.floor(percentile / width).long()
    span = max(float(local.max() - local.min()), 1e-12)
    frozen_ids = candidate_ids.detach().cpu() if candidate_ids is not None else None

    def record(index: int) -> dict[str, Any]:
        item = dict(metadata[index])
        item.setdefault("flat_index", index)
        item.update(
            g=float(local[index]),
            future_gain=float(future[index]),
        )
        if frozen_ids is not None:
            item["candidate_ids"] = frozen_ids[index].tolist()
        return item

    candidates: list[tuple[float, int, int, int]] = []
    for bin_id in sorted(set(bin_ids.tolist())):
        indices = torch.nonzero(bin_ids.eq(bin_id), as_tuple=False).squeeze(-1)
        if indices.numel() < 2:
            continue
        ranked = sorted(indices.tolist(), key=lambda index: (float(future[index]), index))
        low_index, high_index = ranked[0], ranked[-1]
        gap = abs(float(local[high_index] - local[low_index])) / span
        if gap <= float(normalized_g_tolerance):
            candidates.append(
                (float(future[high_index] - future[low_index]), bin_id, low_index, high_index)
            )
    candidates.sort(key=lambda item: (-item[0], item[1], item[2], item[3]))
    return [
        {
            "percentile_bin": int(bin_id),
            "normalized_g_gap": abs(float(local[high] - local[low])) / span,
            "future_gain_gap": spread,
            "low_future": record(low),
            "high_future": record(high),
        }
        for spread, bin_id, low, high in candidates[: int(max_pairs)]
    ]
