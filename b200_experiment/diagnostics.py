from __future__ import annotations

from typing import Any

import torch

from .selectors import top_budget_mask


def _pearson(x: torch.Tensor, y: torch.Tensor) -> float:
    x, y = x.float(), y.float()
    if x.numel() < 2:
        return float("nan")
    x, y = x - x.mean(), y - y.mean()
    denominator = x.square().sum().sqrt() * y.square().sum().sqrt()
    return (
        float("nan")
        if denominator.item() == 0
        else float((x * y).sum().div(denominator).item())
    )


def _ranks(x: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(x, stable=True)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(order.numel(), dtype=torch.float32, device=x.device)
    return ranks


def correlations(ta_scores, rac_diagnostics, valid_mask) -> dict[str, float]:
    ta = ta_scores[valid_mask]
    result = {}
    for name in ("g", "V", "w"):
        target = rac_diagnostics[name][valid_mask]
        result[f"pearson_sTA_{name}"] = _pearson(ta, target)
        result[f"spearman_sTA_{name}"] = _pearson(_ranks(ta), _ranks(target))
    for ratio in (0.05, 0.10):
        ta_mask = top_budget_mask(ta_scores, valid_mask, ratio)
        rac_mask = top_budget_mask(rac_diagnostics["V"], valid_mask, ratio)
        denominator = max(int(ta_mask.sum().item()), 1)
        result[f"top_{int(ratio * 100)}pct_overlap"] = float(
            (ta_mask & rac_mask).sum().item() / denominator
        )
    return result


def tensor_summary(
    values: torch.Tensor, valid_mask: torch.Tensor | None = None
) -> dict[str, float]:
    if valid_mask is not None and values.shape == valid_mask.shape:
        values = values[valid_mask]
    # FP64 preserves large cancellation diagnostics (for example kappa_X can
    # exceed the FP32 dynamic range when X is close to zero).
    finite = values.detach().to(torch.float64)
    finite = finite[torch.isfinite(finite)]
    if finite.numel() == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            **{
                name: float("nan")
                for name in (
                    "q05", "q25", "q50", "q75", "q90", "q95",
                    "q99", "q99.5", "q99.9", "q99.99",
                )
            },
        }
    quantiles = torch.quantile(
        finite,
        torch.tensor(
            [0.05, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 0.995, 0.999, 0.9999],
            device=finite.device,
            dtype=finite.dtype,
        ),
    )
    return {
        "mean": float(finite.mean()),
        "std": float(finite.std(unbiased=False)),
        "min": float(finite.min()),
        "max": float(finite.max()),
        **{
            name: float(value)
            for name, value in zip(
                (
                    "q05", "q25", "q50", "q75", "q90", "q95",
                    "q99", "q99.5", "q99.9", "q99.99",
                ),
                quantiles,
            )
        },
    }


def selector_summary(
    method: str,
    diagnostics: dict[str, Any],
    valid_mask: torch.Tensor,
    selected: torch.Tensor,
):
    if method == "ta":
        keys = ("D", "C", "D_norm", "C_norm", "s_TA")
    elif method == "rac":
        keys = (
            "g",
            "alignment",
            "R",
            "M",
            "V",
            "z",
            "w",
        )
    elif method == "opd":
        keys = ("w",)
    elif method == "iw":
        keys = ("w", "iw_weight")
    elif method == "pgt":
        keys = (
            "gain",
            "s_PGT",
            "euclidean_gain",
            "restricted_reverse_kl",
            "student_union_mass",
            "teacher_union_mass",
            "teacher_tail_mass",
            "support_width",
        )
    elif method == "cmt":
        keys = (
            "gain",
            "support_reverse_kl",
            "support_common_mass",
            "conditional_support_common_mass",
            "alignment",
            "transition_weight",
            "support_coverage",
            "coverage_correction",
            "teacher_deficit",
            "marginal_flux",
            "flux_product_form",
            "flux_identity_error",
            "common_mass_derivative",
            "R",
            "M",
            "V",
            "H",
            "successor_excess",
            "sequential_gain",
            "learning_value",
            "w",
            "student_union_mass",
            "teacher_union_mass",
            "teacher_tail_mass",
            # Optional full-vocabulary audit fields are present only when
            # cmt_full_vocab_diagnostics is enabled; selector_summary skips
            # absent fields below without changing the training hot path.
            "full_log_ratio_mean",
            "full_log_ratio_variance",
            "full_common_mass",
            "sampled_raw_student_prob",
            "sampled_raw_teacher_prob",
            "sampled_cond_student_prob",
            "sampled_cond_teacher_prob",
            "sampled_log_ratio",
            "sampled_cond_r",
            "sampled_conditional_log_ratio",
            "conditional_log_ratio_mean",
            "successor_return",
            "successor_mass",
            "successor_value",
            "successor_contrast",
            "baseline_mass_term",
            "x_difference_form",
            "x_product_form",
            "x_difference_identity_error",
            "x_product_identity_error",
            "x_cancellation_ratio",
            "d_product_form",
            "d_identity_error",
            "abs_marginal_flux",
            "abs_sequential_gain",
            "abs_d_over_gain",
            "response_position_fraction",
            "successor_excess_fp64",
            "x_fp32_fp64_abs_error",
            "x_fp32_fp64_relative_error",
        )
    else:
        raise ValueError(f"Unknown selector-summary method: {method!r}")
    result: dict[str, Any] = {}
    for key in keys:
        value = diagnostics.get(key)
        if torch.is_tensor(value):
            result[key] = tensor_summary(value, valid_mask)
    valid_count = int(valid_mask.sum().item())
    selected_count = int(selected.sum().item())
    result.update(
        valid_tokens=valid_count,
        selected_tokens=selected_count,
        selected_fraction=selected_count / max(valid_count, 1),
    )
    if method == "ta" and selected_count:
        result["selection_threshold"] = float(diagnostics["s_TA"][selected].min())
    if method == "pgt" and selected_count:
        result["selection_threshold"] = float(diagnostics["s_PGT"][selected].min())
    if method == "cmt" and selected_count and diagnostics.get("allocation_mode") == "top_fraction":
        result["selection_threshold"] = float(
            diagnostics["s_CMT"][selected].min()
        )
    if method in {"opd", "rac", "cmt"} and "w" in diagnostics:
        weights = diagnostics["w"][valid_mask].detach().float()
        weight_sum = weights.sum()
        result.update(
            effective_token_weight_mass=float(weight_sum / max(valid_count, 1)),
            effective_sample_size=float(
                weight_sum.square() / weights.square().sum().clamp_min(1e-12)
            ),
        )
    if method == "cmt":
        for key in (
            "allocation_mode",
            "allocation_inverse_temperature",
            "allocation_kl_epsilon",
            "allocation_kl_achieved",
            "allocation_top_fraction",
            "allocation_threshold",
        ):
            value = diagnostics.get(key)
            if value is not None and not torch.is_tensor(value):
                result[key] = value
        if "w" in diagnostics and valid_count:
            weights = diagnostics["w"][valid_mask].detach().float()
            finite_weights = weights[torch.isfinite(weights) & (weights >= 0)]
            if finite_weights.numel():
                total = finite_weights.sum().clamp_min(1.0e-12)
                ordered = torch.sort(finite_weights, descending=True).values
                result["w_max"] = float(finite_weights.max())
                result["max_token_probability"] = float(finite_weights.max() / total)
                result["top_weight_mass_top1"] = float(ordered[:1].sum() / total)
                result["top_weight_mass_top10"] = float(ordered[:10].sum() / total)
                result["top_weight_mass_top100"] = float(ordered[:100].sum() / total)
                result["top_weight_mass_top0p1"] = float(
                    ordered[: max(1, (finite_weights.numel() + 999) // 1000)].sum()
                    / total
                )
                ess = float(total.square() / finite_weights.square().sum().clamp_min(1.0e-12))
                result["effective_sample_size"] = ess
                result["normalized_ess"] = ess / max(valid_count, 1)
                probabilities = finite_weights / total
                result["normalized_weight_entropy"] = float(
                    -(probabilities * probabilities.clamp_min(1.0e-30).log()).sum()
                )
                result["fraction_w_lt_1e-6"] = float(
                    (finite_weights < 1.0e-6).float().mean()
                )
                result["fraction_w_gt_10"] = float(
                    (finite_weights > 10.0).float().mean()
                )
                result["fraction_w_gt_100"] = float(
                    (finite_weights > 100.0).float().mean()
                )
                result["fraction_w_gt_1000"] = float(
                    (finite_weights > 1000.0).float().mean()
                )
    return result


def finite_or_raise(name: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor).all():
        count = int((~torch.isfinite(tensor)).sum().item())
        raise FloatingPointError(f"{name} contains {count} non-finite values")
