from __future__ import annotations

from pathlib import Path
from typing import Any


BASE_TAGS = {
    "train/loss": "train_loss",
    "train/grad_norm": "grad_norm",
    "train/learning_rate": "lr",
    "train/global_step": "step",
    "distillation/topk_opd_loss": "base_topk_opd_loss",
    "distillation/student_entropy": "student_entropy",
    "distillation/teacher_entropy": "teacher_entropy",
    "distillation/entropy_gap": "entropy_gap",
    "distillation/topk_overlap_ratio": "student_teacher_topk_overlap_ratio",
    "distillation/topk_student_mass": "topk_student_mass",
    "distillation/topk_teacher_mass": "topk_teacher_mass",
    "distillation/topk_divergence_proxy_mean": "topk_divergence_proxy_mean",
    "distillation/topk_divergence_proxy_min": "topk_divergence_proxy_min",
    "distillation/topk_divergence_proxy_max": "topk_divergence_proxy_max",
    "optimization/ratio_mean": "opd_ratio_mean",
    "optimization/ratio_min": "opd_ratio_min",
    "optimization/ratio_max": "opd_ratio_max",
    "optimization/clip_fraction": "opd_clip_fraction",
    "optimization/advantage_mean": "opd_advantage_mean",
    "optimization/advantage_abs_mean": "opd_advantage_abs_mean",
    "optimization/advantage_min": "opd_advantage_min",
    "optimization/advantage_max": "opd_advantage_max",
    "opd/advantage_abs_mean": "opd_advantage_abs_mean",
    "opd/teacher_student_logprob_gap": "opd_teacher_student_logprob_gap",
    "opd/clip_fraction": "opd_clip_fraction",
    "rollout/response_length_mean": "mean_response_length",
    "rollout/response_length_min": "min_response_length",
    "rollout/response_length_max": "max_response_length",
    "rollout/response_clip_ratio": "response_clip_ratio",
    "rollout/tokens_per_second": "generated_tokens_per_second",
    "rollout/eos_fraction": "eos_fraction",
    "system/step_time": "wall_clock_step_time",
    "system/step_time_sec": "wall_clock_step_time",
    "system/rollout_time": "rollout_time",
    "system/tokens_per_second": "tokens_per_second",
    "system/peak_vram_gb": "peak_gpu_allocated_gb",
    "system/gpu_memory_allocated_gb": "peak_gpu_allocated_gb",
    "system/gpu_memory_reserved_gb": "peak_gpu_reserved_gb",
}

TA_TAGS = {
    "ta/D_mean": ("D", "mean"),
    "ta/C_mean": ("C", "mean"),
    "ta/teachability_mean": ("s_TA", "mean"),
    "ta/teachability_std": ("s_TA", "std"),
    "ta/selected_fraction": ("selected_fraction",),
}

RAC_TAGS = {
    "rac/local_teachability_mean": ("g", "mean"),
    "rac/alignment_mean": ("alignment", "mean"),
    "rac/V_mean": ("V", "mean"),
    "rac/V_std": ("V", "std"),
    "rac/weight_mean": ("w", "mean"),
    "rac/weight_std": ("w", "std"),
    "rac/weight_min": ("w", "min"),
    "rac/weight_max": ("w", "max"),
}

PGT_TAGS = {
    "pgt/gain_mean": ("gain", "mean"),
    "pgt/gain_std": ("gain", "std"),
    "pgt/euclidean_gain_mean": ("euclidean_gain", "mean"),
    "pgt/teacher_union_mass_mean": ("teacher_union_mass", "mean"),
    "pgt/teacher_tail_mass_mean": ("teacher_tail_mass", "mean"),
    "pgt/restricted_reverse_kl_mean": ("restricted_reverse_kl", "mean"),
    "pgt/selected_fraction": ("selected_fraction",),
}

CMT_TAGS = {
    "cmt/local_pgt_mean": ("gain", "mean"),
    "cmt/common_mass_mean": ("support_common_mass", "mean"),
    "cmt/conditional_common_mass_mean": (
        "conditional_support_common_mass",
        "mean",
    ),
    "cmt/sample_acceptance_mean": ("alignment", "mean"),
    "cmt/transition_weight_mean": ("transition_weight", "mean"),
    "cmt/support_coverage_mean": ("support_coverage", "mean"),
    "cmt/teacher_deficit_rate": ("teacher_deficit", "mean"),
    "cmt/successor_return_mean": ("R", "mean"),
    "cmt/local_excess_mean": ("H", "mean"),
    "cmt/sequential_gain_mean": ("sequential_gain", "mean"),
    "cmt/learning_value_mean": ("learning_value", "mean"),
    "cmt/learning_value_std": ("learning_value", "std"),
    "cmt/weight_mean": ("w", "mean"),
    "cmt/weight_std": ("w", "std"),
    "cmt/weight_max": ("w", "max"),
    "cmt/marginal_flux_mean": ("marginal_flux", "mean"),
    "cmt/marginal_flux_std": ("marginal_flux", "std"),
    "cmt/flux_product_form_mean": ("flux_product_form", "mean"),
    "cmt/flux_identity_error_mean": ("flux_identity_error", "mean"),
    "cmt/successor_mass_mean": ("successor_mass", "mean"),
    "cmt/successor_value_mean": ("successor_value", "mean"),
    "cmt/successor_contrast_mean": ("successor_contrast", "mean"),
    "cmt/successor_excess_mean": ("successor_excess", "mean"),
    "cmt/successor_excess_std": ("successor_excess", "std"),
    "cmt/x_cancellation_ratio_q99": ("x_cancellation_ratio", "q99"),
    "cmt/sequential_gain_abs_mean": ("abs_sequential_gain", "mean"),
    "cmt/selected_count": ("selected_tokens",),
    "cmt/valid_token_count": ("valid_tokens",),
    "cmt/allocation_kl_epsilon": ("allocation_kl_epsilon",),
    "cmt/allocation_kl_achieved": ("allocation_kl_achieved",),
    "cmt/allocation_inverse_temperature": ("allocation_inverse_temperature",),
    "cmt/allocation_top_fraction": ("allocation_top_fraction",),
    "cmt/allocation_threshold": ("allocation_threshold",),
    "cmt/weight_max_scalar": ("w_max",),
    "cmt/max_token_probability": ("max_token_probability",),
    "cmt/top_weight_mass_top1": ("top_weight_mass_top1",),
    "cmt/top_weight_mass_top10": ("top_weight_mass_top10",),
    "cmt/top_weight_mass_top100": ("top_weight_mass_top100",),
    "cmt/top_weight_mass_top0p1": ("top_weight_mass_top0p1",),
    "cmt/fraction_w_lt_1e-6": ("fraction_w_lt_1e-6",),
    "cmt/fraction_w_gt_10": ("fraction_w_gt_10",),
    "cmt/fraction_w_gt_100": ("fraction_w_gt_100",),
    "cmt/fraction_w_gt_1000": ("fraction_w_gt_1000",),
    "cmt/normalized_ess": ("normalized_ess",),
    "cmt/normalized_weight_entropy": ("normalized_weight_entropy",),
}

GRPO_TAGS = {
    "grpo/reward_mean": ("grpo_reward_mean",),
    "grpo/reward_std": ("grpo_reward_std",),
    "grpo/reward_min": ("grpo_reward_min",),
    "grpo/reward_max": ("grpo_reward_max",),
    "grpo/advantage_mean": ("grpo_advantage_mean",),
    "grpo/advantage_std": ("grpo_advantage_std",),
    "grpo/group_size": ("grpo_group_size",),
    "grpo/clip_fraction": ("grpo_clip_fraction",),
}

IW_TAGS = {
    "iw/weight_mean": ("iw_weight_mean",),
    "iw/weight_std": ("iw_weight_std",),
    "iw/weight_min": ("iw_weight_min",),
    "iw/weight_max": ("iw_weight_max",),
}


def _selector_value(selector: dict[str, Any], path: tuple[str, ...]) -> float:
    value: Any = selector
    for key in path:
        value = value[key]
    return float(value)


def _selector_has(selector: dict[str, Any], path: tuple[str, ...]) -> bool:
    value: Any = selector
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return False
        value = value[key]
    return value is not None


def production_tensorboard_metrics(
    metrics: dict[str, Any], method: str
) -> dict[str, float]:
    """Select globally reduced production diagnostics for TensorBoard."""
    selected = {
        tag: float(metrics[field])
        for tag, field in BASE_TAGS.items()
        if field in metrics and metrics[field] is not None
    }
    selector = metrics.get("selector", {})
    if method == "ta":
        selected["ta/selected_token_fraction"] = float(selector["selected_fraction"])
        selected.update(
            {
                tag: _selector_value(selector, path)
                for tag, path in TA_TAGS.items()
            }
        )
    elif method == "rac":
        valid_tokens = max(int(selector["valid_tokens"]), 1)
        selected["rac/effective_token_fraction"] = float(
            selector["effective_sample_size"] / valid_tokens
        )
        selected.update(
            {
                tag: _selector_value(selector, path)
                for tag, path in RAC_TAGS.items()
            }
        )
    elif method == "pgt":
        selected["pgt/selected_token_fraction"] = float(
            selector["selected_fraction"]
        )
        selected.update(
            {
                tag: _selector_value(selector, path)
                for tag, path in PGT_TAGS.items()
            }
        )
    elif method == "cmt":
        valid_tokens = max(int(selector["valid_tokens"]), 1)
        selected["cmt/effective_token_fraction"] = float(
            selector["effective_sample_size"] / valid_tokens
        )
        selected.update(
            {
                tag: _selector_value(selector, path)
                for tag, path in CMT_TAGS.items()
                if _selector_has(selector, path)
            }
        )
        for tag, field in {
            "cmt/influence_weighted_loss_mean": "weighted_loss_contribution_mean",
            "cmt/influence_weighted_loss_abs_mean": "weighted_loss_abs_mean",
            "cmt/influence_abs_top1_fraction": "weighted_loss_abs_top1_fraction",
            "cmt/influence_abs_top10_fraction": "weighted_loss_abs_top10_fraction",
            "cmt/influence_abs_top100_fraction": "weighted_loss_abs_top100_fraction",
            "cmt/influence_abs_top0p1_fraction": "weighted_loss_abs_top0p1_fraction",
            "cmt/influence_top_positive_d_abs_fraction": "weighted_loss_top_positive_d_abs_fraction",
            "cmt/influence_bottom_negative_d_abs_fraction": "weighted_loss_bottom_negative_d_abs_fraction",
            "cmt/influence_top_abs_d_abs_fraction": "weighted_loss_top_abs_d_abs_fraction",
        }.items():
            if field in metrics and metrics[field] is not None:
                selected[tag] = float(metrics[field])
        selected["cmt/allocation_mode_top_fraction"] = float(
            selector.get("allocation_mode") == "top_fraction"
        )
    elif method == "grpo":
        selected.update(
            {
                tag: _selector_value(metrics, path)
                for tag, path in GRPO_TAGS.items()
                if all(key in metrics for key in path)
            }
        )
    elif method == "iw":
        selected.update(
            {
                tag: _selector_value(metrics, path)
                for tag, path in IW_TAGS.items()
                if all(key in metrics and metrics[key] is not None for key in path)
            }
        )
    sanity = metrics.get("vllm_logprob_sanity", {})
    if bool(sanity.get("enabled", False)):
        selected["debug/vllm_hf_logprob_mae"] = float(sanity["mean_abs_error"])
    return selected


class TensorBoardLogger:
    def __init__(
        self,
        output_dir: Path,
        settings: dict[str, Any],
        *,
        enabled: bool,
        resume_step: int,
    ):
        self.writer = None
        self.interval = max(1, int(settings.get("log_interval", 1)))
        if not enabled or not bool(settings.get("enabled", True)):
            return
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as error:
            raise RuntimeError(
                "TensorBoard logging is enabled but tensorboard is not installed; "
                "install requirements.txt"
            ) from error
        configured = settings.get("log_dir", "tensorboard")
        log_dir = Path(configured)
        if not log_dir.is_absolute():
            log_dir = output_dir / log_dir
        self.writer = SummaryWriter(
            log_dir=str(log_dir.resolve()),
            purge_step=(resume_step + 1 if resume_step > 0 else None),
            max_queue=10,
            flush_secs=int(settings.get("flush_secs", 30)),
        )

    def write(self, step: int, metrics: dict[str, Any], method: str) -> None:
        if self.writer is None or int(step) % self.interval != 0:
            return
        for tag, value in production_tensorboard_metrics(metrics, method).items():
            self.writer.add_scalar(tag, value, global_step=int(step))

    def close(self) -> None:
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
