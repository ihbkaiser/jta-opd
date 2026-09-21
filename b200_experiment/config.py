from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    base_name = config.pop("_base_", None)
    if base_name is not None:
        config = deep_merge(load_config(path.parent / base_name), config)
    config["_config_path"] = str(path)
    return config


def load_with_overlays(
    path: str | Path, overlays: list[str | Path] | None = None
) -> dict[str, Any]:
    config = load_config(path)
    for overlay_path in overlays or []:
        overlay = load_config(overlay_path)
        overlay.pop("_config_path", None)
        config = deep_merge(config, overlay)
    config["_config_path"] = str(Path(path).resolve())
    config["_overlays"] = [str(Path(item).resolve()) for item in overlays or []]
    return config


def apply_overrides(config: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    result = copy.deepcopy(config)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid override {item!r}; expected dotted.key=value")
        dotted_key, raw_value = item.split("=", 1)
        value = yaml.safe_load(raw_value)
        target = result
        parts = dotted_key.split(".")
        for part in parts[:-1]:
            target = target.setdefault(part, {})
            if not isinstance(target, dict):
                raise ValueError(
                    f"Cannot set {dotted_key!r}: {part!r} is not a mapping"
                )
        target[parts[-1]] = value
    return result


def resolve_runtime_paths(config: dict[str, Any]) -> dict[str, Any]:
    """Resolve all B200 assets from one configurable storage root."""
    result = copy.deepcopy(config)
    root = Path(
        result.get("paths", {}).get("storage_root", "/workspace/storage-shared")
    )

    def resolved(value: str) -> str:
        path = Path(value).expanduser()
        return str((path if path.is_absolute() else root / path).resolve())

    for key in ("teacher_path", "student_path"):
        result["models"][key] = resolved(result["models"][key])
    result["data"]["path"] = resolved(result["data"]["path"])
    for benchmark in result.get("evaluation", {}).get("benchmarks", {}).values():
        benchmark["path"] = resolved(benchmark["path"])
    result.setdefault("paths", {})["storage_root"] = str(root.resolve())
    return result


def save_config(config: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {
        key: value for key, value in config.items() if not key.startswith("_")
    }
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(serializable, handle, sort_keys=False, allow_unicode=True)


def validate_locality_probe_config(config: dict[str, Any]) -> dict[str, Any]:
    """Validate and materialize the opt-in CMT locality protocol."""
    defaults: dict[str, Any] = {
        "enabled": False,
        "training_allocation": "uniform",
        "g_bins": 10,
        "future_horizons": [8, 16, 32],
        "future_gamma": 1.0,
        "main_future_horizon": 32,
        "conditioning_quantiles": [0.4, 0.6],
        "token_sample_size": 2048,
        "matched_pairs_per_step": 8,
        "fail_on_prompt_overlap": True,
        "other_id": {
            "enabled": True,
            "benchmark_name": "Competition-MATH",
            "num_prompts": 32,
            "max_new_tokens": 2048,
            "seed": 20260921,
            "fixed_support": True,
        },
    }
    configured = config.get("analysis", {}).get("locality_probe", {})
    resolved = deep_merge(defaults, configured)
    if not bool(resolved["enabled"]):
        return resolved
    if str(config.get("experiment", {}).get("method", "")).lower() != "cmt":
        raise ValueError("analysis.locality_probe requires experiment.method=cmt")
    if str(resolved["training_allocation"]).lower() != "uniform":
        raise ValueError("analysis.locality_probe only supports uniform training allocation")
    if int(config.get("rollout", {}).get("num_responses", 1)) != 1:
        raise ValueError("analysis.locality_probe requires rollout.num_responses=1")
    rollout_size = int(config.get("rollout", {}).get("batch_size", 0))
    ppo_size = int(config.get("training", {}).get("ppo_mini_batch_size", rollout_size))
    if rollout_size <= 0 or ppo_size != rollout_size:
        raise ValueError(
            "analysis.locality_probe requires one fresh rollout per optimizer step: "
            "training.ppo_mini_batch_size must equal rollout.batch_size"
        )
    bins = int(resolved["g_bins"])
    if bins < 2:
        raise ValueError("analysis.locality_probe.g_bins must be at least 2")
    horizons = [int(value) for value in resolved["future_horizons"]]
    if not horizons or horizons != sorted(set(horizons)) or horizons[0] <= 0:
        raise ValueError("analysis.locality_probe.future_horizons must be positive and strictly increasing")
    resolved["future_horizons"] = horizons
    if int(resolved["main_future_horizon"]) not in horizons:
        raise ValueError("analysis.locality_probe.main_future_horizon must be listed in future_horizons")
    gamma = float(resolved["future_gamma"])
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("analysis.locality_probe.future_gamma must lie in [0, 1]")
    quantiles = [float(value) for value in resolved["conditioning_quantiles"]]
    if len(quantiles) != 2 or not 0.0 <= quantiles[0] < quantiles[1] <= 1.0:
        raise ValueError("analysis.locality_probe.conditioning_quantiles must satisfy 0 <= low < high <= 1")
    resolved["conditioning_quantiles"] = quantiles
    if int(resolved["token_sample_size"]) < 0 or int(resolved["matched_pairs_per_step"]) < 0:
        raise ValueError("Locality sample limits must be non-negative")
    other = resolved["other_id"]
    benchmark = str(other.get("benchmark_name", "Competition-MATH"))
    benchmarks = config.get("evaluation", {}).get("benchmarks", {})
    if bool(other.get("enabled", True)) and benchmark not in benchmarks:
        raise ValueError(f"Locality OTHER-ID benchmark {benchmark!r} is not configured")
    if bool(other.get("enabled", True)) and not bool(other.get("fixed_support", False)):
        raise ValueError("analysis.locality_probe.other_id requires fixed_support=true")
    if int(other.get("num_prompts", 0)) <= 0 or int(other.get("max_new_tokens", 0)) <= 0:
        raise ValueError("Locality OTHER-ID prompt count and max_new_tokens must be positive")
    return resolved
