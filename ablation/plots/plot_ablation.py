#!/usr/bin/env python3
"""Plot CMT ablation curves and final scores for one or more benchmarks.

The arm comparison (``--benchmarks``) intentionally uses the same six-panel
layout as the main training-progress plot. A legacy ``--benchmark`` argument
is retained for single-benchmark hyperparameter/debug plots.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


BENCHMARK_ORDER = (
    "Competition-MATH",
    "MATH-500",
    "AIME24",
    "AIME25",
    "GPQA-Diamond",
    "AMC23",
)
ARMS = ("g", "g_x", "g_d")
COLORS = {"g": "tab:blue", "g_x": "tab:orange", "g_d": "tab:purple"}


def _normalise_name(value: str) -> str:
    return value.casefold().replace("-", "").replace("_", "").replace(" ", "")


def _resolve_benchmarks(single: str | None, multiple: str | None) -> tuple[str, ...]:
    """Resolve CLI benchmark selection while preserving canonical ordering."""
    raw = multiple if multiple is not None else single
    if raw is None or raw.strip().casefold() in {"", "all", "*"}:
        return BENCHMARK_ORDER
    requested = [item.strip() for item in raw.replace(",", " ").split() if item.strip()]
    aliases = {_normalise_name(name): name for name in BENCHMARK_ORDER}
    resolved = []
    for item in requested:
        canonical = aliases.get(_normalise_name(item))
        if canonical is None:
            raise ValueError(
                f"Unknown benchmark {item!r}; expected one of: "
                + ", ".join(BENCHMARK_ORDER)
            )
        if canonical not in resolved:
            resolved.append(canonical)
    if not resolved:
        raise ValueError("At least one benchmark must be selected")
    return tuple(name for name in BENCHMARK_ORDER if name in resolved)


def _runs(root: Path, selected: set[str] | None = None):
    for spec_path in sorted(root.rglob("ablation_spec.json")):
        if selected and spec_path.parent.name not in selected:
            continue
        try:
            spec = json.loads(spec_path.read_text(encoding="utf-8"))
            history_path = spec_path.parent / "eval_history.jsonl"
            if history_path.is_file():
                rows = [
                    json.loads(line)
                    for line in history_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                yield spec_path.parent, spec, rows
        except (OSError, json.JSONDecodeError):
            continue


def _benchmark(row: dict, requested: str):
    """Return exactly the requested benchmark; never fall back to another one."""
    values = row.get("benchmarks", {})
    if requested in values:
        return values[requested]
    wanted = _normalise_name(requested)
    for name, result in values.items():
        if _normalise_name(str(name)) == wanted:
            return result
    return None


def _accuracy_ylim(values: list[float]) -> tuple[float, float]:
    """Zoom accuracy axes to the observed range without clipping values."""
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return 0.0, 1.05
    tick, padding = 0.05, 0.02
    lower = max(0.0, tick * math.floor((min(finite) - padding) / tick))
    upper = tick * math.ceil((max(finite) + padding) / tick)
    lower, upper = round(lower, 10), round(upper, 10)
    if upper <= lower:
        upper = round(lower + tick, 10)
    return lower, upper


def _save_pair(fig, output_dir: Path, stem: str) -> None:
    """Never replace an existing figure when a plot command is repeated."""
    index = 0
    while True:
        suffix = "" if index == 0 else f"_{index:03d}"
        png = output_dir / f"{stem}{suffix}.png"
        pdf = output_dir / f"{stem}{suffix}.pdf"
        if not png.exists() and not pdf.exists():
            fig.savefig(png, dpi=180, bbox_inches="tight")
            fig.savefig(pdf, bbox_inches="tight")
            return
        index += 1


def _curve_stats(rows: list[tuple[int, float]]):
    """Aggregate repeated runs of one arm by optimizer step."""
    import numpy as np

    by_step = defaultdict(list)
    for step, value in rows:
        by_step[step].append(value)
    steps = sorted(by_step)
    mean = np.asarray([np.mean(by_step[step]) for step in steps])
    std = np.asarray([np.std(by_step[step]) for step in steps])
    return steps, mean, std


def _make_curve_plot(grouped, benchmarks, metric, output_dir):
    import matplotlib.pyplot as plt
    import numpy as np

    columns = min(3, len(benchmarks))
    rows = math.ceil(len(benchmarks) / columns)
    fig, axes = plt.subplots(
        rows, columns, figsize=(6 * columns, 4.5 * rows), squeeze=False
    )
    axes_flat = axes.ravel()
    legend_handles = {}
    for axis, benchmark in zip(axes_flat, benchmarks):
        axis_values = []
        maximum_step = 0
        for arm in ARMS:
            values = grouped.get(benchmark, {}).get(arm, [])
            if not values:
                continue
            steps, mean, std = _curve_stats(values)
            axis_values.extend(mean.tolist())
            maximum_step = max(maximum_step, max(steps))
            (line,) = axis.plot(
                steps,
                mean,
                label=arm,
                color=COLORS[arm],
                linewidth=2,
                marker="o",
                markersize=3.5,
            )
            legend_handles[arm] = line
            if np.any(std > 0):
                axis.fill_between(
                    steps,
                    mean - std,
                    mean + std,
                    color=COLORS[arm],
                    alpha=0.16,
                )
        if axis_values:
            axis.set_ylim(*_accuracy_ylim(axis_values))
            axis.set_xlim(left=0, right=max(maximum_step, 1))
        else:
            axis.text(
                0.5,
                0.5,
                "No data",
                transform=axis.transAxes,
                ha="center",
                va="center",
            )
        axis.set_title(benchmark)
        axis.set_xlabel("Optimizer step")
        axis.set_ylabel(metric)
        axis.grid(alpha=0.25)
    for axis in axes_flat[len(benchmarks) :]:
        axis.set_visible(False)
    handles = [legend_handles[arm] for arm in ARMS if arm in legend_handles]
    labels = [arm for arm in ARMS if arm in legend_handles]
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(handles), frameon=False)
    fig.suptitle(f"CMT ablation learning curves ({metric})", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save_pair(fig, output_dir, "ablation_curves")
    plt.close(fig)


def _make_final_plot(grouped, benchmarks, metric, output_dir):
    import matplotlib.pyplot as plt
    import numpy as np

    columns = min(3, len(benchmarks))
    rows = math.ceil(len(benchmarks) / columns)
    fig, axes = plt.subplots(
        rows, columns, figsize=(6 * columns, 4.5 * rows), squeeze=False
    )
    axes_flat = axes.ravel()
    legend_handles = {}
    for axis, benchmark in zip(axes_flat, benchmarks):
        means, errors, arms = [], [], []
        for arm in ARMS:
            values = grouped.get(benchmark, {}).get(arm, [])
            if not values:
                continue
            steps, _, _ = _curve_stats(values)
            final_step = max(steps)
            final_values = [value for step, value in values if step == final_step]
            means.append(float(np.mean(final_values)))
            errors.append(float(np.std(final_values)))
            arms.append(arm)
        if arms:
            x = np.arange(len(arms))
            bars = axis.bar(
                x,
                means,
                yerr=errors,
                color=[COLORS[arm] for arm in arms],
                capsize=4,
            )
            for arm, bar in zip(arms, bars):
                legend_handles.setdefault(arm, bar)
            axis.set_xticks(x, arms)
            axis.set_ylim(
                *_accuracy_ylim(
                    means
                    + [mean - error for mean, error in zip(means, errors)]
                    + [mean + error for mean, error in zip(means, errors)]
                )
            )
            axis.bar_label(bars, labels=[f"{value:.3f}" for value in means], padding=3)
        else:
            axis.text(
                0.5,
                0.5,
                "No data",
                transform=axis.transAxes,
                ha="center",
                va="center",
            )
        axis.set_title(benchmark)
        axis.set_ylabel(f"Final {metric}")
        axis.grid(axis="y", alpha=0.25)
    for axis in axes_flat[len(benchmarks) :]:
        axis.set_visible(False)
    handles = [legend_handles[arm] for arm in ARMS if arm in legend_handles]
    labels = [arm for arm in ARMS if arm in legend_handles]
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(handles), frameon=False)
    fig.suptitle(f"Final CMT ablation comparison ({metric})", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save_pair(fig, output_dir, "ablation_final")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--benchmark",
        default=None,
        help="Legacy single benchmark selector; omitted means all six benchmarks",
    )
    parser.add_argument(
        "--benchmarks",
        default=None,
        help="Comma- or space-separated benchmarks for the multi-panel plot",
    )
    parser.add_argument(
        "--metric",
        choices=("accuracy", "avg_at_n", "pass_at_8"),
        default="accuracy",
    )
    parser.add_argument(
        "--run-name",
        action="append",
        dest="run_names",
        help="Restrict the plot to these output directory names (repeatable)",
    )
    args = parser.parse_args()

    benchmarks = _resolve_benchmarks(args.benchmark, args.benchmarks)
    grouped = {benchmark: defaultdict(list) for benchmark in benchmarks}
    for path, spec, rows in _runs(args.input_root, set(args.run_names or [])):
        arm = str(spec.get("arm", path.name))
        if arm not in ARMS:
            continue
        for row in rows:
            step = int(row.get("step", 0))
            for benchmark in benchmarks:
                result = _benchmark(row, benchmark)
                if result is None or result.get(args.metric) is None:
                    continue
                grouped[benchmark][arm].append((step, float(result[args.metric])))
    if not any(grouped[benchmark] for benchmark in benchmarks):
        raise SystemExit(
            "No ablation evaluation rows found for: " + ", ".join(benchmarks)
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _make_curve_plot(grouped, benchmarks, args.metric, args.output_dir)
    _make_final_plot(grouped, benchmarks, args.metric, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
