#!/usr/bin/env python3
"""Plot CMT ablation learning curves and final benchmark bars."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def _runs(root: Path, selected: set[str] | None = None):
    for spec_path in sorted(root.rglob("ablation_spec.json")):
        if selected and spec_path.parent.name not in selected:
            continue
        try:
            spec = json.loads(spec_path.read_text(encoding="utf-8"))
            history_path = spec_path.parent / "eval_history.jsonl"
            if history_path.is_file():
                rows = [json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines() if line.strip()]
                yield spec_path.parent, spec, rows
        except (OSError, json.JSONDecodeError):
            continue


def _benchmark(row, requested: str):
    values = row.get("benchmarks", {})
    if requested in values:
        return values[requested]
    wanted = requested.lower().replace("-", "").replace("_", "")
    for name, result in values.items():
        if name.lower().replace("-", "").replace("_", "") == wanted:
            return result
    return next(iter(values.values()), None)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--benchmark", default="MATH-500")
    parser.add_argument("--metric", choices=("accuracy", "avg_at_n", "pass_at_8"), default="accuracy")
    parser.add_argument("--run-name", action="append", dest="run_names", help="Restrict the plot to these output directory names (repeatable)")
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    grouped = defaultdict(list)
    for path, spec, rows in _runs(args.input_root, set(args.run_names or [])):
        arm = spec.get("arm", path.name)
        for row in rows:
            result = _benchmark(row, args.benchmark)
            if result and result.get(args.metric) is not None:
                grouped[arm].append((int(row.get("step", 0)), float(result[args.metric])))
    if not grouped:
        raise SystemExit("No ablation eval_history.jsonl rows found")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    colors = {"g": "tab:blue", "g_x": "tab:orange", "g_d": "tab:purple"}
    fig, ax = plt.subplots(figsize=(8, 5))
    for arm in ("g", "g_x", "g_d"):
        values = grouped.get(arm)
        if not values:
            continue
        by_step = defaultdict(list)
        for step, value in values:
            by_step[step].append(value)
        steps = sorted(by_step)
        mean = np.asarray([np.mean(by_step[s]) for s in steps])
        std = np.asarray([np.std(by_step[s]) for s in steps])
        ax.plot(steps, mean, label=arm, color=colors.get(arm), linewidth=2)
        if np.any(std > 0):
            ax.fill_between(steps, mean - std, mean + std, color=colors.get(arm), alpha=.18)
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel(f"{args.metric} ({args.benchmark})")
    ax.set_title("CMT ablation learning curves")
    ax.grid(alpha=.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "ablation_curves.png", dpi=180)
    fig.savefig(args.output_dir / "ablation_curves.pdf")
    plt.close(fig)

    finals = {}
    for arm, values in grouped.items():
        by_step = defaultdict(list)
        for step, value in values:
            by_step[step].append(value)
        last = by_step[max(by_step)]
        finals[arm] = (float(np.mean(last)), float(np.std(last)))
    fig, ax = plt.subplots(figsize=(6, 4))
    arms = [arm for arm in ("g", "g_x", "g_d") if arm in finals]
    x = np.arange(len(arms))
    ax.bar(x, [finals[a][0] for a in arms], yerr=[finals[a][1] for a in arms], color=[colors.get(a) for a in arms], capsize=4)
    ax.set_xticks(x, arms)
    ax.set_ylabel(f"Final {args.metric} ({args.benchmark})")
    ax.set_title("Final CMT ablation comparison")
    ax.grid(axis="y", alpha=.25)
    fig.tight_layout()
    fig.savefig(args.output_dir / "ablation_final.png", dpi=180)
    fig.savefig(args.output_dir / "ablation_final.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
