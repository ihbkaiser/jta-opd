from __future__ import annotations

import argparse
import gzip
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

plt.switch_backend("Agg")


BLUE = "#0072B2"
SKY = "#56B4E9"
GREEN = "#009E73"
ORANGE = "#E69F00"
VERMILLION = "#D55E00"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.15,
        "grid.linestyle": "-",
        "lines.linewidth": 1.8,
        "lines.markersize": 4,
    }
)


def _locality_root(path: Path) -> Path:
    direct = path / "metrics.jsonl"
    return path if direct.is_file() else path / "analysis" / "locality"


def _load_runs(inputs: Sequence[str | Path]) -> tuple[list[dict[str, Any]], list[float]]:
    rows: list[dict[str, Any]] = []
    future_samples: list[float] = []
    for value in inputs:
        root = _locality_root(Path(value))
        metrics = root / "metrics.jsonl"
        if not metrics.is_file():
            raise FileNotFoundError(f"Missing locality metrics: {metrics}")
        rows.extend(
            json.loads(line)
            for line in metrics.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        for sample_path in sorted((root / "token_samples").glob("*.jsonl.gz")):
            with gzip.open(sample_path, "rt", encoding="utf-8") as handle:
                for line in handle:
                    item = json.loads(line)
                    percentile = item.get("g_percentile")
                    future = item.get("future_gain_h32")
                    if (
                        percentile is not None
                        and future is not None
                        and 0.4 <= float(percentile) <= 0.6
                    ):
                        future_samples.append(float(future))
    if not rows:
        raise ValueError("No locality metric rows were found")
    rows.sort(key=lambda row: int(row["step"]))
    return rows, future_samples


def _decile_matrix(rows: list[dict[str, Any]]) -> np.ndarray:
    matrix = []
    for row in rows:
        by_decile = {
            int(item["decile"]): item.get("realized_gain_mean")
            for item in row["self"].get("deciles", [])
            if item.get("count", 0)
        }
        matrix.append([by_decile.get(index, np.nan) for index in range(1, 11)])
    return np.asarray(matrix, dtype=float)


def _bootstrap_deciles(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.nanmean(matrix, axis=0)
    if matrix.shape[0] < 2:
        return mean, mean, mean
    generator = np.random.default_rng(20260921)
    samples = np.empty((2000, matrix.shape[1]), dtype=float)
    for index in range(samples.shape[0]):
        selection = generator.integers(0, matrix.shape[0], size=matrix.shape[0])
        samples[index] = np.nanmean(matrix[selection], axis=0)
    return mean, np.nanpercentile(samples, 2.5, axis=0), np.nanpercentile(samples, 97.5, axis=0)


def _draw_deciles(ax, rows: list[dict[str, Any]]) -> None:
    mean, low, high = _bootstrap_deciles(_decile_matrix(rows))
    deciles = np.arange(1, 11)
    ax.plot(deciles, mean, color=VERMILLION, marker="o", label="Realized SELF gain")
    ax.fill_between(deciles, low, high, color=VERMILLION, alpha=0.16, linewidth=0)
    ax.axhline(0.0, color="#777777", linewidth=0.8, linestyle="--")
    ax.set_xticks(deciles)
    ax.set_xlabel("Pre-update local gain decile")
    ax.set_ylabel(r"Realized SELF gain $I_t$")


def _save(fig, output: Path, stem: str) -> list[Path]:
    paths = [output / f"{stem}.png", output / f"{stem}.pdf"]
    for path in paths:
        fig.savefig(path)
    plt.close(fig)
    return paths


def plot_locality_analysis(
    inputs: Sequence[str | Path], output_dir: str | Path
) -> list[Path]:
    rows, future_samples = _load_runs(inputs)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    fig, ax = plt.subplots(figsize=(3.25, 2.5))
    _draw_deciles(ax, rows)
    ax.set_title("Local validity")
    paths.extend(_save(fig, output, "locality_self_deciles"))

    steps = np.asarray([int(row["step"]) for row in rows])
    fig, ax = plt.subplots(figsize=(3.25, 2.5))
    ax.plot(
        steps,
        [row["self"].get("spearman_g_realized_gain", np.nan) for row in rows],
        color=BLUE,
        marker="o",
        label="Spearman",
    )
    ax.plot(
        steps,
        [row["self"].get("pearson_g_realized_gain", np.nan) for row in rows],
        color=ORANGE,
        marker="s",
        label="Pearson",
    )
    ax.axhline(0.0, color="#777777", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Correlation with realized SELF gain")
    ax.set_title("Local correlation over training")
    ax.legend()
    paths.extend(_save(fig, output, "locality_self_correlation_over_steps"))

    fig, ax = plt.subplots(figsize=(3.25, 2.5))
    reverse_kl = [
        (row.get("other_id") or {}).get("reverse_kl_after", np.nan) for row in rows
    ]
    transfer = [
        (row.get("other_id") or {}).get("realized_gain", np.nan) for row in rows
    ]
    ax.plot(steps, reverse_kl, color=BLUE, marker="o", label="Reverse-KL")
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Fixed-test reverse-KL", color=BLUE)
    secondary = ax.twinx()
    secondary.bar(steps, transfer, color=ORANGE, alpha=0.28, label="Step gain")
    secondary.set_ylabel("Step-wise OTHER-ID gain", color=ORANGE)
    ax.set_title("Transfer to fixed Competition-MATH test states")
    paths.extend(_save(fig, output, "locality_other_id_over_steps"))

    fig, ax = plt.subplots(figsize=(3.25, 2.5))
    for key, label, color, marker in (
        ("horizon_32_std", "Std.", BLUE, "o"),
        ("horizon_32_iqr", "IQR", GREEN, "s"),
        ("horizon_32_p90_minus_p10", "P90−P10", VERMILLION, "^"),
    ):
        ax.plot(
            steps,
            [row["future"].get(key, np.nan) for row in rows],
            color=color,
            marker=marker,
            label=label,
        )
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel(r"Spread of $F_t^{(32)}$ in Q40–Q60")
    ax.set_title("Conditional future utility spread")
    ax.legend()
    paths.extend(_save(fig, output, "locality_future_conditional_spread"))

    fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.8))
    _draw_deciles(axes[0], rows)
    axes[0].set_title("(a) Local validity")
    if future_samples:
        parts = axes[1].violinplot(
            np.asarray(future_samples),
            positions=[1],
            widths=0.65,
            showmeans=True,
            showmedians=True,
            showextrema=False,
        )
        for body in parts["bodies"]:
            body.set_facecolor(SKY)
            body.set_edgecolor(BLUE)
            body.set_alpha(0.65)
        axes[1].boxplot(
            future_samples,
            positions=[1],
            widths=0.16,
            showfliers=False,
            patch_artist=True,
            boxprops={"facecolor": "white", "edgecolor": BLUE},
            medianprops={"color": VERMILLION},
        )
        axes[1].set_xticks([1], ["Q40–Q60 local gain"])
    else:
        axes[1].text(0.5, 0.5, "No sampled H=32 states", ha="center", va="center")
        axes[1].set_xticks([])
    axes[1].axhline(0.0, color="#777777", linewidth=0.8, linestyle="--")
    axes[1].set_ylabel(r"Realized successor gain $F_t^{(32)}$")
    axes[1].set_title("(b) Local insufficiency")
    fig.tight_layout(w_pad=2.0)
    paths.extend(_save(fig, output, "locality_two_panel_main"))
    return paths


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plot per-step CMT locality analysis")
    parser.add_argument("inputs", nargs="+", help="Run directories or analysis/locality directories")
    parser.add_argument("--output-dir", required=True)
    arguments = parser.parse_args(argv)
    for path in plot_locality_analysis(arguments.inputs, arguments.output_dir):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
