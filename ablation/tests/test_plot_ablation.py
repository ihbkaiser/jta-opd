from __future__ import annotations

import json
import sys
from pathlib import Path

from ablation.plots.plot_ablation import BENCHMARK_ORDER, _resolve_benchmarks, main


def _write_run(root: Path, arm: str) -> None:
    output = root / f"run_{arm}"
    output.mkdir(parents=True)
    (output / "ablation_spec.json").write_text(
        json.dumps({"arm": arm}) + "\n", encoding="utf-8"
    )
    rows = []
    for step in (0, 10):
        rows.append(
            {
                "step": step,
                "benchmarks": {
                    benchmark: {"accuracy": 0.50 + 0.01 * step / 10}
                    for benchmark in BENCHMARK_ORDER
                },
            }
        )
    (output / "eval_history.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def test_default_arm_plot_uses_all_six_benchmarks(tmp_path, monkeypatch):
    for arm in ("g", "g_x", "g_d"):
        _write_run(tmp_path, arm)
    output = tmp_path / "figures"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "plot_ablation.py",
            "--input-root",
            str(tmp_path),
            "--output-dir",
            str(output),
        ],
    )
    assert main() == 0
    assert (output / "ablation_curves.png").is_file()
    assert (output / "ablation_final.png").is_file()
    assert _resolve_benchmarks(None, None) == BENCHMARK_ORDER


def test_arm_plot_accepts_benchmark_subset(tmp_path, monkeypatch):
    _write_run(tmp_path, "g")
    output = tmp_path / "figures"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "plot_ablation.py",
            "--input-root",
            str(tmp_path),
            "--output-dir",
            str(output),
            "--benchmarks",
            "MATH-500,GPQA-Diamond",
        ],
    )
    assert main() == 0
    assert (output / "ablation_curves.png").is_file()
