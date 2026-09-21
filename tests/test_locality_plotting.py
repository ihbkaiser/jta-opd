import gzip
import json
from pathlib import Path

from b200_experiment.locality_plotting import plot_locality_analysis


def test_locality_plotting_writes_all_png_and_pdf_outputs(tmp_path: Path):
    run = tmp_path / "run" / "analysis" / "locality"
    run.mkdir(parents=True)
    rows = []
    for step in range(1, 4):
        rows.append(
            {
                "step": step,
                "self": {
                    "spearman_g_realized_gain": 0.1 * step,
                    "pearson_g_realized_gain": 0.08 * step,
                    "deciles": [
                        {
                            "decile": decile,
                            "count": 20,
                            "realized_gain_mean": 0.001 * decile * step,
                        }
                        for decile in range(1, 11)
                    ],
                },
                "other_id": {
                    "reverse_kl_after": 0.5 - step * 0.02,
                    "realized_gain": 0.02,
                },
                "future": {
                    "horizon_32_std": 0.1 * step,
                    "horizon_32_iqr": 0.05 * step,
                    "horizon_32_p90_minus_p10": 0.2 * step,
                },
            }
        )
    (run / "metrics.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    sample_dir = run / "token_samples"
    sample_dir.mkdir()
    with gzip.open(sample_dir / "step-000001.jsonl.gz", "wt", encoding="utf-8") as handle:
        for index in range(20):
            handle.write(
                json.dumps(
                    {
                        "g_percentile": 0.4 + index / 100,
                        "future_gain_h32": (index - 10) / 100,
                    }
                )
                + "\n"
            )

    output = tmp_path / "figures"
    paths = plot_locality_analysis([tmp_path / "run"], output)

    expected = {
        "locality_self_deciles.png",
        "locality_self_deciles.pdf",
        "locality_self_correlation_over_steps.png",
        "locality_self_correlation_over_steps.pdf",
        "locality_other_id_over_steps.png",
        "locality_other_id_over_steps.pdf",
        "locality_future_conditional_spread.png",
        "locality_future_conditional_spread.pdf",
        "locality_two_panel_main.png",
        "locality_two_panel_main.pdf",
    }
    assert {path.name for path in paths} == expected
    assert all(path.stat().st_size > 0 for path in paths)
