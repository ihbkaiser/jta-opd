from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from b200_experiment.probe_logging import OneStepKLProbeLogger


def _record(step: int = 7, *, kl_after_cmt: float = 0.8) -> dict:
    kl_before = 1.0
    kl_after_uniform = 0.95
    delta_uniform = kl_before - kl_after_uniform
    delta_cmt = kl_before - kl_after_cmt
    return {
        "run_id": "unit-test-run",
        "optimizer_step": step,
        "rollout_id": 3,
        "ppo_group_index": 1,
        "benchmark": "hendrycks/competition_math:test",
        "subset_size": 64,
        "heldout_example_hash": "examples-sha256",
        "heldout_prefix_hash": "prefixes-sha256",
        "valid_prefix_token_count": 123,
        "top_k": 16,
        "metric": "conditional_reverse_kl",
        "kl_before": kl_before,
        "kl_after_uniform": kl_after_uniform,
        "kl_after_cmt": kl_after_cmt,
        "delta_uniform": delta_uniform,
        "delta_cmt": delta_cmt,
        "paired_gap": delta_cmt - delta_uniform,
        "uniform_update_loss": 0.4,
        "cmt_update_loss": 0.3,
        "probe_time_sec": 2.5,
        "uniform_branch_time_sec": 0.9,
        "cmt_eval_time_sec": 0.7,
    }


class OneStepKLProbeLoggerTests(unittest.TestCase):
    def test_upsert_replaces_same_step_in_jsonl_and_csv(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            logger = OneStepKLProbeLogger(root)
            logger.upsert(_record())
            replacement = _record(kl_after_cmt=0.7)
            logger.upsert(replacement)

            artifact = root / "one_step_kl_probe"
            with (artifact / "one_step_kl_probe.jsonl").open(
                encoding="utf-8"
            ) as handle:
                json_rows = [json.loads(line) for line in handle if line.strip()]
            with (artifact / "one_step_kl_probe.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                csv_rows = list(csv.DictReader(handle))

            self.assertEqual(len(json_rows), 1)
            self.assertEqual(len(csv_rows), 1)
            self.assertEqual(json_rows[0], replacement)
            self.assertEqual(int(csv_rows[0]["optimizer_step"]), 7)
            self.assertAlmostEqual(float(csv_rows[0]["kl_after_cmt"]), 0.7)

    def test_rejects_inconsistent_paired_metrics(self):
        with tempfile.TemporaryDirectory() as temporary:
            row = _record()
            row["paired_gap"] += 0.1
            with self.assertRaisesRegex(ValueError, "paired_gap"):
                OneStepKLProbeLogger(temporary).upsert(row)

    def test_rejects_nonfinite_metric(self):
        with tempfile.TemporaryDirectory() as temporary:
            row = _record()
            row["kl_before"] = float("nan")
            with self.assertRaisesRegex(ValueError, "finite"):
                OneStepKLProbeLogger(temporary).upsert(row)

    def test_disabled_or_non_main_logger_is_a_noop(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            OneStepKLProbeLogger(root, enabled=False).upsert(_record())
            OneStepKLProbeLogger(root, is_main=False).upsert(_record())
            self.assertFalse((root / "one_step_kl_probe").exists())


if __name__ == "__main__":
    unittest.main()
