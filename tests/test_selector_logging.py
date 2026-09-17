from __future__ import annotations

import gzip
import json
import tempfile
import unittest
from pathlib import Path

import torch

from b200_experiment.selector_logging import (
    CMTDiagnosticsLogger,
    SelectedTokenLogger,
    TokenScoreStatsLogger,
)


class FakeTokenizer:
    @staticmethod
    def convert_ids_to_tokens(ids):
        return [f"tok-{item}" for item in ids]


class SelectorLoggingTests(unittest.TestCase):
    def test_cmt_diagnostics_disabled_creates_no_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            logger = CMTDiagnosticsLogger(temporary, FakeTokenizer())
            count = logger.write(
                step=1,
                response_ids=torch.tensor([[5]]),
                valid_mask=torch.tensor([[True]]),
                dataset_indices=[0],
                sample_ids=["s"],
                response_indices=[0],
                diagnostics={"gain": torch.ones(1, 1)},
            )
            self.assertEqual(count, 0)
            self.assertFalse((Path(temporary) / "cmt_diagnostics").exists())

    def test_ta_selected_tokens_are_incrementally_gzipped(self):
        with tempfile.TemporaryDirectory() as temporary:
            mask = torch.tensor([[True, False], [False, True]])
            diagnostics = {
                key: torch.arange(4, dtype=torch.float32).reshape(2, 2)
                for key in ("D", "C", "D_norm", "C_norm", "s_TA")
            }
            logger = SelectedTokenLogger(
                temporary, FakeTokenizer(), "ta", chunk_steps=2
            )
            count = logger.write(
                step=1,
                dataset_indices=[10, 11],
                sample_ids=["a", "b"],
                response_ids=torch.tensor([[5, 6], [7, 8]]),
                selected_mask=mask,
                diagnostics=diagnostics,
            )
            self.assertEqual(count, 2)
            path = next((Path(temporary) / "selector_scores").glob("*.jsonl.gz"))
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                rows = [json.loads(line) for line in handle]
            self.assertEqual(rows[0]["token_text"], "tok-5")
            self.assertEqual(rows[1]["sample_id"], "b")
            self.assertIn("s_TA", rows[0])

    def test_distributed_ranks_use_independent_gzip_members(self):
        with tempfile.TemporaryDirectory() as temporary:
            mask = torch.tensor([[True]])
            diagnostics = {
                key: torch.ones(1, 1) for key in ("D", "C", "D_norm", "C_norm", "s_TA")
            }
            for rank in (0, 1):
                logger = SelectedTokenLogger(
                    temporary,
                    FakeTokenizer(),
                    "ta",
                    rank=rank,
                    world_size=2,
                )
                logger.write(
                    step=1,
                    dataset_indices=[10 + rank],
                    sample_ids=[str(rank)],
                    response_ids=torch.tensor([[5 + rank]]),
                    selected_mask=mask,
                    diagnostics=diagnostics,
                    batch_index_offset=rank,
                )
            paths = sorted((Path(temporary) / "selector_scores").glob("*.jsonl.gz"))
            self.assertEqual(len(paths), 2)
            self.assertIn("rank-00000", paths[0].name)
            self.assertIn("rank-00001", paths[1].name)

    def test_cmt_selected_tokens_include_bounded_kernel_diagnostics(self):
        with tempfile.TemporaryDirectory() as temporary:
            mask = torch.tensor([[True, False]])
            keys = (
                "gain",
                "support_common_mass",
                "conditional_support_common_mass",
                "alignment",
                "transition_weight",
                "support_coverage",
                "teacher_deficit",
                "marginal_flux",
                "successor_excess",
                "sequential_gain",
                "learning_value",
                "s_CMT",
            )
            diagnostics = {key: torch.ones(1, 2) for key in keys}
            logger = SelectedTokenLogger(
                temporary, FakeTokenizer(), "cmt", chunk_steps=2
            )
            count = logger.write(
                step=1,
                dataset_indices=[10],
                sample_ids=["cmt"],
                response_ids=torch.tensor([[5, 6]]),
                selected_mask=mask,
                diagnostics=diagnostics,
            )
            self.assertEqual(count, 1)
            path = next((Path(temporary) / "selector_scores").glob("*.jsonl.gz"))
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                row = json.loads(next(handle))
            self.assertEqual(row["token_text"], "tok-5")
            self.assertEqual(row["transition_weight"], 1.0)
            self.assertEqual(row["conditional_support_common_mass"], 1.0)

    def test_compact_rac_stats_cover_all_values_and_bound_raw_sample(self):
        with tempfile.TemporaryDirectory() as temporary:
            diagnostics = {
                key: torch.linspace(0, 1, 100)
                for key in ("g", "alignment", "V", "z", "w")
            }
            logger = TokenScoreStatsLogger(
                temporary, "rac", interval=50, bins=10, raw_sample_size=7
            )
            path = logger.write(1, 100, diagnostics)
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["scores"]["V"]["count"], 100)
            self.assertEqual(sum(payload["scores"]["V"]["histogram"]["counts"]), 100)
            self.assertEqual(len(payload["scores"]["V"]["sample"]), 7)
            self.assertIsNone(logger.write(2, 100, diagnostics))

    def test_compact_opd_stats_record_uniform_all_token_weights(self):
        with tempfile.TemporaryDirectory() as temporary:
            logger = TokenScoreStatsLogger(temporary, "opd", interval=50, bins=10)
            path = logger.write(1, 100, {"w": torch.ones(17)})
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(tuple(payload["scores"]), ("w",))
            self.assertEqual(payload["scores"]["w"]["count"], 17)
            self.assertEqual(payload["scores"]["w"]["mean"], 1.0)

    def test_compact_cmt_stats_accept_all_selector_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            keys = (
                "s_CMT", "gain", "support_reverse_kl", "support_common_mass",
                "alignment", "transition_weight", "support_coverage",
                "coverage_correction", "teacher_deficit", "marginal_flux",
                "common_mass_derivative", "R", "M", "V", "H",
                "sequential_gain", "learning_value", "w",
            )
            diagnostics = {key: torch.linspace(0, 1, 9) for key in keys}
            logger = TokenScoreStatsLogger(temporary, "cmt", interval=10, bins=5)
            path = logger.write(1, 10, diagnostics)
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(set(payload["scores"]), set(keys))
            self.assertEqual(payload["scores"]["learning_value"]["count"], 9)

    def test_cmt_detailed_logger_writes_required_detached_fields(self):
        with tempfile.TemporaryDirectory() as temporary:
            valid = torch.tensor([[True, True, False]])
            diagnostics = {
                "gain": torch.tensor([[1.0, 2.0, 0.0]]),
                "marginal_flux": torch.tensor([[0.1, -0.2, 0.0]]),
                "sampled_log_ratio": torch.tensor([[0.2, -0.3, 0.0]]),
                "sampled_conditional_log_ratio": torch.tensor([[0.2, -0.3, 0.0]]),
                "conditional_log_ratio_mean": torch.tensor([[0.0, 0.0, 0.0]]),
                "student_union_mass": torch.ones(1, 3),
                "teacher_union_mass": torch.ones(1, 3),
                "support_common_mass": torch.ones(1, 3),
                "conditional_support_common_mass": torch.ones(1, 3),
                "support_coverage": torch.ones(1, 3),
                "support_width": torch.full((1, 3), 4.0),
                "in_support": torch.ones(1, 3),
                "teacher_deficit": torch.tensor([[1.0, 0.0, 0.0]]),
                "transition_weight": torch.ones(1, 3),
                "successor_return": torch.tensor([[2.0, 1.0, 0.0]]),
                "successor_mass": torch.tensor([[1.0, 1.0, 0.0]]),
                "successor_value": torch.tensor([[2.0, 1.0, 0.0]]),
                "successor_contrast": torch.tensor([[1.0, -1.0, 0.0]]),
                "baseline_mass_term": torch.tensor([[1.0, 2.0, 0.0]]),
                "successor_excess": torch.tensor([[1.0, -1.0, 0.0]]),
                "x_difference_form": torch.tensor([[1.0, -1.0, 0.0]]),
                "x_product_form": torch.tensor([[1.0, -1.0, 0.0]]),
                "x_difference_identity_error": torch.zeros(1, 3),
                "x_product_identity_error": torch.zeros(1, 3),
                "x_cancellation_ratio": torch.ones(1, 3),
                "sequential_gain": torch.tensor([[0.1, -0.2, 0.0]]),
                "d_product_form": torch.tensor([[0.1, -0.2, 0.0]]),
                "d_identity_error": torch.zeros(1, 3),
                "learning_value": torch.tensor([[1.1, 1.8, 0.0]]),
                "w": torch.tensor([[2.0, 0.0, 0.0]]),
                "selected_mask": torch.tensor([[1.0, 0.0, 0.0]]),
                "gamma": 1.0,
                "token_loss": torch.tensor([[0.5, -0.25, 0.0]]),
                "token_advantage": torch.tensor([[0.5, 0.25, 0.0]]),
                "ppo_ratio": torch.ones(1, 3),
                "token_clipped": torch.zeros(1, 3),
                "weighted_loss_contribution": torch.tensor([[1.0, 0.0, 0.0]]),
                "abs_weighted_loss_contribution": torch.tensor([[1.0, 0.0, 0.0]]),
                "valid_mask": valid,
            }
            logger = CMTDiagnosticsLogger(
                temporary,
                FakeTokenizer(),
                detailed_enabled=True,
                fp64_check=True,
                partial_horizons=(1, 2),
            )
            count = logger.write(
                step=1,
                response_ids=torch.tensor([[5, 6, 0]]),
                valid_mask=valid,
                dataset_indices=[7],
                sample_ids=["sample"],
                response_indices=[0],
                diagnostics=diagnostics,
                max_new_tokens=3,
                global_diagnostics={"w": torch.tensor([2.0, 0.0])},
                allocation_metadata={
                    "allocation_mode": "top_fraction",
                    "allocation_top_fraction": 0.5,
                },
            )
            self.assertEqual(count, 2)
            path = next((Path(temporary) / "cmt_diagnostics").glob("detailed_*.gz"))
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                row = json.loads(next(handle))
            for field in (
                "training_step", "dataset_index", "sample_id", "response_index",
                "response_length", "remaining_valid_tokens", "token_text",
                "marginal_flux", "successor_excess", "x_cancellation_ratio",
                "successor_excess_fp64", "token_loss", "ppo_ratio",
                "weighted_loss_contribution", "gradient_influence_proxy",
                "allocation_mode", "allocation_top_fraction", "rho",
            ):
                self.assertIn(field, row)
            self.assertEqual(row["response_length"], 2)
            self.assertEqual(row["remaining_valid_tokens"], 1)
            self.assertEqual(row["rho"], 1.0)

    def test_cmt_tail_includes_full_and_partial_successor_horizons(self):
        with tempfile.TemporaryDirectory() as temporary:
            valid = torch.tensor([[True, True, True]])
            diagnostics = {
                "gain": torch.tensor([[1.0, 2.0, 4.0]]),
                "transition_weight": torch.ones(1, 3),
                "sequential_gain": torch.tensor([[3.0, 2.0, 1.0]]),
                "successor_excess": torch.tensor([[3.0, 2.0, 0.0]]),
                "successor_mass": torch.tensor([[2.0, 1.0, 0.0]]),
                "marginal_flux": torch.ones(1, 3),
                "learning_value": torch.ones(1, 3),
                "w": torch.ones(1, 3),
                "gamma": 1.0,
            }
            logger = CMTDiagnosticsLogger(
                temporary, FakeTokenizer(), tail_enabled=True, tail_top_k=1,
                partial_horizons=(1,), fp64_check=False,
            )
            logger.write(
                step=2,
                response_ids=torch.tensor([[5, 6, 7]]),
                valid_mask=valid,
                dataset_indices=[0], sample_ids=["s"], response_indices=[0],
                diagnostics=diagnostics, max_new_tokens=8,
            )
            path = next((Path(temporary) / "cmt_diagnostics").glob("tail_*.gz"))
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                row = json.loads(next(handle))
            self.assertIn("successor_excess_h1", row)
            self.assertIn("successor_excess_hfull", row)

    def test_cmt_tail_logger_deduplicates_union_and_writes_global_ranks(self):
        with tempfile.TemporaryDirectory() as temporary:
            valid = torch.tensor([[True, True]])
            diagnostics = {
                "gain": torch.ones(1, 2),
                "sequential_gain": torch.tensor([[2.0, 1.0]]),
                "successor_excess": torch.tensor([[1.0, 3.0]]),
                "successor_mass": torch.tensor([[1.0, 2.0]]),
                "marginal_flux": torch.tensor([[1.0, 2.0]]),
                "learning_value": torch.tensor([[2.0, 3.0]]),
                "w": torch.tensor([[1.0, 4.0]]),
                "transition_weight": torch.ones(1, 2),
                "valid_mask": valid,
                "gamma": 1.0,
            }
            logger = CMTDiagnosticsLogger(
                temporary, FakeTokenizer(), tail_enabled=True, tail_top_k=1,
                partial_horizons=(1,), fp64_check=False,
            )
            logger.write(
                step=5,
                response_ids=torch.tensor([[5, 6]]), valid_mask=valid,
                dataset_indices=[1], sample_ids=["s"], response_indices=[0],
                diagnostics=diagnostics, max_new_tokens=8,
                global_diagnostics={
                    key: value.reshape(-1)
                    for key, value in diagnostics.items()
                    if torch.is_tensor(value) and key != "valid_mask"
                },
            )
            path = next((Path(temporary) / "cmt_diagnostics").glob("tail_*.gz"))
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                rows = [json.loads(line) for line in handle]
            self.assertEqual(len(rows), 2)
            self.assertTrue(all("global_rank_w" in row for row in rows))
            self.assertTrue(any("top_w" in row["tail_reasons"] for row in rows))


if __name__ == "__main__":
    unittest.main()
