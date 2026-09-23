from __future__ import annotations

import csv
import json

import pytest

from b200_experiment.probe_logging import OneStepKLProbeLogger


def _record(step: int = 50, *, cmt_kl: float = 0.8) -> dict:
    uniform_kl = 0.95
    return {
        "run_id": "unit-test-run",
        "optimizer_step": step,
        "rollout_id": 3,
        "ppo_group_index": 1,
        "benchmark": "Competition-MATH",
        "subset_size": 1,
        "num_rollouts_per_problem": 2,
        "horizon": 64,
        "sampling_temperature": 1.0,
        "sampling_top_p": 1.0,
        "sampling_seed": 123,
        "heldout_example_hash": "examples-sha256",
        "heldout_root_hash": "roots-sha256",
        "uniform_trajectory_hash": "uniform-sha256",
        "cmt_trajectory_hash": "cmt-sha256",
        "metric": "full_vocab_reverse_kl",
        "uniform_trajectory_kl": uniform_kl,
        "cmt_trajectory_kl": cmt_kl,
        "trajectory_gap": uniform_kl - cmt_kl,
        "uniform_valid_state_count": 100,
        "cmt_valid_state_count": 101,
        "uniform_mean_length": 63.0,
        "cmt_mean_length": 63.5,
        "uniform_early_eos_rate": 0.1,
        "cmt_early_eos_rate": 0.05,
        "uniform_update_loss": 0.4,
        "cmt_update_loss": 0.3,
        "probe_time_sec": 2.5,
        "uniform_branch_time_sec": 0.9,
        "cmt_branch_time_sec": 0.7,
    }


def _problems(step: int = 50, *, cmt_kl: float = 0.8):
    return [
        {
            "run_id": "unit-test-run",
            "optimizer_step": step,
            "problem_id": "p0",
            "uniform_trajectory_kl": 0.95,
            "cmt_trajectory_kl": cmt_kl,
            "trajectory_gap": 0.95 - cmt_kl,
            "uniform_mean_length": 63.0,
            "cmt_mean_length": 64.0,
            "uniform_early_eos_rate": 0.5,
            "cmt_early_eos_rate": 0.0,
        }
    ]


def test_upsert_replaces_step_and_problem_rows(tmp_path):
    logger = OneStepKLProbeLogger(tmp_path)
    logger.upsert(_record(), _problems())
    logger.upsert(_record(cmt_kl=0.7), _problems(cmt_kl=0.7))

    root = tmp_path / "one_step_trajectory_kl_probe"
    main = [
        json.loads(line)
        for line in (root / "trajectory_kl_probe.jsonl").read_text().splitlines()
    ]
    details = [
        json.loads(line)
        for line in (root / "trajectory_kl_per_problem.jsonl").read_text().splitlines()
    ]
    with (root / "trajectory_kl_probe.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        csv_rows = list(csv.DictReader(handle))

    assert len(main) == len(details) == len(csv_rows) == 1
    assert main[0]["cmt_trajectory_kl"] == pytest.approx(0.7)
    assert details[0]["trajectory_gap"] == pytest.approx(0.25)


def test_rejects_inconsistent_trajectory_gap(tmp_path):
    row = _record()
    row["trajectory_gap"] += 0.1
    with pytest.raises(ValueError, match="trajectory_gap"):
        OneStepKLProbeLogger(tmp_path).upsert(row, _problems())


def test_rejects_nonfinite_metric(tmp_path):
    row = _record()
    row["uniform_trajectory_kl"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        OneStepKLProbeLogger(tmp_path).upsert(row, _problems())


def test_disabled_or_non_main_logger_is_noop(tmp_path):
    OneStepKLProbeLogger(tmp_path, enabled=False).upsert(_record(), _problems())
    OneStepKLProbeLogger(tmp_path, is_main=False).upsert(_record(), _problems())
    assert not (tmp_path / "one_step_trajectory_kl_probe").exists()
