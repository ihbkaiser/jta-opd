from __future__ import annotations

import csv
import json
import math
import os
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence


PROBE_FIELDS = (
    "run_id", "optimizer_step", "rollout_id", "ppo_group_index",
    "benchmark", "subset_size", "num_rollouts_per_problem", "horizon",
    "sampling_temperature", "sampling_top_p", "sampling_seed",
    "heldout_example_hash", "heldout_root_hash", "uniform_trajectory_hash",
    "cmt_trajectory_hash", "metric", "uniform_trajectory_kl",
    "cmt_trajectory_kl", "trajectory_gap", "uniform_valid_state_count",
    "cmt_valid_state_count", "uniform_mean_length", "cmt_mean_length",
    "uniform_early_eos_rate", "cmt_early_eos_rate", "uniform_update_loss",
    "cmt_update_loss", "probe_time_sec", "uniform_branch_time_sec",
    "cmt_branch_time_sec",
)

PROBLEM_FIELDS = (
    "run_id", "optimizer_step", "problem_id", "uniform_trajectory_kl",
    "cmt_trajectory_kl", "trajectory_gap", "uniform_mean_length",
    "cmt_mean_length", "uniform_early_eos_rate", "cmt_early_eos_rate",
)

_PROBE_INTS = {
    "optimizer_step", "rollout_id", "ppo_group_index", "subset_size",
    "num_rollouts_per_problem", "horizon", "sampling_seed",
    "uniform_valid_state_count", "cmt_valid_state_count",
}
_PROBE_FLOATS = set(PROBE_FIELDS) - _PROBE_INTS - {
    "run_id", "benchmark", "heldout_example_hash", "heldout_root_hash",
    "uniform_trajectory_hash", "cmt_trajectory_hash", "metric",
}
_PROBLEM_FLOATS = set(PROBLEM_FIELDS) - {
    "run_id", "optimizer_step", "problem_id"
}


def _validate_gap(row: Mapping[str, Any]) -> None:
    expected = float(row["uniform_trajectory_kl"]) - float(row["cmt_trajectory_kl"])
    if not math.isclose(float(row["trajectory_gap"]), expected, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(
            f"trajectory_gap={row['trajectory_gap']} is inconsistent with {expected}"
        )


class OneStepKLProbeLogger:
    """Atomically log step-level and per-problem trajectory KL outcomes."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        enabled: bool = True,
        is_main: bool = True,
        artifact_subdir: str = "one_step_trajectory_kl_probe",
    ) -> None:
        self.enabled = bool(enabled)
        self.is_main = bool(is_main)
        self.root = Path(output_dir) / artifact_subdir
        self.jsonl_path = self.root / "trajectory_kl_probe.jsonl"
        self.csv_path = self.root / "trajectory_kl_probe.csv"
        self.problem_jsonl_path = self.root / "trajectory_kl_per_problem.jsonl"
        self.problem_csv_path = self.root / "trajectory_kl_per_problem.csv"

    @staticmethod
    def _validate_probe(record: Mapping[str, Any]) -> dict[str, Any]:
        missing = [field for field in PROBE_FIELDS if field not in record]
        if missing:
            raise ValueError(f"Probe record is missing required fields: {missing}")
        row = {field: record[field] for field in PROBE_FIELDS}
        for field in _PROBE_INTS:
            if isinstance(row[field], bool) or not isinstance(row[field], int):
                raise ValueError(f"Probe field {field} must be an integer")
        if row["optimizer_step"] < 1 or row["subset_size"] < 1:
            raise ValueError("Probe step and subset size must be positive")
        for field in _PROBE_FLOATS:
            value = float(row[field])
            if not math.isfinite(value):
                raise ValueError(f"Probe field {field} must be finite")
            row[field] = value
        _validate_gap(row)
        return row

    @staticmethod
    def _validate_problem(record: Mapping[str, Any]) -> dict[str, Any]:
        missing = [field for field in PROBLEM_FIELDS if field not in record]
        if missing:
            raise ValueError(f"Problem record is missing required fields: {missing}")
        row = {field: record[field] for field in PROBLEM_FIELDS}
        if isinstance(row["optimizer_step"], bool) or not isinstance(
            row["optimizer_step"], int
        ):
            raise ValueError("Problem optimizer_step must be an integer")
        for field in _PROBLEM_FLOATS:
            value = float(row[field])
            if not math.isfinite(value):
                raise ValueError(f"Problem field {field} must be finite")
            row[field] = value
        _validate_gap(row)
        return row

    @staticmethod
    def _read(path: Path, validator) -> list[dict[str, Any]]:
        if not path.is_file():
            return []
        rows = []
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    rows.append(validator(json.loads(line)))
                except (json.JSONDecodeError, ValueError) as error:
                    raise ValueError(f"Malformed probe log {path} line {line_number}") from error
        return rows

    @staticmethod
    def _temporary(path: Path) -> Path:
        return path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")

    @staticmethod
    def _write_pair(
        jsonl_path: Path,
        csv_path: Path,
        fields: tuple[str, ...],
        rows: Sequence[Mapping[str, Any]],
    ) -> None:
        json_tmp = OneStepKLProbeLogger._temporary(jsonl_path)
        csv_tmp = OneStepKLProbeLogger._temporary(csv_path)
        try:
            with json_tmp.open("x", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            with csv_tmp.open("x", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
                handle.flush()
                os.fsync(handle.fileno())
            json_tmp.replace(jsonl_path)
            csv_tmp.replace(csv_path)
        except BaseException:
            json_tmp.unlink(missing_ok=True)
            csv_tmp.unlink(missing_ok=True)
            raise

    def upsert(
        self,
        record: Mapping[str, Any],
        problem_records: Sequence[Mapping[str, Any]],
    ) -> tuple[Path, Path] | None:
        if not self.enabled or not self.is_main:
            return None
        row = self._validate_probe(record)
        details = [self._validate_problem(item) for item in problem_records]
        if len(details) != row["subset_size"]:
            raise ValueError("Per-problem rows do not match probe subset_size")
        if any(item["optimizer_step"] != row["optimizer_step"] for item in details):
            raise ValueError("Per-problem rows do not match probe optimizer_step")
        self.root.mkdir(parents=True, exist_ok=True)
        main = {
            item["optimizer_step"]: item
            for item in self._read(self.jsonl_path, self._validate_probe)
        }
        main[row["optimizer_step"]] = row
        detail_rows = [
            item
            for item in self._read(self.problem_jsonl_path, self._validate_problem)
            if item["optimizer_step"] != row["optimizer_step"]
        ] + details
        ordered_main = [main[step] for step in sorted(main)]
        detail_rows.sort(key=lambda item: (item["optimizer_step"], item["problem_id"]))
        self._write_pair(self.jsonl_path, self.csv_path, PROBE_FIELDS, ordered_main)
        self._write_pair(
            self.problem_jsonl_path,
            self.problem_csv_path,
            PROBLEM_FIELDS,
            detail_rows,
        )
        return self.jsonl_path, self.csv_path
