from __future__ import annotations

import csv
import json
import math
import os
import uuid
from pathlib import Path
from typing import Any, Mapping

from b200_experiment.one_step_kl_probe import paired_improvement


PROBE_FIELDS = (
    "run_id",
    "optimizer_step",
    "rollout_id",
    "ppo_group_index",
    "benchmark",
    "subset_size",
    "heldout_example_hash",
    "heldout_prefix_hash",
    "valid_prefix_token_count",
    "top_k",
    "metric",
    "kl_before",
    "kl_after_uniform",
    "kl_after_cmt",
    "delta_uniform",
    "delta_cmt",
    "paired_gap",
    "uniform_update_loss",
    "cmt_update_loss",
    "probe_time_sec",
    "uniform_branch_time_sec",
    "cmt_eval_time_sec",
)

_INTEGER_FIELDS = {
    "optimizer_step",
    "rollout_id",
    "ppo_group_index",
    "subset_size",
    "valid_prefix_token_count",
    "top_k",
}
_FLOAT_FIELDS = {
    "kl_before",
    "kl_after_uniform",
    "kl_after_cmt",
    "delta_uniform",
    "delta_cmt",
    "paired_gap",
    "uniform_update_loss",
    "cmt_update_loss",
    "probe_time_sec",
    "uniform_branch_time_sec",
    "cmt_eval_time_sec",
}


class OneStepKLProbeLogger:
    """Write a paired probe record atomically and idempotently per optimizer step."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        enabled: bool = True,
        is_main: bool = True,
        artifact_subdir: str = "one_step_kl_probe",
    ) -> None:
        self.enabled = bool(enabled)
        self.is_main = bool(is_main)
        self.root = Path(output_dir) / artifact_subdir
        self.jsonl_path = self.root / "one_step_kl_probe.jsonl"
        self.csv_path = self.root / "one_step_kl_probe.csv"

    @staticmethod
    def _validate(record: Mapping[str, Any]) -> dict[str, Any]:
        missing = [field for field in PROBE_FIELDS if field not in record]
        if missing:
            raise ValueError(f"Probe record is missing required fields: {missing}")
        row = {field: record[field] for field in PROBE_FIELDS}
        for field in _INTEGER_FIELDS:
            value = row[field]
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"Probe field {field} must be an integer")
        if row["optimizer_step"] < 1:
            raise ValueError("Probe optimizer_step must be positive")
        if row["valid_prefix_token_count"] < 1:
            raise ValueError("Probe valid_prefix_token_count must be positive")
        for field in _FLOAT_FIELDS:
            try:
                value = float(row[field])
            except (TypeError, ValueError) as error:
                raise ValueError(f"Probe field {field} must be numeric") from error
            if not math.isfinite(value):
                raise ValueError(f"Probe field {field} must be finite")
            row[field] = value

        expected = paired_improvement(
            row["kl_before"], row["kl_after_uniform"], row["kl_after_cmt"]
        )
        for field, value in expected.items():
            if not math.isclose(row[field], value, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(
                    f"Probe field {field}={row[field]} is inconsistent with {value}"
                )
        return row

    def _read_existing(self) -> dict[int, dict[str, Any]]:
        rows: dict[int, dict[str, Any]] = {}
        if not self.jsonl_path.is_file():
            return rows
        with self.jsonl_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = self._validate(json.loads(line))
                except (json.JSONDecodeError, ValueError) as error:
                    raise ValueError(
                        f"Malformed probe log {self.jsonl_path} line {line_number}"
                    ) from error
                step = row["optimizer_step"]
                if step in rows:
                    raise ValueError(
                        f"Duplicate optimizer_step {step} in {self.jsonl_path}"
                    )
                rows[step] = row
        return rows

    @staticmethod
    def _temporary(path: Path) -> Path:
        return path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")

    def upsert(self, record: Mapping[str, Any]) -> tuple[Path, Path] | None:
        if not self.enabled or not self.is_main:
            return None
        row = self._validate(record)
        self.root.mkdir(parents=True, exist_ok=True)
        rows = self._read_existing()
        rows[row["optimizer_step"]] = row
        ordered = [rows[step] for step in sorted(rows)]

        jsonl_temporary = self._temporary(self.jsonl_path)
        csv_temporary = self._temporary(self.csv_path)
        try:
            with jsonl_temporary.open("x", encoding="utf-8") as handle:
                for item in ordered:
                    handle.write(json.dumps(item, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            with csv_temporary.open("x", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=PROBE_FIELDS)
                writer.writeheader()
                writer.writerows(ordered)
                handle.flush()
                os.fsync(handle.fileno())
            jsonl_temporary.replace(self.jsonl_path)
            csv_temporary.replace(self.csv_path)
        except BaseException:
            jsonl_temporary.unlink(missing_ok=True)
            csv_temporary.unlink(missing_ok=True)
            raise
        return self.jsonl_path, self.csv_path
