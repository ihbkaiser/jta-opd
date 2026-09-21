from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any, Iterable


def _atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _atomic_gzip_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


class LocalityLogger:
    def __init__(self, output_dir: str | Path, *, resume_step: int) -> None:
        self.root = Path(output_dir) / "analysis" / "locality"
        self.root.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.root / "metrics.jsonl"
        self.manifest_path = self.root / "manifest.json"
        self._write_manifest()
        self.truncate_after(int(resume_step))

    def _write_manifest(self) -> None:
        payload = {
            "schema_version": 1,
            "metrics": "metrics.jsonl",
            "token_samples": "token_samples/step-XXXXXX.jsonl.gz",
            "matched_pairs": "matched_pairs/step-XXXXXX.jsonl.gz",
        }
        temporary = self.manifest_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(self.manifest_path)

    def _metrics(self) -> list[dict[str, Any]]:
        if not self.metrics_path.exists():
            return []
        return [
            json.loads(line)
            for line in self.metrics_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def truncate_after(self, checkpoint_step: int) -> int:
        rows = self._metrics()
        retained = [row for row in rows if int(row["step"]) <= int(checkpoint_step)]
        removed = len(rows) - len(retained)
        if removed:
            _atomic_jsonl(self.metrics_path, retained)
        return removed

    def write_metrics(self, row: dict[str, Any]) -> Path:
        if "step" not in row:
            raise ValueError("Locality metric rows require an optimizer step")
        step = int(row["step"])
        rows = [item for item in self._metrics() if int(item["step"]) != step]
        rows.append(dict(row))
        rows.sort(key=lambda item: int(item["step"]))
        _atomic_jsonl(self.metrics_path, rows)
        return self.metrics_path

    def write_token_samples(
        self,
        step: int,
        records: list[dict[str, Any]],
        *,
        sample_size: int,
    ) -> list[dict[str, Any]]:
        target = max(0, min(int(sample_size), len(records)))
        groups: dict[int, list[dict[str, Any]]] = {}
        for row in records:
            groups.setdefault(int(row["g_decile"]), []).append(row)
        for rows in groups.values():
            rows.sort(key=lambda row: (int(row.get("flat_index", 0)), str(row.get("sample_id", ""))))
        selected: list[dict[str, Any]] = []
        group_keys = sorted(groups)
        cursor = {key: 0 for key in group_keys}
        while len(selected) < target:
            progressed = False
            for key in group_keys:
                index = cursor[key]
                if index < len(groups[key]) and len(selected) < target:
                    selected.append(dict(groups[key][index]))
                    cursor[key] += 1
                    progressed = True
            if not progressed:
                break
        path = self.root / "token_samples" / f"step-{int(step):06d}.jsonl.gz"
        _atomic_gzip_jsonl(path, selected)
        return selected

    def write_matched_pairs(
        self, step: int, pairs: list[dict[str, Any]]
    ) -> Path:
        path = self.root / "matched_pairs" / f"step-{int(step):06d}.jsonl.gz"
        _atomic_gzip_jsonl(path, pairs)
        return path
