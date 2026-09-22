from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .scoring import RolloutBatch


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def select_heldout_records(
    records: list[dict[str, Any]], subset_size: int, seed: int
) -> list[dict[str, Any]]:
    size = int(subset_size)
    if size <= 0:
        raise ValueError("Held-out subset size must be positive")
    identifiers = [str(record.get("id", index)) for index, record in enumerate(records)]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("CompetitionMath held-out records contain a duplicate benchmark ID")
    if size > len(records):
        raise ValueError(
            f"Held-out subset size {size} exceeds benchmark size {len(records)}"
        )
    indices = sorted(random.Random(int(seed)).sample(range(len(records)), size))
    return [records[index] for index in indices]


def _tensor_digest(payload: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for name in ("input_ids", "attention_mask", "response_ids", "valid_mask"):
        tensor = payload.get(name)
        if not torch.is_tensor(tensor):
            raise ValueError(f"Held-out tensor artifact is missing {name}")
        value = tensor.detach().to(device="cpu").contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(_canonical_json(list(value.shape)))
        digest.update(value.view(torch.uint8).numpy().tobytes())
    prompt_width = int(payload.get("prompt_width", -1))
    if prompt_width <= 0:
        raise ValueError("Held-out tensor artifact has an invalid prompt_width")
    digest.update(str(prompt_width).encode("ascii"))
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _torch_load(path: Path) -> dict[str, Any]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, dict):
        raise ValueError(f"Held-out tensor artifact is not a dictionary: {path}")
    return value


@dataclass(frozen=True)
class HeldoutPrefixArtifact:
    rollout: RolloutBatch
    manifest: dict[str, Any]
    prefix_hash: str


class HeldoutPrefixStore:
    def __init__(self, root: str | Path, distributed):
        self.root = Path(root)
        self.distributed = distributed

    @property
    def manifest_path(self) -> Path:
        return self.root / "heldout_manifest.json"

    def rank_tensor_path(self, rank: int | None = None) -> Path:
        resolved_rank = int(self.distributed.rank if rank is None else rank)
        return self.root / f"heldout_prefixes.rank-{resolved_rank:05d}.pt"

    def save(
        self, rollout: RolloutBatch, manifest_fields: dict[str, Any]
    ) -> dict[str, Any]:
        self.root.mkdir(parents=True, exist_ok=True)
        payload = {
            "input_ids": rollout.input_ids.detach().to(device="cpu").contiguous(),
            "attention_mask": rollout.attention_mask.detach()
            .to(device="cpu")
            .contiguous(),
            "response_ids": rollout.response_ids.detach().to(device="cpu").contiguous(),
            "valid_mask": rollout.valid_mask.detach().to(device="cpu").contiguous(),
            "prompt_width": int(rollout.prompt_width),
        }
        local_hash = _tensor_digest(payload)
        tensor_path = self.rank_tensor_path()
        _atomic_torch_save(tensor_path, payload)
        local_summary = {
            "rank": int(self.distributed.rank),
            "rows": int(rollout.input_ids.shape[0]),
            "valid_tokens": int(rollout.valid_mask.long().sum().item()),
            "file": tensor_path.name,
            "sha256": local_hash,
        }
        summaries = sorted(
            self.distributed.all_gather_objects(local_summary),
            key=lambda item: int(item["rank"]),
        )
        aggregate_hash = hashlib.sha256(_canonical_json(summaries)).hexdigest()
        manifest = {
            **manifest_fields,
            "format_version": 1,
            "world_size": int(self.distributed.world_size),
            "rank_artifacts": summaries,
            "prefix_hash": aggregate_hash,
        }
        if self.distributed.is_main:
            _atomic_json(self.manifest_path, manifest)
        self.distributed.barrier()
        if not self.distributed.is_main:
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        return manifest

    def load(
        self,
        device: torch.device,
        *,
        expected: dict[str, Any] | None = None,
    ) -> HeldoutPrefixArtifact:
        if not self.manifest_path.is_file():
            raise FileNotFoundError(
                f"Held-out manifest does not exist: {self.manifest_path}"
            )
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if int(manifest.get("format_version", -1)) != 1:
            raise ValueError("Unsupported held-out manifest format")
        if int(manifest.get("world_size", -1)) != int(self.distributed.world_size):
            raise ValueError("Held-out manifest world size does not match this run")
        if expected:
            mismatches = {
                key: (manifest.get(key), value)
                for key, value in expected.items()
                if manifest.get(key) != value
            }
            if mismatches:
                raise ValueError(
                    f"held-out manifest does not match requested probe settings: {mismatches}"
                )
        rank_summaries = manifest.get("rank_artifacts")
        if not isinstance(rank_summaries, list):
            raise ValueError("Held-out manifest has no rank artifacts")
        aggregate_hash = hashlib.sha256(_canonical_json(rank_summaries)).hexdigest()
        if aggregate_hash != manifest.get("prefix_hash"):
            raise ValueError("Held-out manifest prefix hash mismatch")
        matches = [
            item
            for item in rank_summaries
            if int(item.get("rank", -1)) == int(self.distributed.rank)
        ]
        if len(matches) != 1:
            raise ValueError("Held-out manifest does not contain exactly one local rank")
        summary = matches[0]
        tensor_path = self.root / str(summary["file"])
        payload = _torch_load(tensor_path)
        actual_hash = _tensor_digest(payload)
        if actual_hash != summary.get("sha256"):
            raise ValueError(
                f"held-out prefix hash mismatch for rank {self.distributed.rank}"
            )
        rollout = RolloutBatch(
            input_ids=payload["input_ids"].to(device),
            attention_mask=payload["attention_mask"].to(device),
            response_ids=payload["response_ids"].to(device),
            valid_mask=payload["valid_mask"].to(device).bool(),
            rollout_log_probs=torch.full(
                payload["response_ids"].shape,
                torch.nan,
                dtype=torch.float32,
                device=device,
            ),
            prompt_width=int(payload["prompt_width"]),
        )
        return HeldoutPrefixArtifact(
            rollout=rollout,
            manifest=manifest,
            prefix_hash=str(manifest["prefix_hash"]),
        )
