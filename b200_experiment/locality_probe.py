from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Callable

import torch

from .evaluation import load_benchmark
from .locality_analysis import reverse_kl_on_fixed_support
from .scoring import RolloutBatch, score_original_rollout


def load_competition_math_test(
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    specs = config.get("evaluation", {}).get("benchmarks", {})
    if "Competition-MATH" not in specs:
        raise ValueError("Competition-MATH test benchmark must be configured")
    return load_benchmark("Competition-MATH", specs["Competition-MATH"])


def deterministic_prompt_sample(
    records: list[dict[str, Any]], num_prompts: int, *, seed: int
) -> list[dict[str, Any]]:
    requested = int(num_prompts)
    if requested <= 0:
        raise ValueError("num_prompts must be positive")
    if requested > len(records):
        raise ValueError(
            f"Requested {requested} probe prompts from only {len(records)} records"
        )
    indices = sorted(random.Random(int(seed)).sample(range(len(records)), requested))
    return [records[index] for index in indices]


def _normalized_prompt(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value)).strip().casefold()


def audit_prompt_overlap(
    train_records: list[dict[str, Any]],
    test_records: list[dict[str, Any]],
    *,
    train_prompt_key: str,
    test_prompt_key: str,
) -> dict[str, Any]:
    train_by_hash: dict[str, list[int]] = {}
    normalized_by_hash: dict[str, str] = {}
    for index, row in enumerate(train_records):
        normalized = _normalized_prompt(row[train_prompt_key])
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        train_by_hash.setdefault(digest, []).append(index)
        normalized_by_hash[digest] = normalized
    overlaps = []
    for index, row in enumerate(test_records):
        normalized = _normalized_prompt(row[test_prompt_key])
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        if digest in train_by_hash:
            overlaps.append(
                {
                    "sha256": digest,
                    "normalized_prompt": normalized_by_hash[digest],
                    "train_indices": train_by_hash[digest],
                    "test_index": index,
                    "test_id": str(row.get("id", index)),
                }
            )
    return {
        "schema_version": 1,
        "train_count": len(train_records),
        "test_count": len(test_records),
        "overlap_count": len(overlaps),
        "overlaps": overlaps,
    }


@dataclass
class FrozenStateProbe:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    response_ids: torch.Tensor
    valid_mask: torch.Tensor
    prompt_width: int
    candidate_ids: torch.Tensor
    support_mask: torch.Tensor
    teacher_log_probs: torch.Tensor
    metadata: list[dict[str, Any]]

    def __post_init__(self) -> None:
        batch, time = self.valid_mask.shape
        if self.input_ids.shape != self.attention_mask.shape:
            raise ValueError("Probe input_ids and attention_mask must align")
        if self.response_ids.shape != (batch, time):
            raise ValueError("Probe response_ids must align with valid_mask")
        expected_support = self.candidate_ids.shape
        if expected_support[:2] != (batch, time):
            raise ValueError("Probe candidate_ids must align with response states")
        if self.support_mask.shape != expected_support:
            raise ValueError("Probe support_mask must align with candidate_ids")
        if self.teacher_log_probs.shape != expected_support:
            raise ValueError("Probe teacher_log_probs must align with candidate_ids")
        if int(self.valid_mask.sum()) != len(self.metadata):
            raise ValueError("Probe metadata must contain one row per valid state")

    @staticmethod
    def tensor_fields() -> tuple[str, ...]:
        return tuple(
            field.name
            for field in fields(FrozenStateProbe)
            if field.name not in {"prompt_width", "metadata"}
        )

    @classmethod
    def from_scored_rollout(
        cls,
        rollout: RolloutBatch,
        scored_support: Any,
        *,
        metadata: list[dict[str, Any]],
    ) -> "FrozenStateProbe":
        required = (
            "candidate_ids",
            "support_mask",
            "teacher_candidate_log_probs",
        )
        missing = [name for name in required if not hasattr(scored_support, name)]
        if missing:
            raise ValueError(f"Scored support is missing fields: {missing}")
        return cls(
            input_ids=rollout.input_ids.detach().cpu().clone(),
            attention_mask=rollout.attention_mask.detach().cpu().clone(),
            response_ids=rollout.response_ids.detach().cpu().clone(),
            valid_mask=rollout.valid_mask.detach().cpu().bool().clone(),
            prompt_width=int(rollout.prompt_width),
            candidate_ids=scored_support.candidate_ids.detach().cpu().clone(),
            support_mask=scored_support.support_mask.detach().cpu().bool().clone(),
            teacher_log_probs=(
                scored_support.teacher_candidate_log_probs.detach().cpu().clone()
            ),
            metadata=[dict(item) for item in metadata],
        )

    def cpu(self) -> "FrozenStateProbe":
        return FrozenStateProbe(
            **{name: getattr(self, name).detach().cpu() for name in self.tensor_fields()},
            prompt_width=int(self.prompt_width),
            metadata=[dict(item) for item in self.metadata],
        )

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "prompt_width": int(self.prompt_width),
            "metadata": self.metadata,
            **{name: getattr(self, name).detach().cpu() for name in self.tensor_fields()},
        }
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(destination)
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "FrozenStateProbe":
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        if int(payload.pop("schema_version", 0)) != 1:
            raise ValueError("Unsupported frozen probe schema")
        return cls(**payload)

    def as_rollout(self, device: torch.device) -> RolloutBatch:
        return RolloutBatch(
            input_ids=self.input_ids.to(device),
            attention_mask=self.attention_mask.to(device),
            response_ids=self.response_ids.to(device),
            valid_mask=self.valid_mask.to(device),
            rollout_log_probs=torch.zeros_like(self.response_ids, dtype=torch.float32, device=device),
            prompt_width=int(self.prompt_width),
        )


class LocalityProbeRunner:
    def __init__(
        self,
        *,
        score_function: Callable[..., Any] = score_original_rollout,
        score_chunk_steps: int = 128,
        temperature: float = 1.0,
        trim_padding: bool = True,
        length_bucketed: bool = True,
    ) -> None:
        self.score_function = score_function
        self.score_chunk_steps = int(score_chunk_steps)
        self.temperature = float(temperature)
        self.trim_padding = bool(trim_padding)
        self.length_bucketed = bool(length_bucketed)
        self.previous_other_id_reverse_kl: float | None = None

    @torch.inference_mode()
    def score_frozen_states(
        self,
        student,
        probe: FrozenStateProbe,
        device: torch.device,
        *,
        micro_batch_size: int,
    ) -> torch.Tensor:
        rollout = probe.as_rollout(device)
        scores = self.score_function(
            student,
            rollout,
            keep_cache=False,
            score_chunk_steps=self.score_chunk_steps,
            retain_response_logits=False,
            top_k=0,
            candidate_ids=probe.candidate_ids.to(device),
            temperature=self.temperature,
            micro_batch_size=int(micro_batch_size),
            trim_padding=self.trim_padding,
            length_bucketed=self.length_bucketed,
        )
        if scores.candidate_log_probs is None:
            raise AssertionError("Student scoring did not return frozen candidates")
        return reverse_kl_on_fixed_support(
            scores.candidate_log_probs,
            probe.teacher_log_probs.to(device),
            probe.support_mask.to(device),
            probe.valid_mask.to(device),
        ).detach()

    score_training_states_post_update = score_frozen_states

    def initialize_other_id_metric(self, reverse_kl: float) -> None:
        value = float(reverse_kl)
        if not torch.isfinite(torch.tensor(value)):
            raise ValueError("Initial OTHER-ID reverse-KL must be finite")
        self.previous_other_id_reverse_kl = value

    def record_other_id_metric(self, reverse_kl: float) -> dict[str, float]:
        if self.previous_other_id_reverse_kl is None:
            raise RuntimeError("OTHER-ID pre-update metric has not been initialized")
        after = float(reverse_kl)
        before = self.previous_other_id_reverse_kl
        self.previous_other_id_reverse_kl = after
        return {
            "reverse_kl_before": before,
            "reverse_kl_after": after,
            "realized_gain": float(round(before - after, 15)),
        }


def write_probe_manifest(
    path: str | Path,
    *,
    probe_name: str,
    num_prompts: int,
    num_states: int,
    seed: int,
    max_new_tokens: int,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "probe_name": probe_name,
        "num_prompts": int(num_prompts),
        "num_states": int(num_states),
        "seed": int(seed),
        "max_new_tokens": int(max_new_tokens),
        "fixed_support": True,
    }
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return destination
