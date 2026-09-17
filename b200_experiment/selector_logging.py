from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import torch


_CMT_ROW_KEYS = (
    "gain",
    "support_reverse_kl",
    "sampled_raw_student_prob",
    "sampled_raw_teacher_prob",
    "sampled_cond_student_prob",
    "sampled_cond_teacher_prob",
    "sampled_log_ratio",
    "sampled_cond_r",
    "sampled_conditional_log_ratio",
    "conditional_log_ratio_mean",
    "student_union_mass",
    "teacher_union_mass",
    "teacher_tail_mass",
    "support_common_mass",
    "conditional_support_common_mass",
    "support_coverage",
    "support_width",
    "in_support",
    "teacher_deficit",
    "transition_weight",
    "marginal_flux",
    "flux_product_form",
    "flux_identity_error",
    "successor_return",
    "successor_mass",
    "R",
    "M",
    "V",
    "H",
    "successor_value",
    "successor_contrast",
    "baseline_mass_term",
    "successor_excess",
    "x_difference_form",
    "x_product_form",
    "x_difference_identity_error",
    "x_product_identity_error",
    "x_cancellation_ratio",
    "sequential_gain",
    "successor_R",
    "d_product_form",
    "d_identity_error",
    "learning_value",
    "gamma",
    "response_position_fraction",
    "w",
    "selected_mask",
    "token_loss",
    "unweighted_token_loss",
    "token_advantage",
    "ppo_ratio",
    "token_clipped",
    "weighted_loss_contribution",
    "abs_weighted_loss_contribution",
    "gradient_influence_proxy",
)


def _scalar_at(value: Any, batch: int, position: int) -> float | int | bool | None:
    if torch.is_tensor(value):
        if value.ndim == 0:
            return value.detach().item()
        if value.ndim >= 2:
            return value[batch, position].detach().item()
        return value[position].detach().item()
    if isinstance(value, (float, int, bool)):
        return value
    return None


class CMTDiagnosticsLogger:
    """Crash-safe detached CMT row/tail diagnostics.

    Detailed rows are written independently by every rank. Tail rows are
    selected from each rank's local top/bottom candidates and merged by rank 0;
    local top-k candidates are sufficient for an exact global top-k selection.
    No tensor is retained in autograd and logging is completely bypassed when
    both flags are disabled.
    """

    def __init__(
        self,
        output_dir: str | Path,
        tokenizer,
        *,
        rank: int = 0,
        world_size: int = 1,
        detailed_enabled: bool = False,
        tail_enabled: bool = False,
        tail_interval: int = 1,
        tail_top_k: int = 128,
        context_tokens: int = 16,
        fp64_check: bool = True,
        partial_horizons: tuple[int, ...] | list[int] = (16, 64, 256, 1024),
    ):
        self.root = Path(output_dir) / "cmt_diagnostics"
        self.tokenizer = tokenizer
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.detailed_enabled = bool(detailed_enabled)
        self.tail_enabled = bool(tail_enabled)
        self.tail_interval = max(1, int(tail_interval))
        self.tail_top_k = max(1, int(tail_top_k))
        self.context_tokens = max(0, int(context_tokens))
        self.fp64_check = bool(fp64_check)
        self.partial_horizons = tuple(
            sorted({max(1, int(h)) for h in partial_horizons})
        )
        self.enabled = self.detailed_enabled or self.tail_enabled
        if self.enabled:
            self.root.mkdir(parents=True, exist_ok=True)
            if self.rank == 0:
                manifest = {
                    "format": "gzip JSONL, one crash-safe file per step/rank",
                    "method": "cmt",
                    "detailed_file_pattern": "detailed_step-*_rank-*.jsonl.gz",
                    "tail_file_pattern": "tail_step-*.jsonl.gz",
                    "tail_selection": "global union of deterministic local top/bottom candidates",
                    "partial_horizons": list(self.partial_horizons) + ["full"],
                    "row_fields": list(
                        dict.fromkeys(
                            list(_CMT_ROW_KEYS)
                            + [
                                "flux_product_form",
                                "flux_identity_error",
                                "rho",
                                "D",
                                "X",
                                "phi",
                                "abs_marginal_flux",
                                "abs_sequential_gain",
                                "sequential_gain_sign",
                                "abs_d_over_gain",
                                "successor_excess_fp64",
                                "x_fp32_fp64_abs_error",
                                "x_fp32_fp64_relative_error",
                                "allocation_mode",
                                "allocation_inverse_temperature",
                                "allocation_kl_epsilon",
                                "allocation_kl_achieved",
                                "allocation_top_fraction",
                                "allocation_threshold",
                            ]
                        )
                    ),
                    "fp64_suffix_definition": "direct finite-horizon suffix excess using production gamma and transition_weight",
                    "gradient_influence_proxy_definition": "abs(weighted_loss_contribution), where weighted_loss_contribution=w*detached token PPO loss; not a parameter-gradient norm",
                }
                temporary = self.root / ".manifest.json.tmp"
                temporary.write_text(
                    json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
                os.replace(temporary, self.root / "manifest.json")

    @staticmethod
    def _token_text(tokenizer, token_ids: list[int]) -> list[str]:
        try:
            return [str(item) for item in tokenizer.convert_ids_to_tokens(token_ids)]
        except Exception:
            return [str(item) for item in token_ids]

    def _row(
        self,
        *,
        step: int,
        batch: int,
        position: int,
        response_ids: torch.Tensor,
        valid_mask: torch.Tensor,
        dataset_indices: list[int],
        sample_ids: list[str],
        response_indices: list[int] | None,
        batch_index_offset: int,
        diagnostics: dict[str, Any],
        max_new_tokens: int,
        global_index: int,
    ) -> dict[str, Any]:
        ids = response_ids[batch].detach().cpu().long().tolist()
        valid_positions = valid_mask[batch].detach().cpu().bool()
        positions = [index for index, flag in enumerate(valid_positions) if flag]
        length = len(positions)
        token_texts = self._token_text(self.tokenizer, ids)
        context_before_all = [
            token_texts[index]
            for index in positions
            if index < position
        ]
        context_after_all = [
            token_texts[index]
            for index in positions
            if index > position
        ]
        context_before = (
            context_before_all[-self.context_tokens :] if self.context_tokens else []
        )
        context_after = (
            context_after_all[: self.context_tokens] if self.context_tokens else []
        )
        response_payload = torch.tensor(
            [ids[index] for index in positions], dtype=torch.int64
        ).numpy().tobytes()
        prompt_response_hash = hashlib.sha256(
            str(sample_ids[batch]).encode("utf-8") + b"\0" + response_payload
        ).hexdigest()
        row: dict[str, Any] = {
            "training_step": int(step),
            "dataset_index": int(dataset_indices[batch]),
            "sample_id": str(sample_ids[batch]),
            "response_index": int(response_indices[batch]) if response_indices else None,
            "trajectory_id": str(sample_ids[batch]),
            "batch_index": int(batch_index_offset) + int(batch),
            "response_position": int(position),
            "response_length": int(length),
            "remaining_valid_tokens": int(sum(1 for index in positions if index > position)),
            "hit_max_new_tokens": bool(length >= int(max_new_tokens)),
            "token_id": int(ids[position]),
            "token_text": token_texts[position] if position < len(token_texts) else str(ids[position]),
            "context_before": " ".join(context_before),
            "context_after": " ".join(context_after),
            "prompt_response_hash": prompt_response_hash,
            "global_token_index": int(global_index),
        }
        for key in _CMT_ROW_KEYS:
            value = _scalar_at(diagnostics.get(key), batch, position)
            if value is not None:
                row[key] = value
        # Formula aliases are deliberately numeric so downstream JSON/CSV
        # consumers can verify identities without parsing a string.
        row["flux_product_form"] = row.get(
            "flux_product_form", row.get("marginal_flux", 0.0)
        )
        row["flux_identity_error"] = abs(
            float(row.get("marginal_flux", 0.0))
            - float(row.get("flux_product_form", 0.0))
        )
        row["x_difference_form"] = row.get(
            "x_difference_form", row.get("successor_excess", 0.0)
        )
        row["x_product_form"] = row.get("x_product_form", row.get("successor_excess", 0.0))
        row["abs_marginal_flux"] = abs(float(row.get("marginal_flux", 0.0)))
        row["abs_sequential_gain"] = abs(float(row.get("sequential_gain", 0.0)))
        row["sequential_gain_sign"] = (
            1 if float(row.get("sequential_gain", 0.0)) > 0 else
            -1 if float(row.get("sequential_gain", 0.0)) < 0 else 0
        )
        row["abs_d_over_gain"] = abs(float(row.get("sequential_gain", 0.0))) / (
            float(row.get("gain", 0.0)) + 1.0e-8
        )
        row["D"] = float(row.get("sequential_gain", 0.0))
        row["X"] = float(row.get("successor_excess", 0.0))
        row["phi"] = float(row.get("marginal_flux", 0.0))
        row["gradient_influence_proxy"] = abs(
            float(row.get("weighted_loss_contribution", 0.0))
        )
        row["unweighted_token_loss"] = float(row.get("token_loss", 0.0))
        if self.fp64_check:
            fp64 = _scalar_at(diagnostics.get("successor_excess_fp64"), batch, position)
            if fp64 is None:
                fp64 = row.get("successor_excess", 0.0)
            fp32 = float(row.get("successor_excess", 0.0))
            fp64 = float(fp64)
            row["successor_excess_fp64"] = fp64
            row["x_fp32_fp64_abs_error"] = abs(fp32 - fp64)
            row["x_fp32_fp64_relative_error"] = abs(fp32 - fp64) / (
                abs(fp64) + 1.0e-8
            )
        # Allocation metadata is shared by all rows in a global step.  The
        # per-token rho is derived from the actual global valid-token count,
        # so consumers can reconstruct the allocation probability without
        # knowing the rank layout.
        return row

    def _write_rows(self, path: Path, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=6) as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        os.replace(temporary, path)

    def _partial_suffix(
        self,
        diagnostics: dict[str, Any],
        batch: int,
        position: int,
        max_new_tokens: int,
    ) -> dict[str, float]:
        g_tensor = diagnostics.get("gain")
        transition_tensor = diagnostics.get("transition_weight")
        valid_tensor = diagnostics.get("valid_mask")
        if not torch.is_tensor(g_tensor) or not torch.is_tensor(transition_tensor):
            return {}
        g = g_tensor[batch].detach().float().cpu().tolist()
        transition = transition_tensor[batch].detach().float().cpu().tolist()
        if torch.is_tensor(valid_tensor):
            valid = valid_tensor[batch].detach().bool().cpu().tolist()
        else:
            valid = [True] * len(g)
        stop = position + 1
        while stop < len(g) and valid[stop]:
            stop += 1
        result: dict[str, float] = {}
        horizons: list[int | str] = list(self.partial_horizons) + ["full"]
        for horizon in horizons:
            end = stop if horizon == "full" else min(stop, position + 1 + int(horizon))
            ret = 0.0
            mass = 0.0
            product = 1.0
            for index in range(position + 1, end):
                ret += product * float(g[index])
                mass += product
                product *= float(diagnostics.get("gamma", 1.0)) * float(transition[index])
            current_g = float(g[position]) if position < len(g) else 0.0
            suffix = str(horizon)
            result[f"successor_return_h{suffix}"] = ret
            result[f"successor_mass_h{suffix}"] = mass
            result[f"successor_excess_h{suffix}"] = ret - current_g * mass
            result[f"survival_product_h{suffix}"] = product
        return result

    def _local_candidates(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        groups = {
            "top_d": ("sequential_gain", True),
            "bottom_d": ("sequential_gain", False),
            "top_abs_d": ("abs_sequential_gain", True),
            "top_x": ("successor_excess", True),
            "bottom_x": ("successor_excess", False),
            "top_successor_mass": ("successor_mass", True),
            "top_flux": ("abs_marginal_flux", True),
            "top_learning_value": ("learning_value", True),
            "top_w": ("w", True),
        }
        by_index: dict[int, dict[str, Any]] = {}
        for reason, (field, descending) in groups.items():
            candidates = [
                row for row in rows if math.isfinite(float(row.get(field, 0.0)))
            ]
            candidates.sort(
                key=lambda row: (
                    -float(row.get(field, 0.0)) if descending else float(row.get(field, 0.0)),
                    int(row["global_token_index"]),
                )
            )
            for row in candidates[: self.tail_top_k]:
                target = by_index.setdefault(int(row["global_token_index"]), dict(row))
                reasons = set(target.get("tail_reasons", []))
                reasons.add(reason)
                target["tail_reasons"] = sorted(reasons)
        return list(by_index.values())

    @staticmethod
    def _global_rank(values: list[float], index: int, descending: bool) -> int:
        order = sorted(
            range(len(values)),
            key=lambda item: (
                -values[item] if descending else values[item], item
            ),
        )
        try:
            return order.index(index) + 1
        except ValueError:
            return len(values) + 1

    def write(
        self,
        *,
        step: int,
        response_ids: torch.Tensor,
        valid_mask: torch.Tensor,
        dataset_indices: list[int],
        sample_ids: list[str],
        response_indices: list[int] | None,
        diagnostics: dict[str, Any],
        batch_index_offset: int = 0,
        max_new_tokens: int = 0,
        distributed=None,
        global_diagnostics: dict[str, Any] | None = None,
        global_token_start: int = 0,
        allocation_metadata: dict[str, Any] | None = None,
    ) -> int:
        if not self.enabled:
            return 0
        diagnostics = dict(diagnostics)
        diagnostics.setdefault("valid_mask", valid_mask)
        valid_coordinates = valid_mask.detach().bool().nonzero(as_tuple=False).cpu().tolist()
        rows = [
            self._row(
                step=step,
                batch=int(batch),
                position=int(position),
                response_ids=response_ids,
                valid_mask=valid_mask,
                dataset_indices=dataset_indices,
                sample_ids=sample_ids,
                response_indices=response_indices,
                batch_index_offset=batch_index_offset,
                diagnostics=diagnostics,
                max_new_tokens=max_new_tokens,
                global_index=global_token_start + offset,
            )
            for offset, (batch, position) in enumerate(valid_coordinates)
        ]
        global_weight_source = (global_diagnostics or {}).get("w")
        if torch.is_tensor(global_weight_source):
            valid_count = int(global_weight_source.numel())
        else:
            valid_count = len(rows)
        if allocation_metadata:
            for row in rows:
                row.update(allocation_metadata)
        for row in rows:
            row["rho"] = float(row.get("w", 0.0)) / max(valid_count, 1)
        if self.detailed_enabled:
            path = self.root / f"detailed_step-{step:06d}_rank-{self.rank:05d}.jsonl.gz"
            self._write_rows(path, rows)
        if not self.tail_enabled or step % self.tail_interval != 0:
            return len(rows)
        local_candidates = self._local_candidates(rows)
        coordinate_by_global = {
            global_token_start + offset: (int(batch), int(position))
            for offset, (batch, position) in enumerate(valid_coordinates)
        }
        for row in local_candidates:
            coordinate = coordinate_by_global.get(int(row["global_token_index"]))
            if coordinate is not None:
                row.update(
                    self._partial_suffix(
                        diagnostics,
                        coordinate[0],
                        coordinate[1],
                        max_new_tokens,
                    )
                )
        gathered = (
            distributed.all_gather_objects(local_candidates)
            if distributed is not None and getattr(distributed, "enabled", False)
            else [local_candidates]
        )
        if self.rank != 0:
            return len(rows)
        merged: dict[int, dict[str, Any]] = {}
        for rank_rows in gathered:
            for row in rank_rows:
                index = int(row["global_token_index"])
                if index not in merged:
                    merged[index] = dict(row)
                else:
                    merged[index]["tail_reasons"] = sorted(
                        set(merged[index].get("tail_reasons", []))
                        | set(row.get("tail_reasons", []))
                    )
        global_rows = list(merged.values())
        full = global_diagnostics or diagnostics
        # Compute each global rank vector once.  Re-sorting the complete
        # rollout for every tail row is needlessly quadratic for 8k-token
        # responses; seven stable O(N log N) sorts are inexpensive and keep
        # tie-breaking deterministic by flattened token order.
        rank_vectors: dict[str, list[int] | None] = {}
        rank_specs = (
            ("sequential_gain", True, "global_rank_d"),
            ("abs_sequential_gain", True, "global_rank_abs_d"),
            ("successor_excess", True, "global_rank_x"),
            ("successor_mass", True, "global_rank_successor_mass"),
            ("abs_marginal_flux", True, "global_rank_flux"),
            ("learning_value", True, "global_rank_learning_value"),
            ("w", True, "global_rank_w"),
        )
        for field, descending, _output in rank_specs:
            source = full.get(field)
            if not torch.is_tensor(source):
                rank_vectors[field] = None
                continue
            values = source.detach().float().reshape(-1).cpu()
            order = torch.argsort(values, descending=descending, stable=True)
            ranks = torch.empty(order.numel(), dtype=torch.int64)
            ranks[order] = torch.arange(1, order.numel() + 1, dtype=torch.int64)
            rank_vectors[field] = ranks.tolist()
        for row in global_rows:
            index = int(row["global_token_index"])
            for field, _descending, output in rank_specs:
                ranks = rank_vectors[field]
                row[output] = ranks[index] if ranks is not None and index < len(ranks) else None
            if allocation_metadata:
                row.update(allocation_metadata)
        path = self.root / f"tail_step-{step:06d}.jsonl.gz"
        self._write_rows(path, sorted(global_rows, key=lambda item: int(item["global_token_index"])))
        return len(rows)


class SelectedTokenLogger:
    """Incremental, crash-safe gzip JSONL logger containing selected positions only."""

    def __init__(
        self,
        output_dir: str | Path,
        tokenizer,
        method: str,
        chunk_steps: int = 50,
        enabled: bool = True,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.root = Path(output_dir) / "selector_scores"
        self.tokenizer = tokenizer
        self.method = method
        self.chunk_steps = max(1, int(chunk_steps))
        self.enabled = bool(enabled)
        self.rank = int(rank)
        self.world_size = int(world_size)
        if self.enabled:
            self.root.mkdir(parents=True, exist_ok=True)
            manifest = {
                "format": "gzip JSONL; concatenated gzip members are valid",
                "scope": "selected/accepted response positions only",
                "method": method,
                "chunk_steps": self.chunk_steps,
                "distributed_world_size": self.world_size,
                "distributed_file_pattern": (
                    "selected_steps_*_rank-*.jsonl.gz"
                    if self.world_size > 1
                    else "selected_steps_*.jsonl.gz"
                ),
                "common_fields": [
                    "training_step",
                    "sample_id",
                    "dataset_index",
                    "batch_index",
                    "response_position",
                    "token_id",
                    "token_text",
                ],
                "score_fields": {
                    "opd": ["w"],
                    "ta": ["D", "C", "D_norm", "C_norm", "s_TA"],
                    "rac": ["g", "alignment", "R", "M", "V", "z", "w"],
                    "pgt": [
                        "gain",
                        "euclidean_gain",
                        "restricted_reverse_kl",
                        "student_union_mass",
                        "teacher_union_mass",
                        "teacher_tail_mass",
                        "s_PGT",
                    ],
                    "cmt": [
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
                    ],
                }[method],
            }
            if self.rank == 0:
                with (self.root / "manifest.json").open(
                    "w", encoding="utf-8"
                ) as handle:
                    json.dump(manifest, handle, indent=2, ensure_ascii=False)
                    handle.write("\n")

    def _path(self, step: int) -> Path:
        first = ((step - 1) // self.chunk_steps) * self.chunk_steps + 1
        last = first + self.chunk_steps - 1
        suffix = f"_rank-{self.rank:05d}" if self.world_size > 1 else ""
        return self.root / (f"selected_steps_{first:06d}_{last:06d}{suffix}.jsonl.gz")

    def write(
        self,
        *,
        step: int,
        dataset_indices: list[int],
        sample_ids: list[str],
        response_ids: torch.Tensor,
        selected_mask: torch.Tensor,
        diagnostics: dict[str, Any],
        batch_index_offset: int = 0,
    ) -> int:
        if not self.enabled:
            return 0
        coordinates = selected_mask.nonzero(as_tuple=False)
        if coordinates.numel() == 0:
            return 0
        coords_cpu = coordinates.detach().cpu()
        token_ids = response_ids[selected_mask].detach().cpu().tolist()
        token_texts = self.tokenizer.convert_ids_to_tokens(token_ids)
        keys = {
            "opd": ("w",),
            "ta": ("D", "C", "D_norm", "C_norm", "s_TA"),
            "rac": ("g", "alignment", "R", "M", "V", "z", "w"),
            "pgt": (
                "gain",
                "euclidean_gain",
                "restricted_reverse_kl",
                "student_union_mass",
                "teacher_union_mass",
                "teacher_tail_mass",
                "s_PGT",
            ),
            "cmt": (
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
            ),
        }[self.method]
        values = {
            key: diagnostics[key][selected_mask].detach().float().cpu().tolist()
            for key in keys
        }
        path = self._path(step)
        with gzip.open(path, "at", encoding="utf-8", compresslevel=6) as handle:
            for offset, (batch_index_tensor, position_tensor) in enumerate(coords_cpu):
                batch_index, position = int(batch_index_tensor), int(position_tensor)
                row = {
                    "training_step": step,
                    "sample_id": sample_ids[batch_index],
                    "dataset_index": int(dataset_indices[batch_index]),
                    "batch_index": int(batch_index_offset) + batch_index,
                    "response_position": position,
                    "token_id": int(token_ids[offset]),
                    "token_text": str(token_texts[offset]),
                }
                row.update({key: float(values[key][offset]) for key in keys})
                handle.write(
                    json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
                )
        return len(token_ids)


class TokenScoreStatsLogger:
    """Compact all-valid-token histograms, quantiles, and bounded samples."""

    def __init__(
        self,
        output_dir: str | Path,
        method: str,
        interval: int = 50,
        bins: int = 64,
        raw_sample_size: int = 2048,
        enabled: bool = True,
    ):
        self.root = Path(output_dir) / "token_score_stats"
        self.method = method
        self.interval = max(1, int(interval))
        self.bins = max(2, int(bins))
        self.raw_sample_size = max(0, int(raw_sample_size))
        self.enabled = bool(enabled)
        if method == "ta":
            self.ranges = {
                "D": (0.0, 10.0),
                "C": (0.0, 1.0),
                "s_TA": (0.0, 1.0),
            }
        elif method == "rac":
            self.ranges = {key: (0.0, 1.0) for key in ("g", "alignment", "V", "z", "w")}
        elif method == "opd":
            self.ranges = {"w": (0.0, 1.0)}
        elif method == "iw":
            self.ranges = {
                "w": (0.0, 1.0),
                "iw_weight": (1.0, 1.5),
            }
        elif method == "pgt":
            self.ranges = {
                "s_PGT": (0.0, 10.0),
                "gain": (0.0, 10.0),
                "euclidean_gain": (0.0, 10.0),
                "restricted_reverse_kl": (-10.0, 10.0),
                "student_union_mass": (0.0, 1.0),
                "teacher_union_mass": (0.0, 1.0),
                "teacher_tail_mass": (0.0, 1.0),
            }
        elif method == "cmt":
            self.ranges = {
                "s_CMT": (-100.0, 100.0),
                "gain": (0.0, 10.0),
                "support_reverse_kl": (0.0, 20.0),
                "support_common_mass": (0.0, 1.0),
                "conditional_support_common_mass": (0.0, 1.0),
                "alignment": (0.0, 1.0),
                "transition_weight": (0.0, 1.0),
                "support_coverage": (0.0, 1.0),
                "coverage_correction": (0.0, 1.0),
                "teacher_deficit": (0.0, 1.0),
                "marginal_flux": (-100.0, 100.0),
                "common_mass_derivative": (-10.0, 10.0),
                "R": (-100.0, 100.0),
                # A gamma=1 finite response can have one unit of occupancy per
                # remaining token.  Keep the normal B200 8k context visible;
                # raw max/overflow still preserve runs longer than this.
                "M": (0.0, 8192.0),
                "V": (-100.0, 100.0),
                "H": (-100.0, 100.0),
                "successor_excess": (-100.0, 100.0),
                "sequential_gain": (-100.0, 100.0),
                "learning_value": (-100.0, 100.0),
                "w": (0.0, 20.0),
                "full_log_ratio_mean": (-20.0, 20.0),
                "full_log_ratio_variance": (0.0, 100.0),
                "full_common_mass": (0.0, 1.0),
                "sampled_raw_student_prob": (0.0, 1.0),
                "sampled_raw_teacher_prob": (0.0, 1.0),
                "sampled_cond_student_prob": (0.0, 1.0),
                "sampled_cond_teacher_prob": (0.0, 1.0),
                "sampled_log_ratio": (-20.0, 20.0),
                "sampled_cond_r": (-20.0, 20.0),
                "sampled_conditional_log_ratio": (-20.0, 20.0),
                "conditional_log_ratio_mean": (-20.0, 20.0),
                "successor_return": (-100.0, 100.0),
                "successor_mass": (0.0, 8192.0),
                "successor_value": (-100.0, 100.0),
                "successor_contrast": (-100.0, 100.0),
                "baseline_mass_term": (-100.0, 100.0),
                "flux_product_form": (-100.0, 100.0),
                "flux_identity_error": (0.0, 1.0),
                "x_difference_form": (-100.0, 100.0),
                "x_product_form": (-100.0, 100.0),
                "x_difference_identity_error": (0.0, 1.0),
                "x_product_identity_error": (0.0, 1.0),
                "x_cancellation_ratio": (0.0, 1.0e6),
                "d_product_form": (-100.0, 100.0),
                "d_identity_error": (-1.0, 1.0),
                "abs_marginal_flux": (0.0, 100.0),
                "abs_sequential_gain": (0.0, 100.0),
                "abs_d_over_gain": (0.0, 1.0e6),
                "response_position_fraction": (0.0, 1.0),
                "successor_excess_fp64": (-100.0, 100.0),
                "x_fp32_fp64_abs_error": (0.0, 1.0),
                "x_fp32_fp64_relative_error": (0.0, 1.0e6),
                "log10_abs_successor_excess": (0.0, 8.0),
                "log10_abs_sequential_gain": (0.0, 8.0),
            }
        else:
            raise ValueError(f"Unknown token-score method: {method!r}")
        if self.enabled:
            self.root.mkdir(parents=True, exist_ok=True)
            manifest = {
                "format": "one JSON file per logged optimizer step",
                "scope": "all valid response positions in the global rollout batch",
                "method": method,
                "logged_steps": "step 1, every configured interval, and final step",
                "bins": self.bins,
                "histogram_ranges": self.ranges,
                "raw_sample_size_max": self.raw_sample_size,
                "note": "Raw values are never clipped for quantiles; histogram underflow/overflow counts retain tail mass.",
                "quantiles": [
                    0.05, 0.25, 0.50, 0.75, 0.90, 0.95,
                    0.99, 0.995, 0.999, 0.9999,
                ],
            }
            (self.root / "manifest.json").write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

    def should_log(self, step: int, final_step: int) -> bool:
        return self.enabled and (
            step == 1 or step == final_step or step % self.interval == 0
        )

    def write(
        self, step: int, final_step: int, diagnostics: dict[str, Any]
    ) -> Path | None:
        if not self.should_log(step, final_step):
            return None
        payload: dict[str, Any] = {
            "step": int(step),
            "method": self.method,
            "scope": "global_valid_response_tokens",
            "scores": {},
        }
        diagnostics = dict(diagnostics)
        if self.method == "cmt":
            successor_excess = diagnostics.get("successor_excess")
            sequential_gain = diagnostics.get("sequential_gain")
            if torch.is_tensor(successor_excess):
                diagnostics["log10_abs_successor_excess"] = torch.log10(
                    1.0 + successor_excess.detach().float().abs()
                )
            if torch.is_tensor(sequential_gain):
                diagnostics["log10_abs_sequential_gain"] = torch.log10(
                    1.0 + sequential_gain.detach().float().abs()
                )
        quantile_levels = torch.tensor(
            [0.05, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 0.995, 0.999, 0.9999],
            dtype=torch.float64,
            device=next(
                value.device for value in diagnostics.values() if torch.is_tensor(value)
            ),
        )
        for key, (low, high) in self.ranges.items():
            if key not in diagnostics:
                continue
            # Keep FP64 audit values intact in raw summaries and overflow
            # accounting; converting to FP32 here could turn a valid large
            # cancellation ratio into +inf.
            values = diagnostics[key].detach().to(torch.float64).reshape(-1)
            values = values[torch.isfinite(values)]
            if values.numel() == 0:
                continue
            clipped = values.clamp(low, high)
            # Histogram kernels are most portable in FP32; clipping is only
            # for bin assignment while raw FP64 values remain below.
            counts = torch.histc(clipped.float(), bins=self.bins, min=low, max=high)
            edges = torch.linspace(
                low, high, self.bins + 1, device=values.device, dtype=torch.float64
            )
            quantiles = torch.quantile(values, quantile_levels)
            sample_count = min(values.numel(), self.raw_sample_size)
            if sample_count:
                indices = torch.linspace(
                    0,
                    values.numel() - 1,
                    sample_count,
                    device=values.device,
                ).long()
                sample = values.index_select(0, indices)
            else:
                sample = values.new_empty((0,))
            payload["scores"][key] = {
                "count": int(values.numel()),
                "mean": float(values.mean()),
                "min": float(values.min()),
                "max": float(values.max()),
                "quantiles": {
                    name: float(value)
                    for name, value in zip(
                        (
                            "q05", "q25", "q50", "q75", "q90", "q95",
                            "q99", "q99.5", "q99.9", "q99.99",
                        ),
                        quantiles,
                    )
                },
                "histogram": {
                    "edges": edges.cpu().tolist(),
                    "counts": counts.long().cpu().tolist(),
                    "underflow": int((values < low).sum()),
                    "overflow": int((values > high).sum()),
                },
                "sample": sample.cpu().tolist(),
            }
        if self.method == "cmt":
            self._write_cmt_conditional_statistics(payload, diagnostics)
            self._write_cmt_concentration_statistics(payload, diagnostics)
        destination = self.root / f"step-{step:06d}.json"
        temporary = destination.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)
        return destination

    @staticmethod
    def _spearman(x: torch.Tensor, y: torch.Tensor) -> float | None:
        if x.numel() < 2:
            return None
        x = torch.argsort(torch.argsort(x, stable=True), stable=True).float()
        y = torch.argsort(torch.argsort(y, stable=True), stable=True).float()
        x = x - x.mean()
        y = y - y.mean()
        denominator = x.square().sum().sqrt() * y.square().sum().sqrt()
        if float(denominator) == 0.0:
            return None
        return float((x * y).sum().div(denominator).item())

    def _write_cmt_conditional_statistics(
        self, payload: dict[str, Any], diagnostics: dict[str, Any]
    ) -> None:
        gain = diagnostics.get("gain")
        if not torch.is_tensor(gain):
            return
        base = torch.ones_like(gain, dtype=torch.bool).reshape(-1)
        masks: dict[str, torch.Tensor] = {"all_valid": base}
        for name, key, predicate in (
            ("teacher_deficit", "teacher_deficit", lambda x: x > 0),
            ("active_flux", "marginal_flux", lambda x: x != 0),
            ("positive_D", "sequential_gain", lambda x: x > 0),
            ("negative_D", "sequential_gain", lambda x: x < 0),
            ("clipped_response", "response_clipped", lambda x: x > 0),
            ("unclipped_response", "response_clipped", lambda x: x <= 0),
        ):
            value = diagnostics.get(key)
            if torch.is_tensor(value) and value.numel() == base.numel():
                masks[name] = predicate(value.detach().reshape(-1).float())
        position = diagnostics.get("response_position_fraction")
        if torch.is_tensor(position) and position.numel() == base.numel():
            flat_position = position.detach().reshape(-1).to(torch.float64)
            for name, low, high in (
                ("position_0_10", 0.0, 0.10),
                ("position_10_25", 0.10, 0.25),
                ("position_25_50", 0.25, 0.50),
                ("position_50_75", 0.50, 0.75),
                ("position_75_90", 0.75, 0.90),
                ("position_90_100", 0.90, 1.01),
            ):
                masks[name] = (flat_position >= low) & (flat_position < high)
        fields = (
            "gain", "marginal_flux", "successor_return", "successor_mass",
            "successor_value", "successor_contrast", "successor_excess",
            "sequential_gain", "learning_value", "w", "x_cancellation_ratio",
        )
        conditional: dict[str, Any] = {}
        for name, mask in masks.items():
            conditional[name] = {}
            for field in fields:
                value = diagnostics.get(field)
                if not torch.is_tensor(value) or value.numel() != base.numel():
                    continue
                selected = value.detach().to(torch.float64).reshape(-1)[mask]
                if selected.numel() == 0:
                    continue
                finite = selected[torch.isfinite(selected)]
                if finite.numel() == 0:
                    continue
                q = torch.quantile(
                    finite,
                    torch.tensor(
                        [0.50, 0.90, 0.95, 0.99, 0.995, 0.999, 0.9999],
                        dtype=torch.float64,
                        device=finite.device,
                    ),
                )
                conditional[name][field] = {
                    "count": int(finite.numel()),
                    "min": float(finite.min()),
                    "max": float(finite.max()),
                    "mean": float(finite.mean()),
                    "std": float(finite.std(unbiased=False)),
                    "quantiles": {
                        label: float(item)
                        for label, item in zip(
                            ("q50", "q90", "q95", "q99", "q99.5", "q99.9", "q99.99"),
                            q,
                        )
                    },
                }
        payload["conditional_statistics"] = conditional
        flux = diagnostics.get("marginal_flux")
        d = diagnostics.get("sequential_gain")
        mass = diagnostics.get("successor_mass")
        contrast = diagnostics.get("successor_contrast")
        if all(torch.is_tensor(value) for value in (flux, d, mass, contrast)):
            active = (
                base
                & flux.reshape(-1).ne(0)
                & torch.isfinite(flux.reshape(-1))
                & torch.isfinite(d.reshape(-1))
            )
            active &= torch.isfinite(mass.reshape(-1)) & torch.isfinite(contrast.reshape(-1))
            log_d = torch.log1p(d.reshape(-1)[active].abs())
            payload["factor_correlations"] = {
                "spearman_log1p_abs_flux_vs_log1p_abs_D": self._spearman(
                    torch.log1p(flux.reshape(-1)[active].abs()), log_d
                ),
                "spearman_log1p_successor_mass_vs_log1p_abs_D": self._spearman(
                    torch.log1p(mass.reshape(-1)[active].abs()), log_d
                ),
                "spearman_log1p_abs_successor_contrast_vs_log1p_abs_D": self._spearman(
                    torch.log1p(contrast.reshape(-1)[active].abs()), log_d
                ),
            }
            # Which factor actually contains the extreme |D| tokens?  Report
            # overlap on a deterministic 0.1% tail (the same convention used
            # by the influence diagnostics); no clipping or extra allocator
            # hyperparameter is involved.
            factors = {
                "abs_marginal_flux": flux.reshape(-1).abs(),
                "successor_mass": mass.reshape(-1),
                "abs_successor_contrast": contrast.reshape(-1).abs(),
            }
            finite = (
                flux.reshape(-1).ne(0)
                & torch.isfinite(d.reshape(-1))
                & torch.isfinite(flux.reshape(-1))
                & torch.isfinite(mass.reshape(-1))
                & torch.isfinite(contrast.reshape(-1))
            )
            positions = torch.nonzero(finite, as_tuple=False).flatten()
            if positions.numel():
                d_values = d.reshape(-1).abs().index_select(0, positions)
                tail_count = max(1, math.ceil(0.001 * positions.numel()))
                d_tail = torch.argsort(d_values, descending=True, stable=True)[:tail_count]
                d_tail = positions.index_select(0, d_tail)
                overlap = {}
                for name, factor in factors.items():
                    factor_values = factor.index_select(0, positions)
                    factor_tail = torch.argsort(
                        factor_values, descending=True, stable=True
                    )[:tail_count]
                    factor_tail = positions.index_select(0, factor_tail)
                    # Global token coordinates are represented by flattened
                    # positions at this point, so this intersection is exact.
                    overlap[name] = float(
                        torch.isin(d_tail, factor_tail).float().mean()
                    )
                payload["top_0p1_percent_overlap"] = overlap

    def _write_cmt_concentration_statistics(
        self, payload: dict[str, Any], diagnostics: dict[str, Any]
    ) -> None:
        weights = diagnostics.get("w")
        if not torch.is_tensor(weights):
            return
        weights = weights.detach().float().reshape(-1)
        finite = weights[torch.isfinite(weights) & (weights >= 0)]
        if finite.numel() == 0:
            return
        total = finite.sum().clamp_min(1.0e-12)
        ordered = torch.sort(finite, descending=True).values
        n = int(finite.numel())
        top_mass = {
            "top_1": float(ordered[:1].sum() / total),
            "top_10": float(ordered[:10].sum() / total),
            "top_100": float(ordered[:100].sum() / total),
            "top_0.1_percent": float(ordered[: max(1, math.ceil(0.001 * n))].sum() / total),
        }
        ess = float(total.square() / finite.square().sum().clamp_min(1.0e-12))
        probabilities = finite / total
        entropy = float(-(probabilities * probabilities.clamp_min(1.0e-30).log()).sum())
        payload["allocation_statistics"] = {
            "allocation_mode": diagnostics.get("allocation_mode", "gibbs"),
            "allocation_inverse_temperature": diagnostics.get("allocation_inverse_temperature"),
            "allocation_kl_epsilon": diagnostics.get("allocation_kl_epsilon"),
            "allocation_kl_achieved": diagnostics.get("allocation_kl_achieved"),
            "valid_token_count": n,
            "w_max": float(finite.max()),
            "max_token_probability": float(finite.max() / total),
            "top_weight_mass": top_mass,
            "fractions": {
                "w_lt_1e-6": float((finite < 1.0e-6).float().mean()),
                "w_gt_10": float((finite > 10).float().mean()),
                "w_gt_100": float((finite > 100).float().mean()),
                "w_gt_1000": float((finite > 1000).float().mean()),
            },
            "effective_sample_size": ess,
            "normalized_ess": ess / max(n, 1),
            "normalized_weight_entropy": entropy,
        }
