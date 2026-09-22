from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Callable

import torch

from b200_experiment.distributed import contiguous_partition
from b200_experiment.scoring import BaseScores, RolloutBatch, score_original_rollout


@dataclass(frozen=True)
class SuccessorProbeBatch:
    rollout: RolloutBatch
    parent_state_count: int
    successors_per_parent: int
    successor_state_count: int
    state_hash: str
    sampling_seed: int


def _query_rollout(prefixes: list[torch.Tensor], pad_token_id: int) -> RolloutBatch:
    if not prefixes:
        raise ValueError("Successor probe received no state prefixes")
    device = prefixes[0].device
    width = max(int(prefix.numel()) for prefix in prefixes)
    prompt_ids = torch.full(
        (len(prefixes), width),
        int(pad_token_id),
        dtype=torch.long,
        device=device,
    )
    prompt_mask = torch.zeros_like(prompt_ids)
    for row, prefix in enumerate(prefixes):
        length = int(prefix.numel())
        prompt_ids[row, width - length :] = prefix
        prompt_mask[row, width - length :] = 1
    dummy = torch.full(
        (len(prefixes), 1),
        int(pad_token_id),
        dtype=torch.long,
        device=device,
    )
    valid = torch.ones((len(prefixes), 1), dtype=torch.bool, device=device)
    return RolloutBatch(
        input_ids=torch.cat((prompt_ids, dummy), dim=1),
        attention_mask=torch.cat((prompt_mask, valid.long()), dim=1),
        response_ids=dummy,
        valid_mask=valid,
        rollout_log_probs=torch.zeros(
            (len(prefixes), 1), dtype=torch.float32, device=device
        ),
        prompt_width=width,
    )


def _rollout_hash(rollout: RolloutBatch) -> str:
    digest = hashlib.sha256()
    for value in (rollout.input_ids, rollout.attention_mask, rollout.valid_mask):
        tensor = value.detach().cpu().contiguous()
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


class SuccessorStateBuilder:
    """Sample frozen one-token successors from the current pre-update student."""

    def __init__(
        self,
        distributed,
        *,
        parent_state_count: int,
        successors_per_parent: int,
        seed: int,
        sampling_temperature: float,
        pad_token_id: int,
        score_micro_batch_size: int = 1,
        score_chunk_steps: int = 128,
        score_fn: Callable[..., BaseScores] = score_original_rollout,
    ) -> None:
        if int(parent_state_count) <= 0:
            raise ValueError("parent_state_count must be positive")
        if int(successors_per_parent) <= 0:
            raise ValueError("successors_per_parent must be positive")
        if float(sampling_temperature) <= 0.0:
            raise ValueError("sampling_temperature must be positive")
        self.distributed = distributed
        self.parent_state_count = int(parent_state_count)
        self.successors_per_parent = int(successors_per_parent)
        self.seed = int(seed)
        self.sampling_temperature = float(sampling_temperature)
        self.pad_token_id = int(pad_token_id)
        self.score_micro_batch_size = int(score_micro_batch_size)
        self.score_chunk_steps = int(score_chunk_steps)
        self.score_fn = score_fn

    def _select_parent_prefixes(
        self,
        rollout: RolloutBatch,
        objective_valid_mask: torch.Tensor,
        optimizer_step: int,
    ) -> list[torch.Tensor]:
        valid = objective_valid_mask.to(
            device=rollout.valid_mask.device, dtype=torch.bool
        )
        if valid.shape != rollout.valid_mask.shape:
            raise ValueError("Successor parent mask must align with rollout.valid_mask")
        valid &= rollout.valid_mask.bool()
        begin, end = contiguous_partition(
            self.parent_state_count,
            int(self.distributed.rank),
            int(self.distributed.world_size),
        )
        target = end - begin
        positions = valid.nonzero(as_tuple=False)
        if int(positions.shape[0]) < target:
            raise ValueError(
                "Successor probe rank has fewer valid rollout states than its "
                f"requested share: available={positions.shape[0]}, requested={target}"
            )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(
            self.seed + 1_000_003 * int(optimizer_step) + 10_007 * int(self.distributed.rank)
        )
        order = torch.randperm(int(positions.shape[0]), generator=generator)[:target]
        selected = positions.index_select(0, order.to(positions.device))
        prefixes: list[torch.Tensor] = []
        for row, response_position in selected.tolist():
            stop = int(rollout.prompt_width) + int(response_position)
            tokens = rollout.input_ids[row, :stop]
            attention = rollout.attention_mask[row, :stop].bool()
            prefix = tokens[attention].detach().clone().long()
            if prefix.numel() == 0:
                raise ValueError("Successor probe selected an empty parent state")
            prefixes.append(prefix)
        return prefixes

    def build(
        self,
        student,
        rollout: RolloutBatch,
        *,
        objective_valid_mask: torch.Tensor,
        optimizer_step: int,
    ) -> SuccessorProbeBatch:
        parent_prefixes = self._select_parent_prefixes(
            rollout, objective_valid_mask, optimizer_step
        )
        parent_queries = _query_rollout(parent_prefixes, self.pad_token_id)
        was_training = bool(getattr(student, "training", False))
        try:
            scores = self.score_fn(
                student,
                parent_queries,
                keep_cache=False,
                score_chunk_steps=self.score_chunk_steps,
                retain_response_logits=True,
                top_k=0,
                candidate_ids=None,
                temperature=self.sampling_temperature,
                micro_batch_size=self.score_micro_batch_size,
                trim_padding=True,
                length_bucketed=True,
            )
        finally:
            if hasattr(student, "train"):
                student.train(was_training)
        if scores.response_logits is None:
            raise RuntimeError("Successor sampler requires full-vocabulary logits")
        logits = scores.response_logits[:, 0].float() / self.sampling_temperature
        probabilities = torch.softmax(logits, dim=-1)
        successor_prefixes: list[torch.Tensor] = []
        base_seed = self.seed + 1_000_003 * int(optimizer_step)
        for local_parent, prefix in enumerate(parent_prefixes):
            global_parent = (
                int(self.distributed.rank) * self.parent_state_count + local_parent
            )
            generator = torch.Generator(device=probabilities.device)
            generator.manual_seed(base_seed + 97_409 * global_parent)
            actions = torch.multinomial(
                probabilities[local_parent],
                self.successors_per_parent,
                replacement=True,
                generator=generator,
            )
            successor_prefixes.extend(
                torch.cat((prefix, action.reshape(1).long())) for action in actions
            )
        successor_rollout = _query_rollout(successor_prefixes, self.pad_token_id)
        local_hash = _rollout_hash(successor_rollout)
        hashes = self.distributed.all_gather_objects(local_hash)
        global_hash = hashlib.sha256("|".join(hashes).encode("ascii")).hexdigest()
        local_parents = len(parent_prefixes)
        global_parents = self.distributed.sum_int(local_parents)
        global_successors = self.distributed.sum_int(len(successor_prefixes))
        return SuccessorProbeBatch(
            rollout=successor_rollout,
            parent_state_count=int(global_parents),
            successors_per_parent=self.successors_per_parent,
            successor_state_count=int(global_successors),
            state_hash=global_hash,
            sampling_seed=base_seed,
        )
