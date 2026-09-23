from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F

from b200_experiment.scoring import RolloutBatch, generate_on_policy


@dataclass(frozen=True)
class TrajectoryProbeBatch:
    rollout: RolloutBatch
    problem_ids: tuple[str, ...]
    num_rollouts_per_problem: int
    sampling_seed: int
    trajectory_hash: str


def _concatenate(
    prompt_ids: torch.Tensor,
    prompt_mask: torch.Tensor,
    batches: list[RolloutBatch],
    *,
    pad_token_id: int,
) -> RolloutBatch:
    if not batches:
        raise ValueError("Trajectory probe generation returned no batches")
    width = max(int(item.response_ids.shape[1]) for item in batches)
    responses, valid_masks, log_probs = [], [], []
    for item in batches:
        padding = width - int(item.response_ids.shape[1])
        responses.append(F.pad(item.response_ids, (0, padding), value=pad_token_id))
        valid_masks.append(F.pad(item.valid_mask, (0, padding), value=False))
        log_probs.append(
            F.pad(item.rollout_log_probs, (0, padding), value=float("nan"))
        )
    response_ids = torch.cat(responses, dim=0)
    valid = torch.cat(valid_masks, dim=0).bool()
    return RolloutBatch(
        input_ids=torch.cat((prompt_ids, response_ids), dim=1),
        attention_mask=torch.cat((prompt_mask, valid.long()), dim=1),
        response_ids=response_ids,
        valid_mask=valid,
        rollout_log_probs=torch.cat(log_probs, dim=0),
        prompt_width=int(prompt_ids.shape[1]),
    )


def _trajectory_digest(rollout: RolloutBatch) -> str:
    digest = hashlib.sha256()
    for value in (rollout.input_ids, rollout.attention_mask, rollout.valid_mask):
        tensor = value.detach().cpu().contiguous()
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


class OnPolicyTrajectoryBuilder:
    """Generate matched-seed branch trajectories from fixed held-out roots."""

    def __init__(
        self,
        distributed,
        *,
        roots,
        num_rollouts_per_problem: int,
        horizon: int,
        seed: int,
        temperature: float,
        top_p: float,
        eos_token_id: int | list[int],
        pad_token_id: int,
        generation_batch_size: int,
        generate_fn: Callable = generate_on_policy,
    ) -> None:
        self.distributed = distributed
        self.roots = roots
        self.num_rollouts_per_problem = int(num_rollouts_per_problem)
        self.horizon = int(horizon)
        self.seed = int(seed)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.eos_token_id = eos_token_id
        self.pad_token_id = int(pad_token_id)
        self.generation_batch_size = int(generation_batch_size)
        self.generate_fn = generate_fn

    def build(self, student, *, optimizer_step: int) -> TrajectoryProbeBatch:
        prompt_ids = self.roots.prompt_ids.repeat_interleave(
            self.num_rollouts_per_problem, dim=0
        )
        prompt_mask = self.roots.attention_mask.repeat_interleave(
            self.num_rollouts_per_problem, dim=0
        )
        sampling_seed = self.seed + 1_000_003 * int(optimizer_step)
        batches: list[RolloutBatch] = []
        was_training = bool(getattr(student, "training", False))
        try:
            for begin in range(0, int(prompt_ids.shape[0]), self.generation_batch_size):
                end = min(begin + self.generation_batch_size, int(prompt_ids.shape[0]))
                batches.append(
                    self.generate_fn(
                        student,
                        prompt_ids[begin:end],
                        prompt_mask[begin:end],
                        max_new_tokens=self.horizon,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        eos_token_ids=self.eos_token_id,
                        pad_token_id=self.pad_token_id,
                        seed=sampling_seed,
                        sample_seed_offset=begin,
                    )
                )
        finally:
            if hasattr(student, "train"):
                student.train(was_training)
        rollout = _concatenate(
            prompt_ids, prompt_mask, batches, pad_token_id=self.pad_token_id
        )
        local_hash = _trajectory_digest(rollout)
        gathered = self.distributed.all_gather_objects(local_hash)
        trajectory_hash = hashlib.sha256("|".join(gathered).encode("ascii")).hexdigest()
        return TrajectoryProbeBatch(
            rollout=rollout,
            problem_ids=tuple(self.roots.problem_ids),
            num_rollouts_per_problem=self.num_rollouts_per_problem,
            sampling_seed=sampling_seed,
            trajectory_hash=trajectory_hash,
        )
