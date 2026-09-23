from types import SimpleNamespace

import torch

from b200_experiment.scoring import RolloutBatch
from b200_experiment.trajectory_probe import OnPolicyTrajectoryBuilder


class _Distributed:
    rank = 0
    world_size = 1

    @staticmethod
    def all_gather_objects(value):
        return [value]


def test_builder_repeats_roots_and_forwards_locked_sampling_settings():
    calls = []

    def generate(_student, prompt_ids, prompt_mask, **kwargs):
        calls.append((prompt_ids.clone(), prompt_mask.clone(), dict(kwargs)))
        rows = prompt_ids.shape[0]
        response = torch.arange(rows).reshape(rows, 1) + kwargs["sample_seed_offset"]
        valid = torch.ones_like(response, dtype=torch.bool)
        return RolloutBatch(
            input_ids=torch.cat((prompt_ids, response), dim=1),
            attention_mask=torch.cat((prompt_mask, valid.long()), dim=1),
            response_ids=response,
            valid_mask=valid,
            rollout_log_probs=torch.zeros(rows, 1),
            prompt_width=prompt_ids.shape[1],
        )

    roots = SimpleNamespace(
        prompt_ids=torch.tensor([[0, 1], [2, 3]]),
        attention_mask=torch.tensor([[0, 1], [1, 1]]),
        problem_ids=("a", "b"),
        root_hash="root-hash",
    )
    builder = OnPolicyTrajectoryBuilder(
        _Distributed(),
        roots=roots,
        num_rollouts_per_problem=2,
        horizon=64,
        seed=20260923,
        temperature=1.0,
        top_p=1.0,
        eos_token_id=9,
        pad_token_id=0,
        generation_batch_size=3,
        generate_fn=generate,
    )

    result = builder.build("student", optimizer_step=50)

    assert result.problem_ids == ("a", "b")
    assert result.rollout.input_ids.shape[0] == 4
    assert len(calls) == 2
    assert calls[0][0].tolist() == [[0, 1], [0, 1], [2, 3]]
    assert calls[1][0].tolist() == [[2, 3]]
    assert all(call[2]["max_new_tokens"] == 64 for call in calls)
    assert all(call[2]["temperature"] == 1.0 for call in calls)
    assert all(call[2]["top_p"] == 1.0 for call in calls)
    assert calls[0][2]["sample_seed_offset"] == 0
    assert calls[1][2]["sample_seed_offset"] == 3
    assert result.sampling_seed == 20260923 + 1_000_003 * 50
    assert len(result.trajectory_hash) == 64
