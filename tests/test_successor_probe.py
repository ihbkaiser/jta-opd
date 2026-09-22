from __future__ import annotations

import torch
from torch import nn

from b200_experiment.scoring import BaseScores, RolloutBatch
from b200_experiment.successor_probe import SuccessorStateBuilder


class _Distributed:
    rank = 0
    world_size = 1

    @staticmethod
    def sum_int(value):
        return int(value)

    @staticmethod
    def all_gather_objects(value):
        return [value]


def _training_rollout() -> RolloutBatch:
    # Two left-padded prompts followed by three response positions.  Only the
    # first response has all three valid parent states.
    input_ids = torch.tensor(
        [
            [0, 10, 11, 20, 21, 22],
            [12, 13, 14, 30, 31, 0],
        ]
    )
    attention = torch.tensor(
        [
            [0, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 0],
        ]
    )
    response = input_ids[:, 3:]
    valid = torch.tensor([[True, True, True], [True, True, False]])
    return RolloutBatch(
        input_ids=input_ids,
        attention_mask=attention,
        response_ids=response,
        valid_mask=valid,
        rollout_log_probs=torch.zeros(2, 3),
        prompt_width=3,
    )


def _full_vocab_scores(_model, rollout, **kwargs):
    assert kwargs["retain_response_logits"] is True
    batch = rollout.input_ids.shape[0]
    vocab = 64
    logits = torch.full((batch, 1, vocab), -100.0)
    logits[:, :, 40] = 0.0
    logits[:, :, 41] = 0.0
    zeros = torch.zeros(batch, 1)
    return BaseScores(
        response_logits=logits,
        log_normalizers=zeros,
        scaled_log_normalizers=zeros,
        sampled_log_probs=zeros,
        entropies=zeros,
    )


def test_builder_samples_four_successors_for_each_selected_parent():
    builder = SuccessorStateBuilder(
        _Distributed(),
        parent_state_count=3,
        successors_per_parent=4,
        seed=17,
        sampling_temperature=1.0,
        pad_token_id=0,
        score_fn=_full_vocab_scores,
    )

    first = builder.build(
        object(),
        _training_rollout(),
        objective_valid_mask=torch.tensor(
            [[True, True, True], [True, False, False]]
        ),
        optimizer_step=5,
    )
    repeated = builder.build(
        object(),
        _training_rollout(),
        objective_valid_mask=torch.tensor(
            [[True, True, True], [True, False, False]]
        ),
        optimizer_step=5,
    )

    assert first.parent_state_count == 3
    assert first.successors_per_parent == 4
    assert first.successor_state_count == 12
    assert first.rollout.input_ids.shape[0] == 12
    assert first.rollout.valid_mask.tolist() == [[True]] * 12
    # The sampled successor action is the last attended prompt token; the
    # final token is only a dummy target used to request next-token logits.
    sampled = first.rollout.input_ids[:, -2]
    assert set(sampled.tolist()) <= {40, 41}
    assert torch.equal(first.rollout.input_ids, repeated.rollout.input_ids)
    assert first.state_hash == repeated.state_hash


def test_builder_changes_successor_sample_with_optimizer_step():
    builder = SuccessorStateBuilder(
        _Distributed(),
        parent_state_count=4,
        successors_per_parent=4,
        seed=17,
        sampling_temperature=1.0,
        pad_token_id=0,
        score_fn=_full_vocab_scores,
    )
    mask = torch.tensor([[True, True, True], [True, True, False]])

    step_five = builder.build(
        object(), _training_rollout(), objective_valid_mask=mask, optimizer_step=5
    )
    step_six = builder.build(
        object(), _training_rollout(), objective_valid_mask=mask, optimizer_step=6
    )

    assert step_five.state_hash != step_six.state_hash


def test_builder_runs_deterministic_tiny_model_forward_and_restores_train_mode():
    class TinyCausalLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(64, 8)
            self.output = nn.Linear(8, 64, bias=False)

        def forward(
            self,
            input_ids,
            attention_mask,
            position_ids=None,
            use_cache=False,
            return_dict=True,
        ):
            del attention_mask, position_ids, use_cache, return_dict
            return type(
                "Output", (), {"logits": self.output(self.embedding(input_ids))}
            )()

    torch.manual_seed(3)
    model = TinyCausalLM().train()
    builder = SuccessorStateBuilder(
        _Distributed(),
        parent_state_count=2,
        successors_per_parent=3,
        seed=9,
        sampling_temperature=1.0,
        pad_token_id=0,
        score_micro_batch_size=2,
    )
    valid = torch.tensor([[True, True, True], [True, True, False]])

    first = builder.build(
        model, _training_rollout(), objective_valid_mask=valid, optimizer_step=1
    )
    second = builder.build(
        model, _training_rollout(), objective_valid_mask=valid, optimizer_step=1
    )

    assert model.training is True
    assert first.successor_state_count == 6
    assert first.state_hash == second.state_hash
    assert torch.equal(first.rollout.input_ids, second.rollout.input_ids)
