from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import torch
import pytest

try:
    import fcntl  # noqa: F401
except ModuleNotFoundError:
    sys.modules["fcntl"] = types.SimpleNamespace(
        LOCK_EX=1,
        LOCK_UN=2,
        flock=lambda *_args, **_kwargs: None,
    )

from b200_experiment import trainer
from b200_experiment.one_step_kl_probe import OneStepKLProbeConfig
from b200_experiment.probe_runtime import (
    OneStepKLProbeRuntime,
    prepare_one_step_kl_probe,
)


class _Distributed:
    rank = 0
    world_size = 1
    is_main = True

    @staticmethod
    def all_gather_objects(value):
        return [value]

    @staticmethod
    def sum_int(value):
        return int(value)

    @staticmethod
    def sum_float(value):
        return float(value)


class _Probe:
    failure_policy = "error"

    def __init__(self):
        self.written = []

    @staticmethod
    def should_probe(step):
        return True

    def prepare_before_step(self, model, step, **kwargs):
        self.prepared_kwargs = kwargs
        return {"step": step, "before": float(next(model.parameters()).item())}

    @staticmethod
    def score_student(model, _prepared):
        return float(next(model.parameters()).item())

    def write_pair(self, prepared, **values):
        self.written.append((prepared, values))


def _result(kwargs, loss):
    valid = kwargs["objective_valid_mask"].bool()
    uses_gibbs = kwargs["gibbs_scores"] is not None
    metric = {
        "loss": loss,
        "weighted_final_loss": loss,
        "training_forward_time": 0.1,
        "backward_time": 0.2,
        "optimizer_time": 0.3,
        "global_weight_mass": float(valid.sum().item()),
    }
    if uses_gibbs:
        weights = valid.float() * 2.0
        groups = torch.where(valid, torch.zeros_like(valid, dtype=torch.long), -1)
        steps = torch.where(
            valid,
            torch.full_like(
                valid, kwargs["optimizer_step_start"] + 1, dtype=torch.long
            ),
            -1,
        )
        processed = valid.clone()
    else:
        weights = groups = steps = processed = None
    return {
        **metric,
        "optimizer_steps": 1,
        "gibbs_allocations": int(uses_gibbs),
        "gibbs_allocation_time": 0.0,
        "allocated_position_weights": weights,
        "allocated_raw_position_weights": weights,
        "allocated_group_indices": groups,
        "allocated_optimizer_steps": steps,
        "allocated_processed_mask": processed,
        "minibatches": [metric],
    }


def test_paired_probe_runs_uniform_shadow_then_restored_real_cmt(monkeypatch):
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(4.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    calls = []

    def fake_impl(**kwargs):
        before = float(next(kwargs["model"].parameters()).item())
        calls.append(
            {
                "before": before,
                "gibbs": kwargs["gibbs_scores"] is not None,
                "weights": kwargs["position_weights"].clone(),
                "callback": kwargs["on_optimizer_step"],
                "offset": kwargs["ppo_minibatch_offset"],
            }
        )
        with torch.no_grad():
            next(kwargs["model"].parameters()).add_(1.0)
        return _result(kwargs, 0.2 if kwargs["gibbs_scores"] is None else 0.1)

    monkeypatch.setattr(trainer, "_opd_train_step_impl", fake_impl)
    rollout = SimpleNamespace(
        input_ids=torch.ones(1, 2, dtype=torch.long),
        valid_mask=torch.tensor([[True]]),
    )
    callback_steps = []
    probe = _Probe()
    result = trainer._opd_train_step(
        model,
        optimizer,
        rollout,
        torch.tensor([[0.75]]),
        object(),
        {"training": {"ppo_mini_batch_size": 1}},
        torch.device("cpu"),
        _Distributed(),
        objective_valid_mask=torch.tensor([[True]]),
        trajectory_active_mask=torch.tensor([True]),
        trajectory_group_ids=[0],
        gibbs_scores=torch.tensor([[0.4]]),
        gibbs_epsilon=0.5,
        optimizer_step_start=10,
        on_optimizer_step=lambda step, metric: callback_steps.append((step, metric)),
        one_step_probe=probe,
    )

    assert [call["gibbs"] for call in calls] == [False, True]
    assert calls[0]["weights"].tolist() == [[1.0]]
    assert calls[0]["before"] == calls[1]["before"] == 4.0
    assert all(call["callback"] is None for call in calls)
    assert float(model.weight.item()) == 5.0
    assert callback_steps[0][0] == 11
    assert len(callback_steps) == 1
    assert len(probe.written) == 1
    assert probe.prepared_kwargs["rollout"] is rollout
    assert probe.prepared_kwargs["rollout_id"] == 0
    assert probe.prepared_kwargs["ppo_group_index"] == 0
    assert probe.prepared_kwargs["state_age_steps"] == 0
    assert probe.written[0][1]["uniform_update_loss"] == 0.2
    assert probe.written[0][1]["cmt_update_loss"] == 0.1
    assert result["optimizer_steps"] == 1


def test_no_probe_keeps_single_existing_update_call(monkeypatch):
    calls = []

    def fake_impl(**kwargs):
        calls.append(kwargs)
        return {"sentinel": True}

    monkeypatch.setattr(trainer, "_opd_train_step_impl", fake_impl)
    result = trainer._opd_train_step(
        object(),
        object(),
        object(),
        object(),
        object(),
        {"training": {}},
        torch.device("cpu"),
        _Distributed(),
    )

    assert result == {"sentinel": True}
    assert len(calls) == 1


def test_runtime_writes_complete_paired_record():
    class Builder:
        @staticmethod
        def build(student, rollout, *, objective_valid_mask, optimizer_step):
            assert student == "pre"
            assert rollout == "training-rollout"
            assert objective_valid_mask.tolist() == [[True]]
            assert optimizer_step == 4
            return SimpleNamespace(
                rollout="successor-rollout",
                parent_state_count=64,
                successors_per_parent=4,
                successor_state_count=256,
                state_hash="successor-hash",
                sampling_seed=123,
            )

    class Evaluator:
        @staticmethod
        def build_reference(student, teacher, rollout, *, prefix_hash):
            assert student == "pre"
            assert teacher == "teacher"
            assert rollout == "successor-rollout"
            return SimpleNamespace(
                global_valid_token_count=2,
                prefix_hash=prefix_hash,
                pre_kl_values=torch.tensor([0.9, 1.1]),
            ), 1.0

        @staticmethod
        def score_student_evaluation(student, reference):
            assert reference.prefix_hash == "successor-hash"
            if student == "uniform":
                return SimpleNamespace(
                    mean=0.8,
                    local_values=torch.tensor([0.7, 0.9]),
                    global_count=2,
                )
            assert student == "cmt"
            return SimpleNamespace(
                mean=0.7,
                local_values=torch.tensor([0.5, 0.9]),
                global_count=2,
            )

    class Logger:
        def __init__(self):
            self.rows = []

        def upsert(self, row):
            self.rows.append(row)

    logger = Logger()
    runtime = OneStepKLProbeRuntime(
        OneStepKLProbeConfig(enabled=True, top_k=16),
        evaluator=Evaluator(),
        builder=Builder(),
        logger=logger,
        teacher="teacher",
        run_id="run-1",
        distributed=_Distributed(),
    )

    prepared = runtime.prepare_before_step(
        "pre",
        4,
        rollout="training-rollout",
        objective_valid_mask=torch.tensor([[True]]),
        rollout_id=1,
        ppo_group_index=2,
        state_age_steps=2,
    )
    uniform = runtime.score_student("uniform", prepared)
    cmt = runtime.score_student("cmt", prepared)
    runtime.write_pair(
        prepared,
        optimizer_step=4,
        rollout_id=1,
        ppo_group_index=2,
        kl_after_uniform=uniform,
        kl_after_cmt=cmt,
        uniform_update_loss=0.5,
        cmt_update_loss=0.4,
        uniform_branch_time_sec=1.2,
        cmt_eval_time_sec=0.3,
    )

    row = logger.rows[0]
    assert row["delta_uniform"] == pytest.approx(0.2)
    assert row["delta_cmt"] == pytest.approx(0.3)
    assert row["paired_gap"] == pytest.approx(0.1)
    assert row["successor_state_count"] == 256
    assert row["valid_successor_state_count"] == 2
    assert row["state_age_steps"] == 2
    assert row["successor_state_hash"] == "successor-hash"
    assert row["delta_cmt_standard_error"] == pytest.approx(0.1)


class _SetupDistributed(_Distributed):
    is_main = True

    @staticmethod
    def barrier():
        return None


class _Tokenizer:
    name_or_path = "student-tokenizer"
    padding_side = "right"
    pad_token_id = 0
    eos_token_id = 9

    @staticmethod
    def apply_chat_template(messages, **_kwargs):
        return messages[0]["content"]

    def __call__(self, prompts, **_kwargs):
        width = max(len(prompt.split()) for prompt in prompts)
        ids = torch.zeros(len(prompts), width, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for row, prompt in enumerate(prompts):
            length = len(prompt.split())
            ids[row, width - length :] = torch.arange(1, length + 1)
            mask[row, width - length :] = 1
        return {"input_ids": ids, "attention_mask": mask}


def _probe_config(enabled=True):
    return {
        "experiment": {"method": "cmt", "run_id": "fixed-run"},
        "models": {"student_path": "student", "teacher_path": "teacher"},
        "data": {"chat_template_kwargs": {}},
        "selector": {"score_chunk_steps": 32},
        "one_step_kl_probe": {
            "enabled": enabled,
            "state_source": "training_rollout_successors",
            "parent_state_count": 64,
            "successors_per_parent": 4,
            "seed": 11,
            "interval_steps": 1,
            "top_k": 16,
            "sampling_temperature": 1.0,
            "score_micro_batch_size": 1,
        },
    }


def test_probe_setup_uses_training_successors_without_benchmark_or_generation():
    runtime, metadata = prepare_one_step_kl_probe(
        _probe_config(),
        method="cmt",
        teacher=torch.nn.Linear(1, 1),
        tokenizer=_Tokenizer(),
        output_dir="unused",
        distributed=_SetupDistributed(),
    )

    assert runtime is not None
    assert metadata["state_source"] == "training_rollout_successors"
    assert metadata["parent_state_count"] == 64
    assert metadata["successors_per_parent"] == 4


def test_disabled_probe_does_not_create_runtime():
    runtime, metadata = prepare_one_step_kl_probe(
        _probe_config(enabled=False),
        method="cmt",
        teacher=torch.nn.Linear(1, 1),
        tokenizer=_Tokenizer(),
        output_dir="unused",
        distributed=_SetupDistributed(),
    )
    assert runtime is None
    assert metadata is None
