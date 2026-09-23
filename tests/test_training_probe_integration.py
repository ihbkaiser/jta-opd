from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest
import torch

try:
    import fcntl  # noqa: F401
except ModuleNotFoundError:
    sys.modules["fcntl"] = types.SimpleNamespace(
        LOCK_EX=1, LOCK_UN=2, flock=lambda *_args, **_kwargs: None
    )

from b200_experiment import trainer
from b200_experiment.one_step_kl_probe import OneStepKLProbeConfig
from b200_experiment.probe_runtime import (
    OneStepKLProbeRuntime,
    prepare_one_step_kl_probe,
)
from b200_experiment.trajectory_kl import TrajectoryKLEvaluation


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

    @staticmethod
    def barrier():
        return None


class _Probe:
    failure_policy = "error"

    def __init__(self):
        self.written = []

    @staticmethod
    def should_probe(_step):
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
            torch.full_like(valid, kwargs["optimizer_step_start"] + 1, dtype=torch.long),
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
        calls.append(
            {
                "before": float(next(kwargs["model"].parameters()).item()),
                "gibbs": kwargs["gibbs_scores"] is not None,
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
    assert calls[0]["before"] == calls[1]["before"] == 4.0
    assert float(model.weight.item()) == 5.0
    assert callback_steps[0][0] == 11
    assert len(probe.written) == 1
    assert probe.written[0][1]["uniform_update_loss"] == 0.2
    assert probe.written[0][1]["cmt_update_loss"] == 0.1
    assert result["optimizer_steps"] == 1


def test_no_probe_keeps_single_existing_update_call(monkeypatch):
    calls = []

    def fake_impl(**kwargs):
        calls.append(kwargs)
        return {"sentinel": True}

    monkeypatch.setattr(trainer, "_opd_train_step_impl", fake_impl)
    assert trainer._opd_train_step(
        object(), object(), object(), object(), object(),
        {"training": {}}, torch.device("cpu"), _Distributed()
    ) == {"sentinel": True}
    assert len(calls) == 1


def _evaluation(mean, values, lengths, early):
    return TrajectoryKLEvaluation(
        mean=mean,
        problem_ids=("p0", "p1"),
        per_problem_kl=torch.tensor(values, dtype=torch.float64),
        per_problem_mean_length=torch.tensor(lengths, dtype=torch.float64),
        per_problem_early_eos_rate=torch.tensor(early, dtype=torch.float64),
        valid_state_count=120,
        trajectory_count=4,
        mean_length=sum(lengths) / 2,
        early_eos_rate=sum(early) / 2,
    )


def test_runtime_logs_branch_specific_trajectory_gap():
    class Builder:
        def build(self, student, *, optimizer_step):
            return SimpleNamespace(
                rollout=f"rollout-{student}",
                problem_ids=("p0", "p1"),
                sampling_seed=optimizer_step + 100,
                trajectory_hash=f"hash-{student}",
            )

    class Evaluator:
        def evaluate(self, student, teacher, rollout, *, problem_ids):
            assert teacher == "teacher"
            assert rollout == f"rollout-{student}"
            assert problem_ids == ("p0", "p1")
            return (
                _evaluation(0.9, [1.0, 0.8], [64, 60], [0, 0.5])
                if student == "uniform"
                else _evaluation(0.7, [0.6, 0.8], [64, 64], [0, 0])
            )

    class Logger:
        def __init__(self):
            self.calls = []

        def upsert(self, row, details):
            self.calls.append((row, details))

    logger = Logger()
    runtime = OneStepKLProbeRuntime(
        OneStepKLProbeConfig(enabled=True, subset_size=2),
        evaluator=Evaluator(),
        builder=Builder(),
        logger=logger,
        teacher="teacher",
        run_id="run-1",
        distributed=_Distributed(),
        benchmark="Competition-MATH",
        heldout_example_hash="examples",
        heldout_root_hash="roots",
    )
    prepared = runtime.prepare_before_step(
        "pre", 50, rollout_id=1, ppo_group_index=2
    )
    uniform = runtime.score_student("uniform", prepared)
    cmt = runtime.score_student("cmt", prepared)
    runtime.write_pair(
        prepared,
        optimizer_step=50,
        rollout_id=1,
        ppo_group_index=2,
        kl_after_uniform=uniform,
        kl_after_cmt=cmt,
        uniform_update_loss=0.5,
        cmt_update_loss=0.4,
        uniform_branch_time_sec=1.2,
        cmt_eval_time_sec=0.3,
    )

    row, details = logger.calls[0]
    assert row["trajectory_gap"] == pytest.approx(0.2)
    assert row["uniform_trajectory_kl"] == pytest.approx(0.9)
    assert row["cmt_trajectory_kl"] == pytest.approx(0.7)
    assert [item["trajectory_gap"] for item in details] == pytest.approx([0.4, 0.0])


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
        "evaluation": {
            "benchmarks": {"Competition-MATH": {"path": "test.parquet"}}
        },
        "one_step_kl_probe": {
            "enabled": enabled,
            "benchmark": "Competition-MATH",
            "subset_size": 2,
            "num_rollouts_per_problem": 2,
            "horizon": 64,
            "seed": 11,
            "interval_steps": 50,
            "metric": "full_vocab_reverse_kl",
            "sampling_temperature": 1.0,
            "sampling_top_p": 1.0,
            "generation_batch_size": 1,
            "score_micro_batch_size": 1,
        },
    }


def test_probe_setup_persists_fixed_competition_math_roots(tmp_path):
    records = [
        {"id": "a", "problem": "one two"},
        {"id": "b", "problem": "three four five"},
    ]
    runtime, metadata = prepare_one_step_kl_probe(
        _probe_config(),
        method="cmt",
        student=torch.nn.Linear(1, 1),
        teacher=torch.nn.Linear(1, 1),
        tokenizer=_Tokenizer(),
        rollout_engine=None,
        output_dir=tmp_path,
        distributed=_Distributed(),
        device=torch.device("cpu"),
        resume=False,
        benchmark_loader=lambda *_args: (records, {"file": "test.parquet"}),
    )

    assert runtime is not None
    assert metadata["benchmark"] == "Competition-MATH"
    assert metadata["subset_size"] == 2
    assert metadata["metric"] == "full_vocab_reverse_kl"
    assert (tmp_path / "one_step_trajectory_kl_probe" / "heldout_root_manifest.json").is_file()


def test_disabled_probe_does_not_create_runtime(tmp_path):
    runtime, metadata = prepare_one_step_kl_probe(
        _probe_config(enabled=False),
        method="cmt",
        student=torch.nn.Linear(1, 1),
        teacher=torch.nn.Linear(1, 1),
        tokenizer=_Tokenizer(),
        rollout_engine=None,
        output_dir=tmp_path,
        distributed=_Distributed(),
        device=torch.device("cpu"),
        resume=False,
    )
    assert runtime is None
    assert metadata is None
