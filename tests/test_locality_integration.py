from pathlib import Path

import pytest
import torch
from torch import nn

from b200_experiment.config import load_config, validate_locality_probe_config
from b200_experiment.distributed import DistributedContext
from b200_experiment.opd_core import topk_reference_from_logits
from b200_experiment.scoring import RolloutBatch
from b200_experiment.tensorboard_logging import locality_tensorboard_metrics
from b200_experiment.trainer import (
    _StepSideEffectGate,
    _opd_train_step,
    locality_uniform_position_weights,
)

ROOT = Path(__file__).resolve().parents[1]


def _enabled_config():
    config = load_config(ROOT / "configs" / "qwen3_b200_cmt.yaml")
    config["rollout"]["num_responses"] = 1
    config["training"]["ppo_mini_batch_size"] = config["rollout"]["batch_size"]
    config["analysis"] = {
        "locality_probe": {
            "enabled": True,
            "training_allocation": "uniform",
            "g_bins": 10,
            "future_horizons": [8, 16, 32],
            "future_gamma": 1.0,
            "main_future_horizon": 32,
            "conditioning_quantiles": [0.4, 0.6],
            "token_sample_size": 2048,
            "matched_pairs_per_step": 8,
            "other_id": {
                "enabled": True,
                "benchmark_name": "Competition-MATH",
                "num_prompts": 32,
                "max_new_tokens": 2048,
                "seed": 20260921,
                "fixed_support": True,
            },
        }
    }
    return config


def test_locality_validation_accepts_one_fresh_rollout_per_step():
    resolved = validate_locality_probe_config(_enabled_config())
    assert resolved["enabled"] is True
    assert resolved["future_horizons"] == [8, 16, 32]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda c: c["experiment"].update(method="opd"), "requires experiment.method=cmt"),
        (lambda c: c["training"].update(ppo_mini_batch_size=16), "one fresh rollout"),
        (lambda c: c["rollout"].update(num_responses=2), "num_responses=1"),
        (lambda c: c["analysis"]["locality_probe"].update(training_allocation="gibbs"), "only supports uniform"),
        (lambda c: c["analysis"]["locality_probe"].update(future_horizons=[16, 8]), "strictly increasing"),
        (lambda c: c["analysis"]["locality_probe"].update(conditioning_quantiles=[0.7, 0.2]), "conditioning_quantiles"),
        (lambda c: c["analysis"]["locality_probe"]["other_id"].update(fixed_support=False), "fixed_support=true"),
    ],
)
def test_locality_validation_rejects_invalid_protocols(mutation, message):
    config = _enabled_config()
    mutation(config)
    with pytest.raises(ValueError, match=message):
        validate_locality_probe_config(config)


def test_locality_uniform_weights_are_one_only_on_valid_states():
    config = _enabled_config()
    valid = torch.tensor([[True, False], [True, True]])

    weights = locality_uniform_position_weights(config, valid)

    torch.testing.assert_close(weights, torch.tensor([[1.0, 0.0], [1.0, 1.0]]))


def test_disabled_locality_returns_no_training_override():
    config = _enabled_config()
    config["analysis"]["locality_probe"]["enabled"] = False
    valid = torch.ones((2, 2), dtype=torch.bool)

    assert locality_uniform_position_weights(config, valid) is None


def test_locality_step_side_effects_wait_until_analysis_is_durable():
    events = []
    gate = _StepSideEffectGate(defer=True)

    gate.submit(7, {"loss": 1.0}, lambda step, _metrics: events.append(step))

    assert events == []
    assert gate.pending_steps == (7,)

    gate.flush(lambda step, _metrics: events.append(step), expected_count=1)

    assert events == [7]
    assert gate.pending_steps == ()


def test_locality_tensorboard_metrics_include_deciles_other_id_future_and_timing():
    row = {
        "self": {
            "spearman_g_realized_gain": 0.4,
            "pearson_g_realized_gain": 0.3,
            "decile_mean_slope": 0.2,
            "adjacent_monotonic_fraction": 0.8,
            "deciles": [
                {"decile": index, "count": 2, "realized_gain_mean": index / 100}
                for index in range(1, 11)
            ],
        },
        "other_id": {"reverse_kl_after": 0.1, "realized_gain": 0.02},
        "future": {
            "conditioning_band": [0.4, 0.6],
            "horizon_32_std": 0.6,
            "horizon_32_iqr": 0.5,
            "horizon_32_p90_minus_p10": 0.9,
        },
        "timing": {"train_post_rescore_sec": 2.0, "other_id_probe_sec": 1.0},
    }

    metrics = locality_tensorboard_metrics(row)

    assert metrics["analysis/self/decile_01_realized_gain"] == 0.01
    assert metrics["analysis/self/decile_10_realized_gain"] == 0.1
    assert metrics["analysis/other_id/reverse_kl"] == 0.1
    assert metrics["analysis/future/q40_q60_h32_iqr"] == 0.5


def test_locality_optimizer_path_records_exact_uniform_cmt_weights():
    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(9, 4)
            self.output = nn.Linear(4, 9, bias=False)
            self.config = type("Config", (), {"use_cache": False, "model_type": "tiny"})()

        def forward(self, input_ids, **_kwargs):
            return type("Output", (), {"logits": self.output(self.embedding(input_ids))})()

    torch.manual_seed(3)
    model = TinyModel()
    input_ids = torch.tensor([[1, 2, 3, 4], [2, 3, 4, 5]])
    rollout = RolloutBatch(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        response_ids=input_ids[:, 2:],
        valid_mask=torch.ones((2, 2), dtype=torch.bool),
        rollout_log_probs=torch.zeros((2, 2)),
        prompt_width=2,
    )
    with torch.no_grad():
        logits = model(input_ids).logits[:, 1:3]
        reference = topk_reference_from_logits(
            logits, logits + torch.linspace(-0.2, 0.2, 9), rollout.valid_mask, top_k=4
        )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    context = DistributedContext(0, 0, 1, torch.device("cpu"))
    config = {
        "rollout": {"temperature": 1.0},
        "selector": {"score_chunk_steps": 4},
        "training": {
            "micro_batch_size_per_gpu": 2,
            "ppo_mini_batch_size": 2,
            "ppo_clip_low": 0.2,
            "ppo_clip_high": 0.28,
            "max_grad_norm": 1.0,
        },
    }

    result = _opd_train_step(
        model,
        optimizer,
        rollout,
        rollout.valid_mask.float(),
        reference,
        config,
        torch.device("cpu"),
        context,
        gibbs_scores=torch.tensor([[0.1, 10.0], [100.0, 1000.0]]),
        gibbs_epsilon=0.5,
        force_uniform_allocation=True,
    )

    torch.testing.assert_close(
        result["allocated_position_weights"], torch.ones((2, 2))
    )
    assert result["minibatches"][0]["allocation_solver_status"] == "uniform_locality_diagnostic"
