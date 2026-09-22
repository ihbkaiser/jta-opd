import math

import pytest
import torch

from b200_experiment.one_step_kl_probe import (
    OneStepKLProbeConfig,
    conditional_reverse_kl,
    paired_sample_statistics,
    paired_improvement,
)


def test_probe_config_rejects_enabled_non_cmt_method():
    with pytest.raises(ValueError, match="only supported for method=cmt"):
        OneStepKLProbeConfig.from_mapping(
            {"enabled": True, "parent_state_count": 64, "interval_steps": 1},
            method="opd",
        )


def test_probe_config_resolves_requested_production_defaults():
    resolved = OneStepKLProbeConfig.from_mapping(
        {
            "enabled": True,
            "state_source": "training_rollout_successors",
            "parent_state_count": 64,
            "successors_per_parent": 4,
            "seed": 20260922,
            "interval_steps": 1,
            "top_k": 16,
            "metric": "conditional_reverse_kl",
            "failure_policy": "error",
            "artifact_subdir": "one_step_kl_probe",
        },
        method="cmt",
    )
    assert resolved.enabled is True
    assert resolved.parent_state_count == 64
    assert resolved.successors_per_parent == 4
    assert resolved.state_source == "training_rollout_successors"
    assert resolved.seed == 20260922
    assert resolved.top_k == 16
    assert resolved.should_probe(1)
    assert resolved.should_probe(177)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"parent_state_count": 0}, "parent_state_count must be positive"),
        ({"successors_per_parent": 0}, "successors_per_parent must be positive"),
        ({"state_source": "heldout"}, "state_source"),
        ({"interval_steps": 0}, "interval_steps must be positive"),
        ({"top_k": 8}, "top_k=16"),
        ({"failure_policy": "ignore"}, "failure_policy"),
        ({"metric": "forward_kl"}, "metric"),
        ({"sampling_temperature": -1.0}, "sampling_temperature"),
    ],
)
def test_probe_config_rejects_invalid_values(overrides, message):
    settings = {"enabled": True, **overrides}
    with pytest.raises(ValueError, match=message):
        OneStepKLProbeConfig.from_mapping(settings, method="cmt")


def test_conditional_reverse_kl_matches_manual_distribution():
    student = torch.log(torch.tensor([[[0.75, 0.25]]], dtype=torch.float64))
    teacher = torch.log(torch.tensor([[[0.50, 0.50]]], dtype=torch.float64))
    valid = torch.tensor([[True]])

    total, count = conditional_reverse_kl(student, teacher, valid)

    expected = 0.75 * math.log(1.5) + 0.25 * math.log(0.5)
    assert total.item() == pytest.approx(expected)
    assert count == 1


def test_conditional_reverse_kl_renormalizes_logits_and_masks_padding():
    student = torch.tensor([[[2.0, 1.0], [100.0, -100.0]]])
    teacher = torch.tensor([[[2.0, 1.0], [-100.0, 100.0]]])
    valid = torch.tensor([[True, False]])

    total, count = conditional_reverse_kl(student, teacher, valid)

    assert total.item() == pytest.approx(0.0, abs=1e-7)
    assert count == 1


def test_conditional_reverse_kl_rejects_empty_valid_population():
    values = torch.zeros(1, 1, 2)
    with pytest.raises(ValueError, match="no valid response tokens"):
        conditional_reverse_kl(values, values, torch.tensor([[False]]))


def test_paired_improvement_uses_before_minus_after():
    row = paired_improvement(0.7, 0.5, 0.4)

    assert row == pytest.approx(
        {
            "delta_uniform": 0.2,
            "delta_cmt": 0.3,
            "paired_gap": 0.1,
        }
    )


def test_paired_sample_statistics_use_successor_level_standard_error():
    class Distributed:
        @staticmethod
        def sum_float(value):
            return float(value)

        @staticmethod
        def sum_int(value):
            return int(value)

    result = paired_sample_statistics(
        torch.tensor([1.0, 2.0, 3.0, 4.0]),
        torch.tensor([0.0, 1.0, 2.0, 3.0]),
        torch.tensor([0.0, 0.0, 2.0, 2.0]),
        Distributed(),
    )

    assert result["sample_count"] == 4
    assert result["delta_uniform"] == pytest.approx(1.0)
    assert result["delta_uniform_standard_error"] == pytest.approx(0.0)
    assert result["delta_cmt"] == pytest.approx(1.5)
    assert result["delta_cmt_standard_error"] == pytest.approx(
        math.sqrt(1.0 / 12.0)
    )
    assert result["paired_gap"] == pytest.approx(0.5)
    assert result["paired_gap_standard_error"] == pytest.approx(
        math.sqrt(1.0 / 12.0)
    )
