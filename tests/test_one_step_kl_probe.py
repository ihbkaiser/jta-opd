import math

import pytest
import torch

from b200_experiment.one_step_kl_probe import (
    OneStepKLProbeConfig,
    conditional_reverse_kl,
    paired_improvement,
)


def test_probe_config_rejects_enabled_non_cmt_method():
    with pytest.raises(ValueError, match="only supported for method=cmt"):
        OneStepKLProbeConfig.from_mapping(
            {"enabled": True, "subset_size": 64, "interval_steps": 1},
            method="opd",
        )


def test_probe_config_resolves_requested_production_defaults():
    resolved = OneStepKLProbeConfig.from_mapping(
        {
            "enabled": True,
            "benchmark": "Competition-MATH",
            "subset_size": 64,
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
    assert resolved.subset_size == 64
    assert resolved.seed == 20260922
    assert resolved.top_k == 16
    assert resolved.should_probe(1)
    assert resolved.should_probe(177)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"subset_size": 0}, "subset_size must be positive"),
        ({"interval_steps": 0}, "interval_steps must be positive"),
        ({"top_k": 8}, "top_k=16"),
        ({"failure_policy": "ignore"}, "failure_policy"),
        ({"metric": "forward_kl"}, "metric"),
        ({"temperature": -1.0}, "temperature"),
        ({"top_p": 0.0}, "top_p"),
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
