from __future__ import annotations

import pytest
import torch

from b200_experiment.diagnostics import tensor_summary


@pytest.mark.parametrize(
    "dtype",
    (torch.float16, torch.bfloat16, torch.float32, torch.float64),
)
def test_tensor_summary_quantile_dtype_matches_promoted_input(dtype):
    values = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=dtype)

    summary = tensor_summary(values)

    assert summary["q50"] == pytest.approx(2.5)
    assert summary["q99"] == pytest.approx(3.97)
    assert summary["mean"] == pytest.approx(2.5)


def test_tensor_summary_masks_values_before_fp64_quantiles():
    values = torch.tensor([[1.0, 100.0], [3.0, 200.0]], dtype=torch.float32)
    valid = torch.tensor([[True, False], [True, False]])

    summary = tensor_summary(values, valid)

    assert summary["min"] == pytest.approx(1.0)
    assert summary["max"] == pytest.approx(3.0)
    assert summary["q50"] == pytest.approx(2.0)
