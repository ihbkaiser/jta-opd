import random

import numpy as np
import pytest
import torch

from b200_experiment.training_snapshot import TrainingStateSnapshot


def _model_and_optimizer():
    model = torch.nn.Sequential(
        torch.nn.Linear(3, 5),
        torch.nn.GELU(),
        torch.nn.Linear(5, 2),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    return model, optimizer


def _step(model, optimizer):
    optimizer.zero_grad(set_to_none=True)
    inputs = torch.tensor([[1.0, -2.0, 0.5], [-1.0, 0.25, 2.0]])
    loss = model(inputs).square().mean()
    loss.backward()
    optimizer.step()
    return float(loss.detach())


def _optimizer_tensor_state(optimizer):
    state = optimizer.state_dict()["state"]
    return {
        parameter_id: {
            key: value.detach().cpu().clone() if torch.is_tensor(value) else value
            for key, value in values.items()
        }
        for parameter_id, values in state.items()
    }


def test_snapshot_restores_model_optimizer_rng_and_extra_state_exactly():
    random.seed(11)
    np.random.seed(11)
    torch.manual_seed(11)
    model, optimizer = _model_and_optimizer()
    _step(model, optimizer)
    counter = {"optimizer_step": 7, "callback_count": 2}
    parameters_before = {
        name: value.detach().clone() for name, value in model.named_parameters()
    }
    optimizer_before = _optimizer_tensor_state(optimizer)
    snapshot = TrainingStateSnapshot.capture(model, optimizer, extra_state=counter)

    expected_random = random.random()
    expected_numpy = float(np.random.random())
    expected_torch = torch.rand(4)
    _step(model, optimizer)
    counter.update(optimizer_step=8, callback_count=3)
    random.random()
    np.random.random()
    torch.rand(4)

    snapshot.restore(model, optimizer, extra_state=counter)
    snapshot.verify_restored(model, optimizer, extra_state=counter)

    assert counter == {"optimizer_step": 7, "callback_count": 2}
    for name, parameter in model.named_parameters():
        assert torch.equal(parameter, parameters_before[name])
    restored_optimizer = _optimizer_tensor_state(optimizer)
    assert restored_optimizer.keys() == optimizer_before.keys()
    for parameter_id in optimizer_before:
        for key, expected in optimizer_before[parameter_id].items():
            actual = restored_optimizer[parameter_id][key]
            if torch.is_tensor(expected):
                assert torch.equal(actual, expected)
            else:
                assert actual == expected
    assert random.random() == expected_random
    assert float(np.random.random()) == expected_numpy
    assert torch.equal(torch.rand(4), expected_torch)


def test_snapshot_verification_detects_parameter_drift():
    torch.manual_seed(5)
    model, optimizer = _model_and_optimizer()
    _step(model, optimizer)
    snapshot = TrainingStateSnapshot.capture(model, optimizer)
    snapshot.restore(model, optimizer)
    with torch.no_grad():
        next(model.parameters()).view(-1)[0].add_(1.0)

    with pytest.raises(AssertionError, match="parameter drift"):
        snapshot.verify_restored(model, optimizer)


def test_snapshot_capture_rejects_non_mapping_extra_state():
    model, optimizer = _model_and_optimizer()
    with pytest.raises(TypeError, match="extra_state must be a mutable mapping"):
        TrainingStateSnapshot.capture(model, optimizer, extra_state=[1, 2])
