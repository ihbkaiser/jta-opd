from __future__ import annotations

import copy
import random
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


def _canonical_name(name: str) -> str:
    value = str(name)
    while value.startswith("module."):
        value = value[len("module.") :]
    return value.replace("_fsdp_wrapped_module.", "")


def clone_to_cpu(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().to(device="cpu", copy=True).contiguous()
    if isinstance(value, dict):
        return {key: clone_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clone_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(clone_to_cpu(item) for item in value)
    return copy.deepcopy(value)


def _named_tensors(values) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for raw_name, tensor in values:
        name = _canonical_name(raw_name)
        if name in result:
            raise ValueError(f"Duplicate canonical training-state name: {name}")
        result[name] = tensor
    return result


def _assert_nested_equal(actual: Any, expected: Any, path: str) -> None:
    if torch.is_tensor(expected):
        if not torch.is_tensor(actual):
            raise AssertionError(f"{path} changed type")
        observed = actual.detach().to(device="cpu")
        if observed.shape != expected.shape or observed.dtype != expected.dtype:
            raise AssertionError(f"{path} changed shape or dtype")
        if not torch.equal(observed, expected):
            raise AssertionError(f"{path} tensor drift")
        return
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping) or set(actual) != set(expected):
            raise AssertionError(f"{path} mapping keys drift")
        for key in expected:
            _assert_nested_equal(actual[key], expected[key], f"{path}.{key}")
        return
    if isinstance(expected, (list, tuple)):
        if not isinstance(actual, type(expected)) or len(actual) != len(expected):
            raise AssertionError(f"{path} sequence drift")
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
            _assert_nested_equal(actual_item, expected_item, f"{path}[{index}]")
        return
    if isinstance(expected, np.ndarray):
        if not isinstance(actual, np.ndarray) or not np.array_equal(actual, expected):
            raise AssertionError(f"{path} array drift")
        return
    if actual != expected:
        raise AssertionError(f"{path} value drift")


@dataclass(frozen=True)
class TrainingStateSnapshot:
    parameters: dict[str, torch.Tensor]
    buffers: dict[str, torch.Tensor]
    optimizer_state: dict[str, Any]
    python_rng_state: object
    numpy_rng_state: tuple[Any, ...]
    torch_rng_state: torch.Tensor
    cuda_rng_state_all: tuple[torch.Tensor, ...]
    extra_state: dict[str, Any]
    model_training: bool

    @classmethod
    def capture(
        cls,
        model,
        optimizer,
        *,
        extra_state: MutableMapping[str, Any] | None = None,
    ) -> "TrainingStateSnapshot":
        if extra_state is not None and not isinstance(extra_state, MutableMapping):
            raise TypeError("extra_state must be a mutable mapping")
        parameters = {
            name: clone_to_cpu(tensor)
            for name, tensor in _named_tensors(model.named_parameters()).items()
        }
        buffers = {
            name: clone_to_cpu(tensor)
            for name, tensor in _named_tensors(model.named_buffers()).items()
        }
        cuda_states = (
            tuple(clone_to_cpu(state) for state in torch.cuda.get_rng_state_all())
            if torch.cuda.is_available()
            else ()
        )
        return cls(
            parameters=parameters,
            buffers=buffers,
            optimizer_state=clone_to_cpu(optimizer.state_dict()),
            python_rng_state=copy.deepcopy(random.getstate()),
            numpy_rng_state=copy.deepcopy(np.random.get_state()),
            torch_rng_state=clone_to_cpu(torch.get_rng_state()),
            cuda_rng_state_all=cuda_states,
            extra_state=clone_to_cpu(dict(extra_state or {})),
            model_training=bool(model.training),
        )

    @torch.no_grad()
    def restore(
        self,
        model,
        optimizer,
        *,
        extra_state: MutableMapping[str, Any] | None = None,
    ) -> None:
        if extra_state is not None and not isinstance(extra_state, MutableMapping):
            raise TypeError("extra_state must be a mutable mapping")
        current_parameters = _named_tensors(model.named_parameters())
        current_buffers = _named_tensors(model.named_buffers())
        if set(current_parameters) != set(self.parameters):
            raise ValueError("Model parameter names changed during the shadow update")
        if set(current_buffers) != set(self.buffers):
            raise ValueError("Model buffer names changed during the shadow update")
        for name, saved in self.parameters.items():
            target = current_parameters[name]
            target.copy_(saved.to(device=target.device, dtype=target.dtype))
        for name, saved in self.buffers.items():
            target = current_buffers[name]
            target.copy_(saved.to(device=target.device, dtype=target.dtype))
        optimizer.load_state_dict(clone_to_cpu(self.optimizer_state))
        optimizer.zero_grad(set_to_none=True)
        model.train(self.model_training)
        if extra_state is not None:
            extra_state.clear()
            extra_state.update(clone_to_cpu(self.extra_state))
        random.setstate(copy.deepcopy(self.python_rng_state))
        np.random.set_state(copy.deepcopy(self.numpy_rng_state))
        torch.set_rng_state(self.torch_rng_state.clone())
        if self.cuda_rng_state_all:
            if not torch.cuda.is_available():
                raise RuntimeError("Snapshot contains CUDA RNG state but CUDA is unavailable")
            torch.cuda.set_rng_state_all(
                [state.clone() for state in self.cuda_rng_state_all]
            )

    @torch.no_grad()
    def verify_restored(
        self,
        model,
        optimizer,
        *,
        extra_state: MutableMapping[str, Any] | None = None,
    ) -> None:
        parameters = _named_tensors(model.named_parameters())
        buffers = _named_tensors(model.named_buffers())
        if set(parameters) != set(self.parameters):
            raise AssertionError("parameter names drift")
        if set(buffers) != set(self.buffers):
            raise AssertionError("buffer names drift")
        for name, expected in self.parameters.items():
            actual = parameters[name].detach().to(device="cpu")
            if actual.shape != expected.shape or actual.dtype != expected.dtype:
                raise AssertionError(f"parameter drift: {name} shape or dtype")
            if not torch.equal(actual, expected):
                raise AssertionError(f"parameter drift: {name}")
        for name, expected in self.buffers.items():
            actual = buffers[name].detach().to(device="cpu")
            if actual.shape != expected.shape or actual.dtype != expected.dtype:
                raise AssertionError(f"buffer drift: {name} shape or dtype")
            if not torch.equal(actual, expected):
                raise AssertionError(f"buffer drift: {name}")
        _assert_nested_equal(
            optimizer.state_dict(), self.optimizer_state, "optimizer state"
        )
        if extra_state is not None:
            _assert_nested_equal(dict(extra_state), self.extra_state, "extra state")
        if bool(model.training) != self.model_training:
            raise AssertionError("model training mode drift")
