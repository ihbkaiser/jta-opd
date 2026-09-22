import json
from types import SimpleNamespace

import pytest
import torch

from b200_experiment.heldout_probe import (
    HeldoutPrefixStore,
    select_heldout_records,
)
from b200_experiment.scoring import RolloutBatch


def _distributed():
    return SimpleNamespace(
        rank=0,
        world_size=1,
        is_main=True,
        barrier=lambda: None,
        all_gather_objects=lambda value: [value],
    )


def _rollout():
    return RolloutBatch(
        input_ids=torch.tensor([[0, 4, 5, 8, 9], [0, 6, 7, 10, 0]]),
        attention_mask=torch.tensor([[0, 1, 1, 1, 1], [0, 1, 1, 1, 0]]),
        response_ids=torch.tensor([[8, 9], [10, 0]]),
        valid_mask=torch.tensor([[True, True], [True, False]]),
        rollout_log_probs=torch.full((2, 2), torch.nan),
        prompt_width=3,
    )


def test_select_heldout_records_is_seed_stable_and_keeps_sample_order():
    records = [{"id": f"sample-{index}", "problem": str(index)} for index in range(20)]

    first = select_heldout_records(records, 6, 123)
    repeated = select_heldout_records(records, 6, 123)
    other = select_heldout_records(records, 6, 124)

    assert [row["id"] for row in first] == [row["id"] for row in repeated]
    assert [row["id"] for row in first] != [row["id"] for row in other]
    selected_indices = [int(row["id"].split("-")[-1]) for row in first]
    assert selected_indices == sorted(selected_indices)


def test_select_heldout_records_rejects_duplicate_ids_and_oversize_subset():
    with pytest.raises(ValueError, match="duplicate benchmark ID"):
        select_heldout_records([{"id": "x"}, {"id": "x"}], 1, 9)
    with pytest.raises(ValueError, match="exceeds benchmark size"):
        select_heldout_records([{"id": "x"}], 2, 9)


def test_heldout_store_round_trips_tensors_and_manifest(tmp_path):
    store = HeldoutPrefixStore(tmp_path, _distributed())
    rollout = _rollout()

    manifest = store.save(
        rollout,
        {
            "benchmark": "Competition-MATH",
            "benchmark_path": "/data/test.parquet",
            "selected_ids": ["a", "b"],
            "generation": {"seed": 7, "temperature": 1.0},
            "tokenizer": "student-tokenizer",
            "student_model": "student",
        },
    )
    loaded = store.load(torch.device("cpu"))

    assert loaded.manifest == manifest
    assert loaded.prefix_hash == manifest["prefix_hash"]
    assert torch.equal(loaded.rollout.input_ids, rollout.input_ids)
    assert torch.equal(loaded.rollout.attention_mask, rollout.attention_mask)
    assert torch.equal(loaded.rollout.response_ids, rollout.response_ids)
    assert torch.equal(loaded.rollout.valid_mask, rollout.valid_mask)
    assert loaded.rollout.prompt_width == rollout.prompt_width
    on_disk = json.loads((tmp_path / "heldout_manifest.json").read_text(encoding="utf-8"))
    assert on_disk == manifest


def test_heldout_store_detects_changed_tensor_bytes(tmp_path):
    store = HeldoutPrefixStore(tmp_path, _distributed())
    store.save(
        _rollout(),
        {
            "benchmark": "Competition-MATH",
            "selected_ids": ["a", "b"],
            "generation": {"seed": 7},
        },
    )
    tensor_path = tmp_path / "heldout_prefixes.rank-00000.pt"
    payload = torch.load(tensor_path, map_location="cpu", weights_only=True)
    payload["response_ids"][0, 0] += 1
    torch.save(payload, tensor_path)

    with pytest.raises(ValueError, match="held-out prefix hash mismatch"):
        store.load(torch.device("cpu"))


def test_heldout_store_rejects_manifest_settings_mismatch(tmp_path):
    store = HeldoutPrefixStore(tmp_path, _distributed())
    store.save(
        _rollout(),
        {
            "benchmark": "Competition-MATH",
            "selected_ids": ["a", "b"],
            "generation": {"seed": 7},
        },
    )

    with pytest.raises(ValueError, match="held-out manifest does not match"):
        store.load(
            torch.device("cpu"),
            expected={"benchmark": "Competition-MATH", "generation": {"seed": 8}},
        )
