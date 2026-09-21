import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from b200_experiment.locality_logging import LocalityLogger
from b200_experiment.locality_probe import (
    FrozenStateProbe,
    LocalityProbeRunner,
    audit_prompt_overlap,
    deterministic_prompt_sample,
    load_competition_math_test,
)


def _probe() -> FrozenStateProbe:
    return FrozenStateProbe(
        input_ids=torch.tensor([[11, 12, 3, 4], [21, 22, 5, 0]]),
        attention_mask=torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]]),
        response_ids=torch.tensor([[3, 4], [5, 0]]),
        valid_mask=torch.tensor([[True, True], [True, False]]),
        prompt_width=2,
        candidate_ids=torch.tensor(
            [
                [[0, 1, 2], [1, 2, 3]],
                [[0, 2, 4], [0, 0, 0]],
            ]
        ),
        support_mask=torch.tensor(
            [
                [[True, True, False], [True, True, True]],
                [[True, False, True], [False, False, False]],
            ]
        ),
        teacher_log_probs=torch.tensor(
            [
                [[-0.2, -1.7, 0.0], [-1.0, -1.1, -1.2]],
                [[-1.4, 0.0, -0.3], [0.0, 0.0, 0.0]],
            ]
        ),
        metadata=[
            {"sample_id": "a", "response_position": 0},
            {"sample_id": "a", "response_position": 1},
            {"sample_id": "b", "response_position": 0},
        ],
    )


def test_competition_math_loader_and_deterministic_prompt_sampling(tmp_path: Path):
    path = tmp_path / "test.jsonl"
    path.write_text(
        "\n".join(
            json.dumps({"unique_id": f"id-{index}", "problem": f"p{index}", "answer": str(index)})
            for index in range(10)
        ),
        encoding="utf-8",
    )
    config = {
        "evaluation": {
            "benchmarks": {
                "Competition-MATH": {
                    "path": str(path),
                    "question_key": "problem",
                    "answer_key": "answer",
                    "id_key": "unique_id",
                }
            }
        }
    }

    records, schema = load_competition_math_test(config)
    first = deterministic_prompt_sample(records, 4, seed=17)
    second = deterministic_prompt_sample(records, 4, seed=17)

    assert schema["benchmark"] == "Competition-MATH"
    assert [row["id"] for row in first] == [row["id"] for row in second]
    assert len(first) == 4


def test_probe_cache_round_trip_preserves_every_tensor_and_metadata(tmp_path: Path):
    probe = _probe()
    cache = tmp_path / "probe_states.pt"

    probe.save(cache)
    restored = FrozenStateProbe.load(cache)

    for name in FrozenStateProbe.tensor_fields():
        torch.testing.assert_close(getattr(restored, name), getattr(probe, name))
    assert restored.prompt_width == probe.prompt_width
    assert restored.metadata == probe.metadata


def test_probe_capture_freezes_candidate_support_from_pre_update_scores():
    source = _probe()
    rollout = source.as_rollout(torch.device("cpu"))
    scored = SimpleNamespace(
        candidate_ids=source.candidate_ids.clone(),
        support_mask=source.support_mask.clone(),
        teacher_candidate_log_probs=source.teacher_log_probs.clone(),
    )

    captured = FrozenStateProbe.from_scored_rollout(
        rollout, scored, metadata=source.metadata
    )
    scored.candidate_ids.fill_(99)
    scored.teacher_candidate_log_probs.zero_()

    torch.testing.assert_close(captured.candidate_ids, source.candidate_ids)
    torch.testing.assert_close(captured.teacher_log_probs, source.teacher_log_probs)


def test_student_only_probe_scoring_uses_cached_teacher_and_is_microbatch_invariant():
    probe = _probe()
    teacher_before = probe.teacher_log_probs.clone()
    called_models = []

    def fake_score(model, rollout, **kwargs):
        called_models.append(model)
        candidate_ids = kwargs["candidate_ids"]
        logits = model.logits.to(candidate_ids.device)
        values = logits[candidate_ids]
        log_probs = values - torch.logsumexp(logits, dim=0)
        return SimpleNamespace(candidate_log_probs=log_probs)

    student = SimpleNamespace(logits=torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0]))
    runner = LocalityProbeRunner(score_function=fake_score)

    whole = runner.score_frozen_states(student, probe, torch.device("cpu"), micro_batch_size=2)
    split = runner.score_frozen_states(student, probe, torch.device("cpu"), micro_batch_size=1)

    torch.testing.assert_close(whole, split)
    torch.testing.assert_close(probe.teacher_log_probs, teacher_before)
    assert all(model is student for model in called_models)
    assert not whole.requires_grad


def test_step_post_metric_becomes_next_step_pre_metric():
    runner = LocalityProbeRunner()
    runner.initialize_other_id_metric(0.8)

    first = runner.record_other_id_metric(0.5)
    second = runner.record_other_id_metric(0.45)

    assert first == {"reverse_kl_before": 0.8, "reverse_kl_after": 0.5, "realized_gain": 0.3}
    assert second == {"reverse_kl_before": 0.5, "reverse_kl_after": 0.45, "realized_gain": 0.05}


def test_prompt_overlap_audit_normalizes_whitespace_and_case():
    report = audit_prompt_overlap(
        [{"problem": "  Find X "}, {"problem": "different"}],
        [{"problem": "find   x"}, {"problem": "held out"}],
        train_prompt_key="problem",
        test_prompt_key="problem",
    )

    assert report["overlap_count"] == 1
    assert report["overlaps"][0]["normalized_prompt"] == "find x"


def test_locality_logger_upserts_steps_and_truncates_future_rows(tmp_path: Path):
    logger = LocalityLogger(tmp_path, resume_step=0)
    logger.write_metrics({"step": 1, "value": "old"})
    logger.write_metrics({"step": 2, "value": "future"})
    logger.write_metrics({"step": 1, "value": "new"})

    rows = [json.loads(line) for line in logger.metrics_path.read_text().splitlines()]
    assert rows == [{"step": 1, "value": "new"}, {"step": 2, "value": "future"}]

    resumed = LocalityLogger(tmp_path, resume_step=1)
    rows = [json.loads(line) for line in resumed.metrics_path.read_text().splitlines()]
    assert rows == [{"step": 1, "value": "new"}]


def test_locality_logger_serializes_degenerate_statistics_as_null(tmp_path: Path):
    logger = LocalityLogger(tmp_path, resume_step=0)

    logger.write_metrics({"step": 1, "self": {"spearman": float("nan")}})

    row = json.loads(logger.metrics_path.read_text(encoding="utf-8"))
    assert row["self"]["spearman"] is None


def test_locality_logger_stratifies_samples_and_writes_gzip_pairs(tmp_path: Path):
    logger = LocalityLogger(tmp_path, resume_step=0)
    records = [
        {"flat_index": index, "g_decile": index // 3, "g": float(index)}
        for index in range(30)
    ]
    selected = logger.write_token_samples(3, records, sample_size=10)
    pair_path = logger.write_matched_pairs(3, [{"pair": 1}])

    assert len(selected) == 10
    assert {row["g_decile"] for row in selected} == set(range(10))
    with gzip.open(pair_path, "rt", encoding="utf-8") as handle:
        assert [json.loads(line) for line in handle] == [{"pair": 1}]
