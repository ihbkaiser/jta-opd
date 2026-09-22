from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from b200_experiment.distributed import DistributedContext
from b200_experiment.heldout_kl import FiniteSupportKLEvaluator
from b200_experiment.scoring import BaseScores


def _scores(*, top_k_ids=None, top_k_log_probs=None, candidate_log_probs=None):
    shape = (
        top_k_ids.shape[:2]
        if top_k_ids is not None
        else candidate_log_probs.shape[:2]
    )
    zeros = torch.zeros(shape)
    return BaseScores(
        response_logits=None,
        log_normalizers=zeros,
        scaled_log_normalizers=zeros,
        sampled_log_probs=zeros,
        entropies=zeros,
        top_k_ids=top_k_ids,
        top_k_log_probs=top_k_log_probs,
        candidate_log_probs=candidate_log_probs,
    )


def test_evaluator_freezes_preupdate_student_support_and_reuses_it_after_update():
    student = torch.nn.Linear(1, 1)
    teacher = torch.nn.Linear(1, 1)
    student.phase = "before"
    candidate_ids = torch.tensor([[[2, 5], [1, 3]]])
    student_before = torch.log(torch.tensor([[[0.8, 0.2], [0.6, 0.4]]]))
    teacher_values = torch.log(torch.tensor([[[0.5, 0.5], [0.5, 0.5]]]))
    student_after = torch.log(torch.tensor([[[0.6, 0.4], [0.5, 0.5]]]))
    calls = []

    def score_fn(model, rollout, **kwargs):
        model.eval()
        calls.append((model, kwargs.get("top_k"), kwargs.get("candidate_ids")))
        if model is teacher:
            assert torch.equal(kwargs["candidate_ids"], candidate_ids)
            return _scores(candidate_log_probs=teacher_values)
        if kwargs.get("top_k") == 2:
            return _scores(
                top_k_ids=candidate_ids, top_k_log_probs=student_before
            )
        assert torch.equal(kwargs["candidate_ids"], candidate_ids)
        return _scores(candidate_log_probs=student_after)

    rollout = SimpleNamespace(valid_mask=torch.tensor([[True, True]]))
    evaluator = FiniteSupportKLEvaluator(
        DistributedContext(0, 0, 1, torch.device("cpu")),
        top_k=2,
        score_fn=score_fn,
    )
    student.train()
    reference, before = evaluator.build_reference(
        student, teacher, rollout, prefix_hash="fixed-prefix-hash"
    )
    evaluation = evaluator.score_student_evaluation(student, reference)
    after = evaluation.mean

    assert reference.prefix_hash == "fixed-prefix-hash"
    assert torch.equal(reference.candidate_ids, candidate_ids)
    assert reference.pre_kl_values.shape == (2,)
    assert evaluation.local_values.shape == (2,)
    assert before > after
    assert student.training is True
    assert calls[-1][1] == 0
    assert torch.equal(calls[-1][2], candidate_ids)


def test_evaluator_uses_global_sum_and_count_for_token_weighted_mean():
    class FakeDistributed:
        def __init__(self):
            self.summed = []

        def sum_float(self, value):
            self.summed.append(("float", value))
            return value + 3.0

        def sum_int(self, value):
            self.summed.append(("int", value))
            return value + 2

    values = torch.log(torch.tensor([[[0.75, 0.25]]]))
    candidates = torch.tensor([[[0, 1]]])

    def score_fn(model, rollout, **kwargs):
        if kwargs.get("top_k") == 2:
            return _scores(top_k_ids=candidates, top_k_log_probs=values)
        return _scores(candidate_log_probs=values)

    distributed = FakeDistributed()
    evaluator = FiniteSupportKLEvaluator(
        distributed, top_k=2, score_fn=score_fn
    )
    model = torch.nn.Linear(1, 1)
    rollout = SimpleNamespace(valid_mask=torch.tensor([[True]]))
    reference, value = evaluator.build_reference(
        model, model, rollout, prefix_hash="hash"
    )

    # Local KL is zero, then the fake second rank contributes total=3,count=2.
    assert value == pytest.approx(1.0)
    assert reference.global_valid_token_count == 3
    assert distributed.summed[-2][0] == "float"
    assert distributed.summed[-1] == ("int", 1)


def test_evaluator_rejects_missing_topk_outputs():
    def score_fn(model, rollout, **kwargs):
        return _scores(candidate_log_probs=torch.zeros(1, 1, 2))

    evaluator = FiniteSupportKLEvaluator(
        DistributedContext(0, 0, 1, torch.device("cpu")),
        top_k=2,
        score_fn=score_fn,
    )
    with pytest.raises(RuntimeError, match="Top-K"):
        evaluator.build_reference(
            torch.nn.Linear(1, 1),
            torch.nn.Linear(1, 1),
            SimpleNamespace(valid_mask=torch.tensor([[True]])),
            prefix_hash="hash",
        )
