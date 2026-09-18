import torch

from b200_experiment.opd_core import build_topk_opd_reference
from b200_experiment.selectors.cmt_selector import CMTSelector
from b200_experiment.selectors.pgt_selector import PGTSelector


def _inputs():
    student_ids = torch.tensor([[[1, 2, 3], [4, 5, 6]]])
    teacher_ids = torch.tensor([[[1, 2, 7], [4, 8, 9]]])
    student = torch.log_softmax(
        torch.tensor([[[2.0, 1.0, -1.0], [1.5, 0.0, -0.5]]]), dim=-1
    )
    teacher = torch.log_softmax(
        torch.tensor([[[1.0, 2.0, 0.5], [0.5, 1.0, 2.0]]]), dim=-1
    )
    # Cross scores are global model log-probabilities. The selector uses only
    # teacher_on_student and conditionalizes both models on student Top-K.
    teacher_on_student = torch.tensor(
        [[[-1.0, -0.5, -2.5], [-1.2, -2.0, -0.7]]]
    )
    student_on_teacher = torch.tensor(
        [[[-0.4, -1.1, -2.0], [-0.9, -1.8, -0.3]]]
    )
    valid = torch.tensor([[True, True]])
    return (
        student_ids,
        teacher_ids,
        student,
        teacher_on_student,
        teacher,
        student_on_teacher,
        valid,
    )


def test_pgt_uses_exact_student_topk_support_and_finite_gain():
    output = PGTSelector().compute_scores_from_topk(*_inputs())
    student_ids, teacher_ids, student_logp, teacher_on_student, *_ = _inputs()
    assert output.candidate_ids.shape[-1] == 3
    assert torch.equal(output.candidate_ids, student_ids)
    assert torch.all(output.support_mask)
    teacher_only = set(teacher_ids.flatten().tolist()) - set(student_ids.flatten().tolist())
    assert not teacher_only.intersection(output.candidate_ids.flatten().tolist())
    assert torch.isfinite(output.scores).all()
    assert (output.scores >= 0).all()
    assert torch.equal(output.scores, output.diagnostics["s_PGT"])
    p_mass = (output.student_candidate_log_probs.exp() * output.support_mask).sum(-1)
    q_mass = (output.teacher_candidate_log_probs.exp() * output.support_mask).sum(-1)
    assert torch.allclose(p_mass, torch.ones_like(p_mass), atol=1e-6)
    assert torch.allclose(q_mass, torch.ones_like(q_mass), atol=1e-6)
    expected_teacher = teacher_on_student - torch.logsumexp(
        teacher_on_student, dim=-1, keepdim=True
    )
    assert torch.allclose(output.teacher_candidate_log_probs, expected_teacher)
    assert output.diagnostics["support_definition"] == "student_topk"
    assert "student_union_mass" not in output.diagnostics
    assert "teacher_union_mass" not in output.diagnostics


def test_teacher_only_topk_values_cannot_change_student_support_result():
    args = list(_inputs())
    reference = PGTSelector().compute_scores_from_topk(*args)
    args[1] = torch.full_like(args[1], 99)
    args[4] = torch.randn_like(args[4]) * 100
    args[5] = torch.randn_like(args[5]) * 100
    changed_teacher_only = PGTSelector().compute_scores_from_topk(*args)
    assert torch.equal(changed_teacher_only.candidate_ids, reference.candidate_ids)
    assert torch.allclose(changed_teacher_only.scores, reference.scores)
    assert torch.allclose(
        changed_teacher_only.teacher_candidate_log_probs,
        reference.teacher_candidate_log_probs,
    )


def test_candidate_ids_remain_exact_student_topk_on_padding():
    args = list(_inputs())
    args[-1] = torch.tensor([[True, False]])
    output = PGTSelector().compute_scores_from_topk(*args)
    assert torch.equal(output.candidate_ids, args[0])
    assert torch.all(output.scores[:, 1] == 0)


def test_constant_log_ratio_has_zero_projected_policy_gain():
    args = list(_inputs())
    # Make teacher log-probs equal to student log-probs on student Top-K.
    # The expected policy gradient is then zero even though the raw values are
    # non-zero, which is exactly the distinction from a divergence heuristic.
    args[3] = args[2].clone()
    output = PGTSelector().compute_scores_from_topk(*args)
    assert torch.allclose(output.scores, torch.zeros_like(output.scores), atol=1e-6)


def test_pgt_reference_uses_student_only_support():
    output = PGTSelector().compute_scores_from_topk(*_inputs())
    reference = build_topk_opd_reference(
        output.candidate_ids,
        output.student_candidate_log_probs,
        output.teacher_candidate_log_probs,
        torch.tensor([[True, True]]),
        support_mask=output.support_mask,
    )
    assert torch.isfinite(reference.advantages).all()
    assert torch.allclose(
        reference.student_weights.sum(dim=-1), torch.ones(1, 2), atol=1e-6
    )
    assert torch.equal(reference.candidate_ids, _inputs()[0])


def test_cmt_preserves_student_topk_candidate_support():
    support = PGTSelector().compute_scores_from_topk(*_inputs())
    result = CMTSelector().compute_scores(
        support,
        sampled_token_ids=torch.tensor([[1, 4]]),
        valid_mask=torch.tensor([[True, True]]),
    )
    assert torch.equal(result.candidate_ids, _inputs()[0])
    assert result.candidate_ids.shape[-1] == _inputs()[0].shape[-1]
