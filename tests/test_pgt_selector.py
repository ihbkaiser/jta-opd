import torch

from b200_experiment.opd_core import build_topk_opd_reference
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
    # Cross scores are global model log-probabilities on the compact supports;
    # the selector conditionalizes both models on the literal union.
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


def test_pgt_builds_literal_union_and_finite_gain():
    output = PGTSelector().compute_scores_from_topk(*_inputs())
    assert output.candidate_ids.shape[-1] == 6
    assert output.support_mask[0, 0].sum().item() == 4
    assert output.support_mask[0, 1].sum().item() == 5
    assert torch.isfinite(output.scores).all()
    assert (output.scores >= 0).all()
    assert torch.equal(output.scores, output.diagnostics["s_PGT"])
    p_mass = (output.student_candidate_log_probs.exp() * output.support_mask).sum(-1)
    q_mass = (output.teacher_candidate_log_probs.exp() * output.support_mask).sum(-1)
    assert torch.allclose(p_mass, torch.ones_like(p_mass), atol=1e-6)
    assert torch.allclose(q_mass, torch.ones_like(q_mass), atol=1e-6)


def test_constant_log_ratio_has_zero_projected_policy_gain():
    args = list(_inputs())
    # Make teacher log-probs equal to student log-probs on every union action.
    # The expected policy gradient is then zero even though the raw values are
    # non-zero, which is exactly the distinction from a divergence heuristic.
    args[1] = args[0].clone()
    args[3] = args[2].clone()
    args[4] = args[2].clone()
    args[5] = args[2].clone()
    output = PGTSelector().compute_scores_from_topk(*args)
    assert torch.allclose(output.scores, torch.zeros_like(output.scores), atol=1e-6)


def test_pgt_reference_masks_duplicate_padding():
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
    assert torch.all(reference.advantages[~output.support_mask] == 0)
