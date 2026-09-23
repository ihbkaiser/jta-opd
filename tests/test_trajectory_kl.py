from types import SimpleNamespace

import pytest
import torch

from b200_experiment.trajectory_kl import (
    FullVocabularyTrajectoryKLEvaluator,
    full_vocab_reverse_kl_values,
)


class _Distributed:
    @staticmethod
    def sum_float(value):
        return float(value)

    @staticmethod
    def sum_int(value):
        return int(value)


def test_full_vocab_reverse_kl_matches_analytic_distribution():
    student = torch.log(torch.tensor([[[0.75, 0.25]]]))
    teacher = torch.log(torch.tensor([[[0.50, 0.50]]]))
    values = full_vocab_reverse_kl_values(student, teacher)
    expected = 0.75 * torch.log(torch.tensor(1.5)) + 0.25 * torch.log(
        torch.tensor(0.5)
    )
    assert values.dtype == torch.float64
    assert values.item() == pytest.approx(expected.item())


def test_trajectory_evaluator_uses_fixed_horizon_and_root_equal_weighting():
    # KL values by rollout: [1+1, 2+2+2+2, 3, 4+4+4+4].  Divide each by H=4,
    # then average adjacent pairs because there are two rollouts per root.
    kl = torch.tensor(
        [[1.0, 1.0, 0.0, 0.0], [2.0, 2.0, 2.0, 2.0],
         [3.0, 0.0, 0.0, 0.0], [4.0, 4.0, 4.0, 4.0]]
    )
    valid = torch.tensor(
        [[True, True, False, False], [True, True, True, True],
         [True, False, False, False], [True, True, True, True]]
    )

    def score_fn(*_args, **kwargs):
        assert kwargs["compute_full_vocab_metrics"] is True
        return SimpleNamespace(full_log_ratio_mean=-kl), SimpleNamespace()

    rollout = SimpleNamespace(valid_mask=valid)
    evaluator = FullVocabularyTrajectoryKLEvaluator(
        _Distributed(), horizon=4, num_rollouts_per_problem=2, score_fn=score_fn
    )
    student = torch.nn.Linear(1, 1)
    teacher = torch.nn.Linear(1, 1)
    student.train()
    teacher.train()

    result = evaluator.evaluate(
        student, teacher, rollout, problem_ids=("a", "b")
    )

    assert result.problem_ids == ("a", "b")
    assert result.per_problem_kl.tolist() == pytest.approx([1.25, 2.375])
    assert result.mean == pytest.approx(1.8125)
    assert result.valid_state_count == 11
    assert result.mean_length == pytest.approx(2.75)
    assert result.early_eos_rate == pytest.approx(0.5)
    assert student.training is True
    assert teacher.training is True


def test_trajectory_evaluator_rejects_wrong_rollout_population():
    def score_fn(*_args, **_kwargs):
        return SimpleNamespace(full_log_ratio_mean=torch.zeros(3, 2)), SimpleNamespace()

    evaluator = FullVocabularyTrajectoryKLEvaluator(
        _Distributed(), horizon=2, num_rollouts_per_problem=2, score_fn=score_fn
    )
    with pytest.raises(ValueError, match="rollout population"):
        evaluator.evaluate(
            torch.nn.Linear(1, 1),
            torch.nn.Linear(1, 1),
            SimpleNamespace(valid_mask=torch.ones(3, 2, dtype=torch.bool)),
            problem_ids=("a", "b"),
        )
