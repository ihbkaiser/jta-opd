import math

import torch

from b200_experiment.locality_analysis import (
    assign_global_quantile_bins,
    build_matched_pairs,
    conditional_future_summary,
    correlation_summary,
    decile_summary,
    finite_horizon_successor_gain,
    realized_self_gain,
    reverse_kl_on_fixed_support,
)


def test_reverse_kl_matches_hand_calculation_on_masked_fixed_support():
    student = torch.log(torch.tensor([[[0.6, 0.3, 0.1]]]))
    teacher = torch.log(torch.tensor([[[0.2, 0.2, 0.6]]]))
    support = torch.tensor([[[True, True, False]]])

    actual = reverse_kl_on_fixed_support(student, teacher, support)

    p0, p1 = 2.0 / 3.0, 1.0 / 3.0
    expected = p0 * math.log(p0 / 0.5) + p1 * math.log(p1 / 0.5)
    torch.testing.assert_close(actual, torch.tensor([[expected]]))


def test_realized_self_gain_has_pre_minus_post_sign_and_no_grad():
    pre = torch.tensor([[0.8, 0.2]], requires_grad=True)
    post = torch.tensor([[0.5, 0.3]], requires_grad=True)

    gain = realized_self_gain(pre, post)

    torch.testing.assert_close(gain, torch.tensor([[0.3, -0.1]]))
    assert not gain.requires_grad


def test_quantile_bins_keep_ties_together_and_cover_requested_range():
    values = torch.tensor([0.0, 0.0, 1.0, 2.0, 2.0, 3.0])

    assignments, edges = assign_global_quantile_bins(values, bins=3)

    assert assignments.tolist() == [0, 0, 1, 1, 1, 2]
    assert edges.shape == (4,)
    assert assignments.min().item() == 0
    assert assignments.max().item() == 2


def test_successor_gain_respects_padding_terminal_and_horizon_boundaries():
    gains = torch.tensor([[1.0, 2.0, 3.0, 99.0], [4.0, 5.0, 99.0, 99.0]])
    valid = torch.tensor([[True, True, True, False], [True, True, False, False]])

    result = finite_horizon_successor_gain(gains, valid, [1, 2], gamma=1.0)

    torch.testing.assert_close(result[1], torch.tensor([[2.0, 3.0, 0.0, 0.0], [5.0, 0.0, 0.0, 0.0]]))
    torch.testing.assert_close(result[2], torch.tensor([[5.0, 3.0, 0.0, 0.0], [5.0, 0.0, 0.0, 0.0]]))


def test_gamma_one_successor_scan_matches_reference_loop():
    torch.manual_seed(7)
    gains = torch.randn(3, 8)
    valid = torch.tensor(
        [
            [True] * 8,
            [True] * 5 + [False] * 3,
            [True] * 2 + [False] * 6,
        ]
    )

    actual = finite_horizon_successor_gain(gains, valid, [1, 3, 20], gamma=1.0)
    for horizon, values in actual.items():
        expected = torch.zeros_like(gains)
        for row in range(gains.shape[0]):
            length = int(valid[row].sum())
            for position in range(length):
                expected[row, position] = gains[
                    row, position + 1 : min(length, position + horizon + 1)
                ].sum()
        torch.testing.assert_close(values, expected)
        assert not values.requires_grad


def test_correlation_and_deciles_summarize_observed_gain():
    local_gain = torch.arange(1.0, 11.0)
    realized = 2.0 * local_gain
    assignments = torch.arange(10)
    pre = torch.ones(10)
    post = pre - realized

    correlations = correlation_summary(local_gain, realized)
    deciles = decile_summary(local_gain, pre, post, realized, assignments, bins=10)

    assert correlations["pearson_g_realized_gain"] == 1.0
    assert correlations["spearman_g_realized_gain"] == 1.0
    assert len(deciles["deciles"]) == 10
    assert deciles["adjacent_monotonic_fraction"] == 1.0
    assert deciles["decile_mean_slope"] == 2.0
    assert deciles["q10_to_q1_gain_difference"] == 18.0


def test_conditional_future_summary_uses_requested_band_and_reports_spread():
    local_gain = torch.arange(10.0)
    future = torch.arange(10.0) - 4.0

    summary = conditional_future_summary(
        local_gain,
        {32: future},
        conditioning_quantiles=(0.4, 0.6),
        main_horizon=32,
        minimum_count=1,
    )

    assert summary["conditioning_band"] == [0.4, 0.6]
    assert summary["count"] == 2
    assert summary["horizon_32_mean"] == 0.5
    assert summary["horizon_32_fraction_positive"] == 0.5
    assert summary["horizon_32_fraction_negative"] == 0.0


def test_matched_pairs_are_deterministic_and_hold_candidate_ids_fixed():
    local_gain = torch.tensor([0.40, 0.401, 0.405, 0.406, 0.50, 0.501])
    future = torch.tensor([-5.0, 8.0, 1.0, -2.0, 7.0, -9.0])
    candidate_ids = torch.arange(18).reshape(6, 3)
    metadata = [{"sample_id": f"s{index}"} for index in range(6)]

    first = build_matched_pairs(
        local_gain,
        future,
        metadata,
        candidate_ids=candidate_ids,
        max_pairs=2,
        percentile_bin_width=50.0,
        normalized_g_tolerance=0.02,
    )
    second = build_matched_pairs(
        local_gain,
        future,
        metadata,
        candidate_ids=candidate_ids,
        max_pairs=2,
        percentile_bin_width=50.0,
        normalized_g_tolerance=0.02,
    )

    assert first == second
    assert first
    for pair in first:
        assert pair["low_future"]["candidate_ids"] == candidate_ids[
            pair["low_future"]["flat_index"]
        ].tolist()
        assert pair["high_future"]["candidate_ids"] == candidate_ids[
            pair["high_future"]["flat_index"]
        ].tolist()
