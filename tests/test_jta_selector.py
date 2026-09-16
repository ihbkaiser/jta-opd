from __future__ import annotations

import inspect

import torch

import b200_experiment.selectors.jta_selector as jta_module
from b200_experiment.opd_core import TopKOPDReference, topk_candidate_ppo_loss
from b200_experiment.selectors.jta_selector import (
    CountSketchOperator,
    JTASelector,
    TensorSketchOperator,
    kl_ball_linear_oracle,
)


def _reference(
    advantages: torch.Tensor,
    candidate_ids: torch.Tensor | None = None,
) -> TopKOPDReference:
    if candidate_ids is None:
        candidate_ids = torch.arange(advantages.shape[-1]).reshape(1, 1, -1)
        candidate_ids = candidate_ids.expand_as(advantages).clone()
    valid = torch.ones(advantages.shape[:2], dtype=torch.bool)
    return TopKOPDReference(
        candidate_ids=candidate_ids,
        old_student_log_probs=torch.zeros_like(advantages),
        teacher_log_probs=torch.zeros_like(advantages),
        student_weights=torch.ones_like(advantages),
        advantages=advantages,
        support_mask=torch.ones_like(advantages, dtype=torch.bool),
    )


def test_jta_delta_matches_topk_ppo_autograd_derivative_at_on_policy_point():
    advantages = torch.tensor([[[0.25, -0.5, 1.25]]])
    old = torch.zeros_like(advantages)
    current = old.clone().requires_grad_(True)
    reference = TopKOPDReference(
        candidate_ids=torch.arange(3).reshape(1, 1, 3),
        old_student_log_probs=old,
        teacher_log_probs=old,
        student_weights=torch.ones_like(old),
        advantages=advantages,
    )
    topk_candidate_ppo_loss(
        current,
        reference,
        clip_low=0.2,
        clip_high=0.28,
        dual_clip=None,
    ).sum().backward()
    assert torch.allclose(current.grad, -advantages)


def test_countsketch_is_order_invariant_and_chunked_reduce_matches_dense_reference():
    ids = torch.tensor([[1, 7, 7, 19], [3, 11, 23, 31]], dtype=torch.long)
    coefficients = torch.tensor([[1.0, -2.0, 0.5, 3.0], [2.0, 1.5, -1.0, 0.25]])
    operator = CountSketchOperator(sketch_dim=8, sketch_seed=42, token_chunk_size=1)
    actual = operator.reduce(coefficients, ids)

    expected = torch.zeros(2, 8)
    buckets, signs = operator.hash(ids)
    expected.scatter_add_(1, buckets, coefficients * signs)
    assert torch.equal(actual, expected)

    permutation = torch.tensor([3, 0, 2, 1])
    reordered = operator.reduce(coefficients[:, permutation], ids[:, permutation])
    assert torch.equal(actual, reordered)


def test_kl_oracle_handles_zero_constant_and_large_epsilon_cases():
    scores = torch.tensor([0.0, 1.0, 3.0, -2.0])
    prior = torch.full((4,), 0.25)

    zero = kl_ball_linear_oracle(scores, prior, epsilon=0.0)
    assert torch.allclose(zero, prior, atol=1e-7)

    constant = kl_ball_linear_oracle(torch.ones(4), prior, epsilon=0.5)
    assert torch.allclose(constant, prior, atol=1e-7)

    large = kl_ball_linear_oracle(scores, prior, epsilon=100.0)
    assert torch.equal(large, torch.tensor([0.0, 0.0, 1.0, 0.0]))

    assert torch.all(large >= 0)
    assert torch.isclose(large.sum(), torch.tensor(1.0))
    kl = (large * (large.clamp_min(1e-12) / prior).log()).sum()
    assert float(kl) <= 100.0 + 1e-6


def test_kl_oracle_accepts_padded_prompt_batch_and_matches_rowwise_oracle():
    scores = torch.tensor(
        [
            [0.0, 1.0, 3.0, -2.0],
            [0.5, -1.0, 0.25, 0.0],
        ]
    )
    mask = torch.tensor(
        [
            [True, True, True, True],
            [True, True, False, False],
        ]
    )
    prior = mask.float()
    prior = prior / prior.sum(dim=1, keepdim=True)
    batched = kl_ball_linear_oracle(
        scores,
        prior,
        epsilon=0.5,
        mask=mask,
        iterations=30,
    )
    expected = torch.stack(
        [
            kl_ball_linear_oracle(scores[0], prior[0], epsilon=0.5, iterations=30),
            torch.cat(
                [
                    kl_ball_linear_oracle(
                        scores[1, :2], prior[1, :2], epsilon=0.5, iterations=30
                    ),
                    torch.zeros(2),
                ]
            ),
        ]
    )
    assert torch.allclose(batched, expected, atol=2e-5, rtol=2e-5)


def test_batched_solver_has_no_host_scalarization_in_iteration_kernel():
    source = inspect.getsource(jta_module._solve_prompts_batched)
    assert ".item(" not in source


def test_jta_epsilon_zero_reproduces_uniform_opd_multipliers():
    advantages = torch.tensor(
        [
            [[-1.0, 0.5, 0.0], [-0.25, 0.75, 0.0]],
            [[0.25, -0.5, 0.0], [1.0, -1.0, 0.5]],
        ]
    )
    valid = torch.tensor([[True, True], [True, False]])
    reference = _reference(advantages)
    output = JTASelector(
        epsilon=0.0, sketch_dim=16, embedding_backend="topk_logprob_countsketch"
    ).allocate(
        reference,
        valid,
        torch.tensor([True, True]),
        num_responses=1,
    )
    assert torch.allclose(output.weights, valid.float(), atol=1e-6)
    assert torch.allclose(output.probabilities[valid], torch.tensor([0.5, 0.5, 1.0]))
    assert torch.all(output.weights[~valid] == 0)


def test_jta_leave_one_prompt_out_excludes_all_rollouts_of_current_prompt():
    # Two prompts, two rollouts, one token each. Prompt 0's reference must be
    # prompt 1 only, and vice versa. IDs make the expected sketch means exact.
    advantages = torch.tensor(
        [
            [[-1.0]],
            [[-2.0]],
            [[-3.0]],
            [[-4.0]],
        ]
    )
    ids = torch.tensor([[[10]], [[11]], [[20]], [[21]]])
    valid = torch.ones(4, 1, dtype=torch.bool)
    output = JTASelector(
        epsilon=0.5,
        sketch_dim=32,
        sketch_seed=3,
        embedding_backend="topk_logprob_countsketch",
        fw_max_iterations=8,
    ).allocate(
        _reference(advantages, ids),
        valid,
        torch.tensor([True, True]),
        num_responses=2,
    )
    assert output.diagnostics["reference_scope"] == "active_batch_leave_one_prompt_out"
    assert output.diagnostics["candidate_scope"] == "all_rollouts_of_prompt"
    reference = output.diagnostics["reference"]
    operator = CountSketchOperator(sketch_dim=32, sketch_seed=3)
    prompt_0_expected = operator.reduce(-advantages[2:4], ids[2:4]).sum(dim=0) / 2
    prompt_1_expected = operator.reduce(-advantages[0:2], ids[0:2]).sum(dim=0) / 2
    assert torch.allclose(reference[0], prompt_0_expected)
    assert torch.allclose(reference[1], prompt_1_expected)
    assert not torch.allclose(reference[0], reference[1])
    assert output.diagnostics["prompt_token_counts"].tolist() == [2, 2]


def test_jta_r2_prompt_local_budget_and_fallback_for_single_active_prompt():
    advantages = torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]]])
    valid = torch.ones(2, 2, dtype=torch.bool)
    result = JTASelector(
        epsilon=0.5, sketch_dim=8, embedding_backend="topk_logprob_countsketch"
    ).allocate(
        _reference(advantages),
        valid,
        torch.tensor([True]),
        num_responses=2,
    )
    assert result.diagnostics["reference_fallback"] is True
    assert torch.allclose(result.probabilities[valid], torch.full((4,), 0.25))
    assert torch.allclose(result.weights[valid], torch.ones(4))


def test_jta_r4_variable_lengths_preserves_each_prompt_budget():
    responses = 4
    prompt_lengths = [1, 3, 2]
    batch = len(prompt_lengths) * responses
    topk = 3
    advantages = torch.randn(batch, max(prompt_lengths), topk)
    ids = torch.randint(0, 1000, (batch, max(prompt_lengths), topk))
    valid = torch.zeros(batch, max(prompt_lengths), dtype=torch.bool)
    for prompt, length in enumerate(prompt_lengths):
        valid[prompt * responses : (prompt + 1) * responses, :length] = True
    reference = _reference(advantages, ids)
    output = JTASelector(
        epsilon=0.2,
        sketch_dim=16,
        embedding_backend="topk_logprob_countsketch",
        fw_max_iterations=4,
        kl_bisection_iterations=10,
    ).allocate(
        reference,
        valid,
        torch.ones(len(prompt_lengths), dtype=torch.bool),
        responses,
    )
    grouped = output.weights.reshape(len(prompt_lengths), responses, -1)
    for prompt, length in enumerate(prompt_lengths):
        assert torch.isclose(
            grouped[prompt, :, :length].sum(),
            torch.tensor(float(responses * length)),
            atol=1e-5,
        )
        assert torch.all(grouped[prompt, :, length:] == 0)


def test_jta_solver_preserves_prompt_feasibility_and_reports_monotone_objective():
    torch.manual_seed(4)
    advantages = torch.randn(3, 4, 2)
    valid = torch.tensor([[True, True, True, False], [True, True, False, False], [True, True, True, True]])
    result = JTASelector(
        epsilon=0.2,
        sketch_dim=16,
        embedding_backend="topk_logprob_countsketch",
        fw_max_iterations=30,
        fw_gap_tolerance=1e-6,
    ).allocate(
        _reference(advantages),
        valid,
        torch.tensor([True, True, True]),
        num_responses=1,
    )
    probabilities = result.probabilities
    weights = result.weights
    for row in range(3):
        p = probabilities[row][valid[row]]
        assert torch.all(p >= 0)
        assert torch.isclose(p.sum(), torch.tensor(1.0), atol=1e-5)
        prior = torch.full_like(p, 1.0 / p.numel())
        kl = (p * (p.clamp_min(1e-12) / prior).log()).sum()
        assert float(kl) <= 0.2 + 1e-5
        assert torch.isclose(weights[row][valid[row]].sum(), torch.tensor(float(p.numel())), atol=1e-4)
    assert torch.all(weights[~valid] == 0)
    objective_history = result.diagnostics["objective_history"]
    for history in objective_history:
        assert all(b + 1e-6 >= a for a, b in zip(history, history[1:]))


def test_tensorsketch_uses_independent_deterministic_vocab_and_hidden_hashes():
    first = TensorSketchOperator(
        sketch_dim=32,
        vocab_hash_seed=17,
        hidden_hash_seed=991,
        token_chunk_size=1,
    )
    second = TensorSketchOperator(
        sketch_dim=32,
        vocab_hash_seed=17,
        hidden_hash_seed=991,
        token_chunk_size=7,
    )
    ids = torch.tensor([[3, 3, 11, 29]])
    coefficients = torch.tensor([[1.0, -0.25, 0.5, 2.0]])
    hidden = torch.randn(1, 5)
    assert torch.equal(first.vocab.hash(ids)[0], second.vocab.hash(ids)[0])
    assert torch.equal(first.vocab.hash(ids)[1], second.vocab.hash(ids)[1])
    assert torch.equal(
        first.hidden.hash(torch.arange(hidden.shape[-1]))[0],
        second.hidden.hash(torch.arange(hidden.shape[-1]))[0],
    )
    assert torch.equal(
        first.hidden.hash(torch.arange(hidden.shape[-1]))[1],
        second.hidden.hash(torch.arange(hidden.shape[-1]))[1],
    )
    assert torch.equal(
        first.reduce(coefficients, ids, hidden),
        second.reduce(coefficients, ids, hidden),
    )


def test_tensorsketch_masks_invalid_states_and_accumulates_duplicate_candidates():
    operator = TensorSketchOperator(
        sketch_dim=64, vocab_hash_seed=3, hidden_hash_seed=5
    )
    ids = torch.tensor([[7, 7, 13, 0]])
    coefficients = torch.tensor([[2.0, -0.5, 1.25, 100.0]])
    hidden = torch.tensor([[1.0, -2.0, 0.5]])
    mask = torch.tensor([[True, True, True, False]])
    actual = operator.reduce(coefficients, ids, hidden, mask=mask)
    coalesced = torch.tensor([[1.5, 1.25]])
    coalesced_ids = torch.tensor([[7, 13]])
    expected = operator.reduce(coalesced, coalesced_ids, hidden)
    assert torch.allclose(actual, expected, atol=1e-6)
    assert torch.equal(
        operator.coefficient_norm(coefficients, ids, mask=mask),
        torch.tensor([torch.sqrt(torch.tensor(1.5**2 + 1.25**2))]),
    )


def test_fft_tensorsketch_matches_explicit_small_outer_product_sketch():
    operator = TensorSketchOperator(
        sketch_dim=32, vocab_hash_seed=23, hidden_hash_seed=71
    )
    ids = torch.tensor([[2, 2, 9]])
    coefficients = torch.tensor([[1.25, -0.25, 0.75]])
    hidden = torch.tensor([[0.5, -1.0, 2.0, 0.25]])
    actual = operator.reduce(coefficients, ids, hidden)
    vocab_buckets, vocab_signs = operator.vocab.hash(ids)
    hidden_buckets, hidden_signs = operator.hidden.hash(
        torch.arange(hidden.shape[-1])
    )
    explicit = torch.zeros(1, operator.sketch_dim)
    for candidate in range(ids.shape[-1]):
        for dimension in range(hidden.shape[-1]):
            bucket = int(
                (vocab_buckets[0, candidate] + hidden_buckets[dimension])
                % operator.sketch_dim
            )
            explicit[0, bucket] += (
                coefficients[0, candidate]
                * vocab_signs[0, candidate]
                * hidden[0, dimension]
                * hidden_signs[dimension]
            )
    assert torch.allclose(actual, explicit, atol=1e-5, rtol=1e-5)


def test_tensorsketch_approximately_preserves_contextual_inner_products():
    torch.manual_seed(29)
    operator = TensorSketchOperator(
        sketch_dim=4096, vocab_hash_seed=101, hidden_hash_seed=211
    )
    ids = torch.randint(0, 1000, (32, 12))
    coefficients = torch.randn(32, 12)
    hidden = torch.randn(32, 24)
    embedding = operator.reduce(coefficients, ids, hidden)
    exact_vocab = torch.zeros(32, 1000)
    exact_vocab.scatter_add_(1, ids, coefficients)
    expected = (exact_vocab @ exact_vocab.T) * (hidden @ hidden.T)
    actual = embedding @ embedding.T
    expected_flat, actual_flat = expected.flatten(), actual.flatten()
    correlation = torch.corrcoef(torch.stack((expected_flat, actual_flat)))[0, 1]
    assert float(correlation) > 0.9


def test_jta_epsilon_zero_matches_vanilla_opd_loss_and_gradient():
    advantages = torch.tensor([[[0.5, -1.0], [1.25, 0.25]]])
    reference = _reference(advantages)
    valid = torch.ones(1, 2, dtype=torch.bool)
    hidden = torch.ones(1, 2, 4)
    output = JTASelector(
        epsilon=0.0,
        sketch_dim=32,
        embedding_backend="topk_context_tensorsketch",
    ).allocate(
        reference,
        valid,
        torch.tensor([True]),
        num_responses=1,
        hidden_states=hidden,
    )
    vanilla_logits = torch.zeros_like(advantages).requires_grad_(True)
    jta_logits = vanilla_logits.detach().clone().requires_grad_(True)
    vanilla_per_position = topk_candidate_ppo_loss(
        vanilla_logits, reference, clip_low=0.2, clip_high=0.2
    )
    vanilla_loss = vanilla_per_position.sum() / valid.sum()
    jta_loss = (
        topk_candidate_ppo_loss(
            jta_logits, reference, clip_low=0.2, clip_high=0.2
        )
        * output.weights
    ).sum() / output.weights.sum()
    vanilla_loss.backward()
    jta_loss.backward()
    assert torch.allclose(vanilla_loss, jta_loss, atol=1e-7)
    assert torch.allclose(vanilla_logits.grad, jta_logits.grad, atol=1e-7)


def test_context_backend_zeroes_invalid_states_and_is_distributed_consistent():
    advantages = torch.tensor(
        [[[-1.0, 0.5], [0.25, -0.75]], [[0.5, -0.25], [1.0, -1.0]]]
    )
    valid = torch.tensor([[True, False], [True, True]])
    hidden = torch.randn(2, 2, 6)
    reference = _reference(advantages)
    selector = JTASelector(
        epsilon=0.1,
        sketch_dim=32,
        embedding_backend="topk_context_tensorsketch",
    )

    class IdentityDistributed:
        enabled = True

        @staticmethod
        def sum_tensor(value):
            return value.detach().clone()

        @staticmethod
        def sum_int(value):
            return value

    local = selector.allocate(
        reference,
        valid,
        torch.tensor([True, True]),
        num_responses=1,
        hidden_states=hidden,
    )
    distributed = selector.allocate(
        reference,
        valid,
        torch.tensor([True, True]),
        num_responses=1,
        distributed=IdentityDistributed(),
        hidden_states=hidden,
    )
    assert torch.equal(local.weights, distributed.weights)
    assert torch.equal(local.diagnostics["reference"], distributed.diagnostics["reference"])
    assert torch.all(local.diagnostics["hidden_state_norm"][~valid] == 0)
    assert torch.all(local.diagnostics["tensorsketch_embedding_norm"][~valid] == 0)
