from __future__ import annotations

import torch

from .base import SelectorOutput, robust_quantile_normalize, scatter_valid


class TASelector:
    """Faithful tensor implementation of the released TA-OPD Dlearn selector."""

    def __init__(
        self,
        top_k: int = 16,
        q_low: float = 0.05,
        q_high: float = 0.95,
        eps: float = 1e-12,
    ):
        self.top_k = int(top_k)
        self.q_low = float(q_low)
        self.q_high = float(q_high)
        self.eps = float(eps)

    @torch.inference_mode()
    def compute_scores_from_topk(
        self,
        student_top_k_ids: torch.Tensor,
        teacher_top_k_ids: torch.Tensor,
        student_top_k_log_probs: torch.Tensor,
        teacher_on_student_log_probs: torch.Tensor,
        teacher_top_k_log_probs: torch.Tensor,
        student_on_teacher_log_probs: torch.Tensor,
        valid_mask: torch.Tensor,
        *,
        normalize: bool = True,
        token_chunk_size: int = 2048,
    ) -> SelectorOutput:
        """Compute exact TA statistics on the student Top-K support.

        Full-vocabulary log-probabilities differ from logits only by a constant
        at each position. Renormalizing both models on the student Top-K IDs
        therefore produces exactly the same D and C as
        ``compute_scores_from_logits`` without retaining a [B,T,V] tensor for
        either model.  Teacher Top-K tensors remain accepted for compatibility
        with the shared scorer but are not used to construct the support.
        """
        tensors = (
            student_top_k_ids,
            teacher_top_k_ids,
            student_top_k_log_probs,
            teacher_on_student_log_probs,
            teacher_top_k_log_probs,
            student_on_teacher_log_probs,
        )
        if any(value.ndim != 3 for value in tensors):
            raise ValueError("Compact TA inputs must have shape [batch, time, K]")
        if any(value.shape != tensors[0].shape for value in tensors[1:]):
            raise ValueError("Compact TA Top-K inputs must share one shape")
        if tensors[0].shape[:2] != valid_mask.shape:
            raise ValueError("Compact TA inputs must align with valid_mask")
        flat_valid = valid_mask.reshape(-1)
        valid_indices = flat_valid.nonzero(as_tuple=False).squeeze(-1)
        if valid_indices.numel() == 0:
            zeros = torch.zeros_like(valid_mask, dtype=torch.float32)
            return SelectorOutput(
                zeros,
                {name: zeros for name in ("D", "C", "D_norm", "C_norm", "s_TA")},
            )

        flattened = [value.reshape(-1, value.shape[-1]) for value in tensors]
        disagreement_chunks, compatibility_chunks = [], []
        chunk_size = max(1, int(token_chunk_size))
        for begin in range(0, valid_indices.numel(), chunk_size):
            indices = valid_indices[begin : begin + chunk_size]
            _stu_ids, _tea_ids, stu_logp, tea_on_stu, _tea_logp, _stu_on_tea = [
                value.index_select(0, indices) for value in flattened
            ]
            p_norm = torch.softmax(stu_logp.float(), dim=-1).clamp_min(self.eps)
            q_norm = torch.softmax(tea_on_stu.float(), dim=-1).clamp_min(self.eps)
            disagreement_chunks.append(
                (q_norm * (q_norm.log() - p_norm.log())).sum(dim=-1)
            )
            compatibility_chunks.append(
                torch.exp(tea_on_stu.float()).sum(dim=-1).clamp_(0.0, 1.0)
            )
        disagreement = torch.cat(disagreement_chunks)
        compatibility = torch.cat(compatibility_chunks)
        if normalize:
            d_norm = robust_quantile_normalize(
                disagreement, self.q_low, self.q_high, self.eps
            )
            c_norm = robust_quantile_normalize(
                compatibility, self.q_low, self.q_high, self.eps
            )
        else:
            d_norm = torch.zeros_like(disagreement)
            c_norm = torch.zeros_like(compatibility)
        score = d_norm * c_norm
        diagnostics = {
            "D": scatter_valid(disagreement, valid_mask),
            "C": scatter_valid(compatibility, valid_mask),
            "D_norm": scatter_valid(d_norm, valid_mask),
            "C_norm": scatter_valid(c_norm, valid_mask),
            "s_TA": scatter_valid(score, valid_mask),
            "compatibility_convention": "teacher_mass_on_student_topk",
            "score_input": "compact_teacher_gathered_on_student_topk",
            "support_definition": "student_topk",
        }
        return SelectorOutput(diagnostics["s_TA"], diagnostics)

    @torch.inference_mode()
    def compute_scores(
        self,
        student_probs: torch.Tensor,
        teacher_probs: torch.Tensor,
        valid_mask: torch.Tensor,
        *,
        normalize: bool = True,
    ):
        if student_probs.shape != teacher_probs.shape:
            raise ValueError(
                "Student and teacher probability tensors must have identical shape"
            )
        if student_probs.shape[:2] != valid_mask.shape:
            raise ValueError("Probability leading dimensions must match valid_mask")
        flat_valid = valid_mask.reshape(-1)
        flat_p = student_probs.reshape(-1, student_probs.shape[-1])
        flat_q = teacher_probs.reshape(-1, teacher_probs.shape[-1])
        p, q = (
            (flat_p, flat_q)
            if bool(flat_valid.all())
            else (flat_p[flat_valid], flat_q[flat_valid])
        )
        if p.numel() == 0:
            zeros = torch.zeros_like(valid_mask, dtype=torch.float32)
            return SelectorOutput(
                zeros,
                {name: zeros for name in ("D", "C", "D_norm", "C_norm", "s_TA")},
            )

        k = min(self.top_k, p.shape[-1])
        _, p_top_ids = torch.topk(p, k=k, dim=-1)
        # U is exactly the student's Top-K ID set.  The teacher is evaluated
        # on those same IDs before both distributions are conditionalized.
        p_support = p.gather(-1, p_top_ids)
        q_support = q.gather(-1, p_top_ids)
        p_norm = (
            p_support / p_support.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        ).clamp_min(self.eps)
        q_norm = (
            q_support / q_support.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        ).clamp_min(self.eps)
        disagreement = (q_norm * (q_norm.log() - p_norm.log())).sum(dim=-1)

        compatibility = q.gather(-1, p_top_ids).sum(dim=-1).clamp_(0.0, 1.0)
        if normalize:
            d_norm = robust_quantile_normalize(
                disagreement, self.q_low, self.q_high, self.eps
            )
            c_norm = robust_quantile_normalize(
                compatibility, self.q_low, self.q_high, self.eps
            )
        else:
            d_norm = torch.zeros_like(disagreement)
            c_norm = torch.zeros_like(compatibility)
        score = d_norm * c_norm
        diagnostics = {
            "D": scatter_valid(disagreement, valid_mask),
            "C": scatter_valid(compatibility, valid_mask),
            "D_norm": scatter_valid(d_norm, valid_mask),
            "C_norm": scatter_valid(c_norm, valid_mask),
            "s_TA": scatter_valid(score, valid_mask),
            "compatibility_convention": "teacher_mass_on_student_topk",
            "support_definition": "student_topk",
        }
        return SelectorOutput(diagnostics["s_TA"], diagnostics)

    @torch.inference_mode()
    def compute_scores_from_logits(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        student_log_normalizers: torch.Tensor,
        teacher_log_normalizers: torch.Tensor,
        valid_mask: torch.Tensor,
        *,
        normalize: bool = True,
        token_chunk_size: int = 2048,
    ) -> SelectorOutput:
        """Same TA definition while materializing only student Top-K values."""
        if student_logits.shape != teacher_logits.shape:
            raise ValueError("Student and teacher logits must have identical shape")
        if student_logits.shape[:2] != valid_mask.shape:
            raise ValueError("Logit leading dimensions must match valid_mask")
        if (
            student_log_normalizers.shape != valid_mask.shape
            or teacher_log_normalizers.shape != valid_mask.shape
        ):
            raise ValueError("Log-normalizer shapes must match valid_mask")
        flat_valid = valid_mask.reshape(-1)
        valid_indices = flat_valid.nonzero(as_tuple=False).squeeze(-1)
        if valid_indices.numel() == 0:
            zeros = torch.zeros_like(valid_mask, dtype=torch.float32)
            return SelectorOutput(
                zeros,
                {name: zeros for name in ("D", "C", "D_norm", "C_norm", "s_TA")},
            )

        flat_p = student_logits.reshape(-1, student_logits.shape[-1])
        flat_q = teacher_logits.reshape(-1, teacher_logits.shape[-1])
        flat_q_log_z = teacher_log_normalizers.reshape(-1)
        chunk_size = max(1, int(token_chunk_size))
        disagreement_chunks, compatibility_chunks = [], []
        k = min(self.top_k, flat_p.shape[-1])
        for begin in range(0, valid_indices.numel(), chunk_size):
            indices = valid_indices[begin : begin + chunk_size]
            p = flat_p.index_select(0, indices)
            q = flat_q.index_select(0, indices)
            q_log_z = flat_q_log_z.index_select(0, indices).float()
            p_top_ids = torch.topk(p, k=k, dim=-1).indices
            p_support_logits = p.gather(-1, p_top_ids).float()
            q_support_logits = q.gather(-1, p_top_ids).float()
            p_norm = torch.softmax(p_support_logits, dim=-1).clamp_min(self.eps)
            q_norm = torch.softmax(q_support_logits, dim=-1).clamp_min(self.eps)
            disagreement_chunks.append(
                (q_norm * (q_norm.log() - p_norm.log())).sum(dim=-1)
            )
            compatibility_chunks.append(
                torch.exp(q.gather(-1, p_top_ids).float() - q_log_z.unsqueeze(-1))
                .sum(dim=-1)
                .clamp_(0.0, 1.0)
            )
        disagreement = torch.cat(disagreement_chunks)
        compatibility = torch.cat(compatibility_chunks)
        if normalize:
            d_norm = robust_quantile_normalize(
                disagreement, self.q_low, self.q_high, self.eps
            )
            c_norm = robust_quantile_normalize(
                compatibility, self.q_low, self.q_high, self.eps
            )
        else:
            d_norm = torch.zeros_like(disagreement)
            c_norm = torch.zeros_like(compatibility)
        score = d_norm * c_norm
        diagnostics = {
            "D": scatter_valid(disagreement, valid_mask),
            "C": scatter_valid(compatibility, valid_mask),
            "D_norm": scatter_valid(d_norm, valid_mask),
            "C_norm": scatter_valid(c_norm, valid_mask),
            "s_TA": scatter_valid(score, valid_mask),
            "compatibility_convention": "teacher_mass_on_student_topk",
            "score_input": "bf16_logits_with_fp32_student_topk_and_normalizer",
            "support_definition": "student_topk",
        }
        return SelectorOutput(diagnostics["s_TA"], diagnostics)
