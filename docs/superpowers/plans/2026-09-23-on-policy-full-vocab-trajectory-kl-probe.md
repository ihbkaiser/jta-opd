# On-Policy Full-Vocabulary Trajectory KL Probe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-ml:subagent-driven-development (recommended) or superpowers-ml:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the frozen-successor Top-K probe with a paired Uniform-vs-CMT diagnostic that generates branch-specific trajectories from 64 fixed Competition-MATH test roots and scores exact full-vocabulary reverse KL to the teacher.

**Architecture:** Persist only deterministic held-out prompt roots. After the Uniform shadow update and after the real CMT update, generate two 64-token on-policy rollouts per root with matched seeds, stream student and teacher full-vocabulary logits through bounded micro-batches, reduce exact reverse KL per visited state, and log step-level plus per-problem paired results. The probe remains outside the training loss and restores the Uniform shadow state exactly.

**Tech Stack:** Python, PyTorch, FSDP/DDP-compatible collectives, vLLM/HF rollout backends, pytest, YAML/Bash launch configuration.

---

### Task 1: Lock the trajectory-probe configuration

**Files:**
- Modify: `b200_experiment/one_step_kl_probe.py`
- Modify: `tests/test_one_step_kl_probe.py`

- [ ] Write failing tests asserting the defaults: Competition-MATH, 64 roots, two rollouts/root, horizon 64, interval 50, temperature/top-p 1, full-vocabulary reverse KL, seed 20260923, and score micro-batch 1.
- [ ] Run `pytest -q tests/test_one_step_kl_probe.py` with repository `PYTHONPATH`; expect failures on the old successor fields.
- [ ] Replace successor-specific configuration with the locked experiment-card fields and reject top-p other than 1 or metrics other than `full_vocab_reverse_kl`.
- [ ] Re-run the focused tests; expect all green.

### Task 2: Persist fixed held-out prompt roots

**Files:**
- Modify: `b200_experiment/heldout_probe.py`
- Modify: `tests/test_heldout_probe.py`

- [ ] Write failing round-trip and hash-integrity tests for `HeldoutRootStore`, containing prompt IDs, attention masks, local problem IDs, manifest settings, and an aggregate root hash.
- [ ] Run `pytest -q tests/test_heldout_probe.py`; expect missing-class failures.
- [ ] Implement atomic rank-local root artifacts and strict resume validation without generating responses at setup.
- [ ] Re-run held-out tests; expect all green.

### Task 3: Score exact full-vocabulary on-policy trajectory KL

**Files:**
- Create: `b200_experiment/trajectory_kl.py`
- Create: `tests/test_trajectory_kl.py`

- [ ] Write failing tests using tiny deterministic student/teacher models. Verify per-state `KL(p||q)`, fixed-horizon normalization, root-level averaging over two rollouts, EOS lengths, finite checks, and model-mode restoration.
- [ ] Run `pytest -q tests/test_trajectory_kl.py`; expect module-not-found failure.
- [ ] Implement `TrajectoryKLEvaluation` and `FullVocabularyTrajectoryKLEvaluator` using `score_student_teacher_rollout(..., compute_full_vocab_metrics=True)` and `KL=-E_p[log q-log p]`, retaining only `[batch,time]` reductions.
- [ ] Re-run the focused tests; expect all green.

### Task 4: Generate branch-specific held-out trajectories

**Files:**
- Create: `b200_experiment/trajectory_probe.py`
- Create: `tests/test_trajectory_probe.py`

- [ ] Write failing tests for deterministic root repetition, per-step matched seeds, generation batching, row-to-problem mapping, and `top_p=1`/temperature forwarding.
- [ ] Run `pytest -q tests/test_trajectory_probe.py`; expect module-not-found failure.
- [ ] Implement a builder that repeats each local root twice, generates at most 64 tokens from the current branch policy, concatenates batches, and returns stable metadata without touching global RNG.
- [ ] Re-run the focused tests; expect all green.

### Task 5: Pair Uniform and CMT outcomes and log analysis-ready artifacts

**Files:**
- Modify: `b200_experiment/probe_runtime.py`
- Modify: `b200_experiment/probe_logging.py`
- Modify: `tests/test_probe_logging.py`
- Modify: `tests/test_training_probe_integration.py`

- [ ] Write failing tests for branch-specific generation/scoring calls, positive `trajectory_gap = uniform_kl - cmt_kl`, equal-weight per-problem aggregation, separate per-problem JSONL, idempotent optimizer-step upserts, and unchanged real CMT state.
- [ ] Run the focused logger/runtime/integration tests; expect failures on the old frozen-successor schema.
- [ ] Replace the runtime and logger schema, gather rank-local problem rows, and keep exact Uniform snapshot restoration around generation/scoring.
- [ ] Re-run the focused tests; expect all green.

### Task 6: Load Competition-MATH roots and wire production configuration

**Files:**
- Modify: `b200_experiment/probe_runtime.py`
- Modify: `b200_experiment/trainer.py`
- Modify: `configs/qwen3_b200_cmt.yaml`
- Modify: `scripts/common_b200.sh`
- Modify: `scripts/train_cmt_b200.sh`
- Modify: `tests/test_training_probe_integration.py`

- [ ] Write failing setup tests proving deterministic Competition-MATH selection, root-only persistence, resume hash checks, divisibility across ranks, and exact launcher defaults.
- [ ] Run the setup tests; expect failures on the successor builder and old environment variables.
- [ ] Wire benchmark loading/rendering/tokenization, pass student/rollout engine/device/resume into setup, and expose the locked defaults in YAML and Bash.
- [ ] Re-run focused tests; expect all green.

### Task 7: Verification ladder and documentation

**Files:**
- Modify: `README.md`
- Modify: `HUONG_DAN_CHAY.md`
- Create: `docs/superpowers/experiments/2026-09-23-on-policy-full-vocab-trajectory-kl-probe-card.md`

- [ ] R1: Run import/config and all probe tests; artifact is clean pytest output plus resolved-default assertions.
- [ ] R2: Run deterministic tiny-model generation and exact-KL tests; artifact is finite per-state/root KL with expected analytic values.
- [ ] Run the full CPU regression suite with repository `PYTHONPATH`; artifact is the pytest summary.
- [ ] Document that the 64 selected Competition-MATH test problems are a diagnostic subset, not untouched final test data, plus the new output schemas and launch defaults.
- [ ] R4/R5 handoff: document the exact GPU smoke command and required artifacts (`resolved_config.yaml`, held-out root manifest, step/per-problem JSONL/CSV, timing fields). Do not claim empirical success until R6 completes across three seeds.
