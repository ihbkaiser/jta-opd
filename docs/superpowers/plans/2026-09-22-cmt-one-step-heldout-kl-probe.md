# CMT One-Step Held-Out KL Probe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-ml:subagent-driven-development (recommended) or superpowers-ml:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Log paired, realized uniform-OPD and CMT one-step held-out KL improvements after every real CMT optimizer step without changing the real training trajectory.

**Architecture:** Persist one deterministic set of initial-student CompetitionMath test prefixes, build a frozen pre-step Student Top-16 KL reference, and score both branches against it. Refactor the existing PPO-group update into one reusable helper; run it once with uniform weights under a restorable CPU snapshot, restore exactly, then run the normal CMT update and append one atomic paired record.

**Tech Stack:** Python 3.11, PyTorch/FSDP, Hugging Face Transformers, vLLM rollout server, YAML, pytest, JSONL/CSV.

**Experiment card:** `docs/superpowers/experiments/2026-09-22-cmt-one-step-heldout-kl-probe-card.md`

---

## File Structure

- Create `b200_experiment/one_step_kl_probe.py`: configuration validation, finite-support KL reference/scoring, paired-record construction, and probe orchestration interfaces.
- Create `b200_experiment/heldout_probe.py`: deterministic benchmark selection, frozen rollout serialization, content hashes, rank-local reconstruction, and resume verification.
- Create `b200_experiment/training_snapshot.py`: CPU capture/restore/verification for local model shards, optimizer state, and Python/NumPy/Torch/CUDA RNG.
- Create `b200_experiment/probe_logging.py`: atomic JSONL/CSV upsert keyed by optimizer step.
- Modify `b200_experiment/trainer.py`: prepare the held-out artifact, refactor one PPO-group update, run the paired probe, and report metadata.
- Modify `b200_experiment/resume.py`: rewind probe rows consistently with existing training history.
- Modify `configs/qwen3_b200_base.yaml` and `configs/qwen3_b200_cmt.yaml`: disabled shared defaults and enabled CMT production settings.
- Modify `scripts/common_b200.sh` and `scripts/train_cmt_b200.sh`: environment overrides and launch summary.
- Create `tests/test_one_step_kl_probe.py`, `tests/test_heldout_probe.py`, `tests/test_training_snapshot.py`, and `tests/test_probe_logging.py`.
- Modify `tests/test_training_launchers.py`, `tests/test_resume.py`, and `README.md`.

### Task 1: Lock Configuration and Finite-Support KL Semantics

**Files:**
- Create: `b200_experiment/one_step_kl_probe.py`
- Create: `tests/test_one_step_kl_probe.py`
- Modify: `configs/qwen3_b200_base.yaml`
- Modify: `configs/qwen3_b200_cmt.yaml`

- [ ] **Step 1: Write failing configuration and KL tests**

Add tests that import the wished-for API:

```python
from b200_experiment.one_step_kl_probe import (
    OneStepKLProbeConfig,
    conditional_reverse_kl,
    paired_improvement,
)

def test_probe_config_rejects_non_cmt_enabled_method():
    with pytest.raises(ValueError, match="only supported for method=cmt"):
        OneStepKLProbeConfig.from_mapping(
            {"enabled": True, "subset_size": 64, "interval_steps": 1},
            method="opd",
        )

def test_conditional_reverse_kl_matches_manual_distribution():
    student = torch.log(torch.tensor([[[0.75, 0.25]]]))
    teacher = torch.log(torch.tensor([[[0.50, 0.50]]]))
    valid = torch.tensor([[True]])
    total, count = conditional_reverse_kl(student, teacher, valid)
    expected = 0.75 * math.log(1.5) + 0.25 * math.log(0.5)
    assert total.item() == pytest.approx(expected)
    assert count == 1

def test_paired_improvement_uses_before_minus_after():
    row = paired_improvement(0.7, 0.5, 0.4)
    assert row == pytest.approx({
        "delta_uniform": 0.2,
        "delta_cmt": 0.3,
        "paired_gap": 0.1,
    })
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `pytest -q tests/test_one_step_kl_probe.py`

Expected: collection fails because `b200_experiment.one_step_kl_probe` does not exist.

- [ ] **Step 3: Implement the minimal pure API**

Create immutable `OneStepKLProbeConfig` with validated fields `enabled`,
`benchmark`, `subset_size`, `seed`, `interval_steps`, `top_k`, `metric`,
`failure_policy`, `artifact_subdir`, `max_new_tokens`, `temperature`, `top_p`,
`generation_batch_size`, and `score_micro_batch_size`. Implement:

```python
def conditional_reverse_kl(student_log_probs, teacher_log_probs, valid_mask):
    p_log = student_log_probs.float() - torch.logsumexp(
        student_log_probs.float(), dim=-1, keepdim=True
    )
    q_log = teacher_log_probs.float() - torch.logsumexp(
        teacher_log_probs.float(), dim=-1, keepdim=True
    )
    per_token = (p_log.exp() * (p_log - q_log)).sum(dim=-1)
    selected = per_token[valid_mask.bool()]
    if selected.numel() == 0:
        raise ValueError("Held-out KL reference contains no valid response tokens")
    return selected.double().sum(), int(selected.numel())

def paired_improvement(before, after_uniform, after_cmt):
    uniform = float(before) - float(after_uniform)
    cmt = float(before) - float(after_cmt)
    return {"delta_uniform": uniform, "delta_cmt": cmt,
            "paired_gap": cmt - uniform}
```

Add disabled defaults to the base YAML and the approved enabled values to the CMT YAML. Validate `top_k=16`, positive subset/interval/batches, finite generation settings, and `failure_policy in {error,warn}`.

- [ ] **Step 4: Run tests and verify GREEN**

Run: `pytest -q tests/test_one_step_kl_probe.py`

Expected: all Task 1 tests pass.

- [ ] **Step 5: Commit**

```bash
git add b200_experiment/one_step_kl_probe.py tests/test_one_step_kl_probe.py configs/qwen3_b200_base.yaml configs/qwen3_b200_cmt.yaml
git commit -m "Add held-out KL probe semantics"
```

### Task 2: Persist Deterministic Held-Out Prefixes

**Files:**
- Create: `b200_experiment/heldout_probe.py`
- Create: `tests/test_heldout_probe.py`

- [ ] **Step 1: Write failing subset, hash, and resume tests**

Test that `select_heldout_records(records, 3, seed)` returns the same IDs for the
same seed, rejects duplicate IDs and oversize subsets, and changes for another
seed. Construct a small `RolloutBatch`, save it through `HeldoutPrefixStore`, load
it, and assert every tensor and manifest hash matches. Mutate one saved response
token and assert `load()` raises `ValueError("held-out prefix hash mismatch")`.

- [ ] **Step 2: Run and verify RED**

Run: `pytest -q tests/test_heldout_probe.py`

Expected: failure because the held-out store API is absent.

- [ ] **Step 3: Implement deterministic selection and artifact storage**

Use `random.Random(seed).sample(range(len(records)), subset_size)` followed by
stable index ordering. Hash canonical UTF-8 JSON for IDs/settings and contiguous
CPU bytes for `input_ids`, `attention_mask`, `response_ids`, and `valid_mask`.
Save tensors to `<artifact_dir>/heldout_prefixes.rank-XXXXX.pt`, gather rank hashes,
and let rank 0 atomically write `heldout_manifest.json`. The manifest must include
benchmark source path, selected IDs, tokenizer/model identifiers, generation
settings, world size, per-rank row counts/hashes, and aggregate prefix hash.

Expose:

```python
select_heldout_records(records, subset_size, seed) -> list[dict]
HeldoutPrefixStore(root, distributed).save(local_rollout, manifest_fields)
HeldoutPrefixStore(root, distributed).load(device) -> HeldoutPrefixArtifact
```

- [ ] **Step 4: Run and verify GREEN**

Run: `pytest -q tests/test_heldout_probe.py`

Expected: all deterministic selection, serialization, and corruption tests pass.

- [ ] **Step 5: Commit**

```bash
git add b200_experiment/heldout_probe.py tests/test_heldout_probe.py
git commit -m "Persist fixed held-out probe prefixes"
```

### Task 3: Capture and Restore Shadow Training State

**Files:**
- Create: `b200_experiment/training_snapshot.py`
- Create: `tests/test_training_snapshot.py`
- Modify: `tests/test_fsdp_distributed.py`

- [ ] **Step 1: Write failing single-process restoration tests**

Build a two-layer model with AdamW, take one warm-up step so moments exist,
capture `TrainingStateSnapshot`, perform another step and consume Python, NumPy,
CPU Torch, and CUDA RNG, then restore. Assert bitwise equality for parameters,
buffers, optimizer state tensors/scalars, RNG states, and a caller-owned counter.
Also assert `verify_restored()` detects one deliberately modified parameter.

- [ ] **Step 2: Run and verify RED**

Run: `pytest -q tests/test_training_snapshot.py`

Expected: import failure for `TrainingStateSnapshot`.

- [ ] **Step 3: Implement recursive CPU cloning and exact restoration**

Implement `clone_to_cpu(value)` for nested dict/list/tuple/tensor structures.
Capture local `named_parameters()` and `named_buffers()` by canonical name,
`optimizer.state_dict()`, `random.getstate()`, `np.random.get_state()`,
`torch.get_rng_state()`, and `torch.cuda.get_rng_state_all()` when CUDA exists.
Restore tensors with `copy_`, call `optimizer.load_state_dict`, restore every RNG,
zero gradients, and compare shape/dtype/value plus a SHA-256 digest in
`verify_restored`.

This intentionally snapshots local FSDP shards: production uses
`use_orig_params=true`, and every rank captures/restores its own shard without a
rank-0 full-state gather.

- [ ] **Step 4: Run single-process tests and verify GREEN**

Run: `pytest -q tests/test_training_snapshot.py`

Expected: all snapshot tests pass.

- [ ] **Step 5: Add and run a two-rank FSDP restoration test**

Extend the existing distributed test worker to capture, mutate with one AdamW
step, restore, verify on both ranks, and all-reduce a success flag.

Run: `pytest -q tests/test_fsdp_distributed.py -k shadow_snapshot`

Expected: the two-rank test passes or skips only when CUDA/FSDP is unavailable.

- [ ] **Step 6: Commit**

```bash
git add b200_experiment/training_snapshot.py tests/test_training_snapshot.py tests/test_fsdp_distributed.py
git commit -m "Add exact shadow training snapshots"
```

### Task 4: Add Resume-Safe Paired Logging

**Files:**
- Create: `b200_experiment/probe_logging.py`
- Create: `tests/test_probe_logging.py`
- Modify: `b200_experiment/resume.py`
- Modify: `tests/test_resume.py`

- [ ] **Step 1: Write failing logger tests**

Create two valid paired records, upsert step 1 twice with changed values, and
assert JSONL and CSV contain one step-1 row with the replacement. Assert missing,
non-finite, or inconsistent improvement fields are rejected. Add a resume test
that rewinds rows after checkpoint step 3 while preserving steps 1-3.

- [ ] **Step 2: Run and verify RED**

Run: `pytest -q tests/test_probe_logging.py tests/test_resume.py -k 'probe or rewind'`

Expected: failures for the missing logger/rewind support.

- [ ] **Step 3: Implement validated atomic upsert**

Implement `OneStepKLProbeLogger.upsert(record)` on rank 0. Recompute
`delta_uniform`, `delta_cmt`, and `paired_gap` from the three KL values and reject
disagreement above `1e-10`. Write a temporary JSONL/CSV and replace atomically.
Use `optimizer_step` as the unique integer key and deterministic field order.

Extend resume cleanup to recognize both probe log files and remove rows whose
step exceeds the resume checkpoint. Never remove `heldout_manifest.json` or
prefix tensors.

- [ ] **Step 4: Run and verify GREEN**

Run: `pytest -q tests/test_probe_logging.py tests/test_resume.py -k 'probe or rewind'`

Expected: logger and rewind tests pass.

- [ ] **Step 5: Commit**

```bash
git add b200_experiment/probe_logging.py tests/test_probe_logging.py b200_experiment/resume.py tests/test_resume.py
git commit -m "Log paired one-step KL probes safely"
```

### Task 5: Build the Frozen Held-Out KL Evaluator

**Files:**
- Modify: `b200_experiment/one_step_kl_probe.py`
- Modify: `tests/test_one_step_kl_probe.py`

- [ ] **Step 1: Write failing reference-reuse and global-reduction tests**

Use a tiny causal LM and frozen `RolloutBatch`. Assert
`FiniteSupportKLEvaluator.build_reference()` stores pre-step candidate IDs and
teacher candidate log-probabilities. Mutate the student so its new Top-K changes;
assert `score_student(reference)` still gathers the old IDs. With a fake
distributed reducer, assert the reported mean is global sum divided by global
valid-token count, not the mean of rank means.

- [ ] **Step 2: Run and verify RED**

Run: `pytest -q tests/test_one_step_kl_probe.py -k 'reference or global'`

Expected: failure because `FiniteSupportKLEvaluator` is not implemented.

- [ ] **Step 3: Implement reference building and student rescoring**

Use existing `score_original_rollout`: pre-step student scoring requests Top-K;
teacher scoring receives those IDs through `candidate_ids`. Store detached
candidate IDs, conditional teacher log-probabilities, valid mask, token count,
and prefix hash in `FiniteSupportKLReference`. For every score call, pass the
stored candidate IDs to `score_original_rollout`, conditionalize both sides, call
`conditional_reverse_kl`, then reduce numerator and count through
`DistributedContext.sum_float/sum_int`.

- [ ] **Step 4: Run and verify GREEN**

Run: `pytest -q tests/test_one_step_kl_probe.py`

Expected: all KL evaluator tests pass with finite FP32 values.

- [ ] **Step 5: Commit**

```bash
git add b200_experiment/one_step_kl_probe.py tests/test_one_step_kl_probe.py
git commit -m "Evaluate frozen-support held-out KL"
```

### Task 6: Reuse One PPO-Group Update for Uniform and CMT

**Files:**
- Modify: `b200_experiment/trainer.py`
- Modify: `tests/test_training_evaluation_distributed.py`
- Create: `tests/test_training_probe_integration.py`

- [ ] **Step 1: Write failing update-isolation tests**

Extract a tiny-model integration fixture that runs one PPO group. Assert the
existing no-probe path performs exactly one optimizer step and callback. With a
fake probe enabled, assert the group update helper is invoked first with all-one
weights and then with CMT weights, but the real callback and step counter fire
once. Assert both invocations receive identical row indices, reference, valid
mask, clipping settings, and optimizer-step start state.

- [ ] **Step 2: Run and verify RED**

Run: `pytest -q tests/test_training_probe_integration.py`

Expected: failure because the reusable PPO-group helper/probe hook is absent.

- [ ] **Step 3: Refactor without changing existing behavior**

Move lines responsible for zeroing gradients, micro-batch forward, candidate
gather, `topk_candidate_ppo_loss`, weighted normalization, backward, clipping,
`optimizer.step`, and per-group metrics into `_run_opd_optimizer_group(...)`.
Return `OPDGroupUpdateResult` containing loss/timing/gradient/ratio statistics.
Keep allocation, group partitioning, bookkeeping, and callbacks in
`_opd_train_step`. Run existing training tests before adding the probe branch.

Run: `pytest -q tests/test_training_probe_integration.py tests/test_training_evaluation_distributed.py tests/test_checkpoint_saving.py`

Expected: existing behavior tests pass and the new no-probe assertions pass.

- [ ] **Step 4: Add paired branch orchestration**

Add an optional `one_step_probe` argument to `_opd_train_step`. On scheduled CMT
groups:

```python
reference, kl_before = one_step_probe.prepare_before_step(step)
snapshot = TrainingStateSnapshot.capture(model, optimizer)
uniform_result = _run_opd_optimizer_group(..., weights=ppo_valid.float())
kl_after_uniform = one_step_probe.score(reference)
snapshot.restore(model, optimizer)
snapshot.verify_restored(model, optimizer)
cmt_result = _run_opd_optimizer_group(..., weights=ppo_weights)
kl_after_cmt = one_step_probe.score(reference)
one_step_probe.write_pair(...)
```

Do not append shadow metrics, fill allocation metadata, invoke callbacks, save
checkpoints, or increment the real step until the CMT result exists. On a probe
failure, restore first; `error` re-raises collectively, while `warn` may continue
only after successful restoration verification.

- [ ] **Step 5: Run and verify GREEN**

Run: `pytest -q tests/test_training_probe_integration.py tests/test_training_evaluation_distributed.py tests/test_checkpoint_saving.py`

Expected: paired branch, callback isolation, failure restore, and legacy path
tests all pass.

- [ ] **Step 6: Commit**

```bash
git add b200_experiment/trainer.py tests/test_training_probe_integration.py tests/test_training_evaluation_distributed.py
git commit -m "Run paired uniform and CMT optimizer probes"
```

### Task 7: Wire Fixed CompetitionMath Prefix Generation into Training

**Files:**
- Modify: `b200_experiment/trainer.py`
- Modify: `scripts/common_b200.sh`
- Modify: `scripts/train_cmt_b200.sh`
- Modify: `tests/test_training_launchers.py`
- Modify: `README.md`

- [ ] **Step 1: Write failing setup and launcher tests**

Test that enabled CMT setup loads only the configured `Competition-MATH` test
spec, selects 64 deterministic records, renders with `render_evaluation_prompt`,
partitions records contiguously across ranks, generates one response each, and
persists/reuses the artifact. Assert an OPD run or disabled CMT run never loads or
generates held-out data. Check launcher text exposes `ONE_STEP_KL_PROBE_ENABLED`,
`ONE_STEP_KL_PROBE_SUBSET_SIZE`, `ONE_STEP_KL_PROBE_SEED`,
`ONE_STEP_KL_PROBE_INTERVAL`, and probe response/scoring sizes.

- [ ] **Step 2: Run and verify RED**

Run: `pytest -q tests/test_training_probe_integration.py tests/test_training_launchers.py -k probe`

Expected: setup and environment override assertions fail.

- [ ] **Step 3: Implement startup preparation and resume loading**

After models/optimizer are ready and before optimizer step 1, create
`OneStepKLProbeRuntime` only for enabled CMT. For a new run, every rank loads the
same benchmark metadata, partitions the selected 64 records, tokenizes rendered
evaluation prompts with left padding, and calls the existing rollout engine with
the initial student weights and configured seed. Persist rank-local tensors and
the global manifest. On resume, load and hash-verify existing artifacts; fail if
they are absent or config/model/tokenizer identities disagree.

Pass the runtime to `_opd_train_step`; include its resolved configuration and
artifact hashes in run metadata and final metrics. Ensure the vLLM engine is
sleeping before HF held-out scoring and remains unaffected by shadow updates.

- [ ] **Step 4: Add launch overrides and documentation**

Map environment variables to the `one_step_kl_probe.*` config paths in
`scripts/common_b200.sh`, set requested CMT defaults in `train_cmt_b200.sh`, and
document cost, files, formulas, resume behavior, and the fact that the diagnostic
must not drive model selection.

- [ ] **Step 5: Run and verify GREEN**

Run: `pytest -q tests/test_training_probe_integration.py tests/test_training_launchers.py tests/test_data_evaluation_config.py`

Expected: fixed-prefix setup, disabled-path isolation, launcher, and dataset tests
pass.

- [ ] **Step 6: Commit**

```bash
git add b200_experiment/trainer.py scripts/common_b200.sh scripts/train_cmt_b200.sh tests/test_training_launchers.py README.md
git commit -m "Wire held-out KL probes into CMT training"
```

### Task 8: Verification Ladder and Handoff

**Files:**
- Modify: `RUN_B200.md`
- Modify: `README.md`

- [ ] **Step 1: R1 static and full CPU suite**

Run:

```bash
python -m compileall -q b200_experiment
pytest -q tests/test_one_step_kl_probe.py tests/test_heldout_probe.py tests/test_training_snapshot.py tests/test_probe_logging.py tests/test_training_probe_integration.py
pytest -q
```

Promotion artifact: clean command output plus a resolved CMT config showing the
enabled 64-example, interval-1 probe. Stop on any failure.

- [ ] **Step 2: R2 deterministic tiny forward/backward**

Run the tiny integration test twice under the same seed and compare emitted
JSONL bytes. Require finite loss/KL, non-zero gradients, exact state restoration,
and identical paired records.

Promotion artifact: `test-artifacts/one_step_kl_probe/tiny-pairs.jsonl` and the
captured test log. Stop if records differ or any tensor is non-finite.

- [ ] **Step 3: R4 actual launcher smoke on B200**

Document and run a two-step CMT launch with 4 held-out examples, short response
length, checkpoint at step 1, resume to step 2, and `interval_steps=1` using the
real `scripts/train_cmt_b200.sh` path.

Promotion artifact: checkpoint-1, final checkpoint, one persistent held-out
manifest/hash, and exactly two finite unique JSONL/CSV rows. Stop if resume
regenerates prefixes, shadow callbacks appear, or step keys duplicate.

- [ ] **Step 4: R5 short real-data pilot**

Run the production 64-example probe for a short configured pilot. Inspect finite
paired gaps, probe runtime, host/GPU memory, and training metrics. Do not use the
observed paired gap to tune the locked run.

Promotion artifact: pilot `one_step_kl_probe.jsonl`, resolved config, metadata,
and runtime/memory summary. Promote to the full run only if all scheduled pairs
are complete and there is no training-state drift.

- [ ] **Step 5: R6 user-run full study and later result review**

The user runs the full B200 experiment and returns the immutable paired logs.
Analyze the locked primary metric from the experiment card, including a paired
bootstrap confidence interval. Plotting and the R7 decision memo are separate
follow-up work and must not modify this run.

- [ ] **Step 6: Commit verification documentation**

```bash
git add RUN_B200.md README.md
git commit -m "Document CMT KL probe verification ladder"
```

## Plan Self-Review

- Spec coverage: fixed prefixes, frozen support, exact paired updates, FSDP-global
  metric, atomic logs, resume, configuration, isolation, tests, and cost are each
  assigned to a task.
- Type consistency: `OneStepKLProbeConfig`, `HeldoutPrefixStore`,
  `TrainingStateSnapshot`, `FiniteSupportKLEvaluator`, and
  `OneStepKLProbeLogger` retain the same names and responsibilities throughout.
- Scope: plotting is explicitly excluded; the plan ends with logs suitable for a
  later paired figure.
- Verification: R1/R2/R4/R5/R6 artifacts and promotion/stop conditions are named;
  passing unit tests is not treated as empirical success.
