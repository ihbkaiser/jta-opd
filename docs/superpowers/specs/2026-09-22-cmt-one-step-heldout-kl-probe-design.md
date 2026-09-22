# CMT One-Step Held-Out KL Probe

## Goal

Add an opt-in diagnostic to normal CMT training that measures paired, realized
one-optimizer-step KL improvement for uniform OPD and CMT. Each pair must start
from the same pre-step student, optimizer state, training PPO group, and fixed
held-out CompetitionMath prefixes. The uniform branch is diagnostic only; after
it is measured, training must be restored exactly and continue with the real CMT
update.

This diagnostic produces data for a later paired plot. Plot generation is not
part of this change.

## Research Question

For a fixed held-out set of prefixes, does one CMT-weighted OPD optimizer step
reduce the student-to-teacher finite-support reverse KL more than the same step
with uniform OPD weights?

For optimizer step `s`, report

```text
delta_uniform_s = KL_before_s - KL_after_uniform_s
delta_cmt_s     = KL_before_s - KL_after_cmt_s
paired_gap_s    = delta_cmt_s - delta_uniform_s
```

Positive improvement means that the student moved closer to the teacher.

## Fixed Held-Out Prefix Set

At the start of a new run:

1. Load the configured `Competition-MATH` test benchmark, never the training
   parquet.
2. Select exactly 64 examples without replacement using a dedicated probe seed.
   Selection is deterministic and independent of data-loader rank partitioning.
3. Render prompts with the same tokenizer and no-think template used by training.
4. Generate one response per selected prompt from the initial student before the
   first optimizer update, using explicit probe generation settings.
5. Freeze the resulting prompt-response token prefixes for the complete run.
6. Persist the selected example identifiers, generation settings, token tensors,
   valid masks, and a content hash below the run output directory.

On resume, load the persisted artifact and verify its hash. Do not regenerate
prefixes or silently choose a different subset.

The probe artifact contains no gradients and never enters the training data
loader, loss, selector, checkpoint selection, or early-stopping decisions.

## KL Metric

At the beginning of every probed optimizer step, score the frozen prefixes with
the current pre-step student and frozen teacher. At every valid response
position, define `S_t` as the pre-step Student Top-K IDs, with `K=16` by default.
Freeze these IDs and the teacher probabilities for both branches of that step.

Conditionalize student and teacher probabilities on the same `S_t` and compute

```text
KL_t = KL(p_student,S_t || q_teacher,S_t)
```

The reported scalar is the global valid-token mean across all distributed ranks.
Post-update student KL must be evaluated on the same frozen prefixes, candidate
IDs, and teacher targets used for the corresponding pre-step value. This avoids
moving-support and regenerated-trajectory confounds within a paired update.

The implementation may also log forward KL and teacher cross-entropy when those
values are available from the same logits, but reverse KL is the sole primary
metric and `paired_gap` always refers to reverse KL.

## Paired Update Protocol

For each global PPO group that produces one optimizer step:

1. Construct the normal CMT allocation and the uniform allocation on the exact
   same valid training tokens. Uniform OPD assigns weight one to every valid
   token.
2. Measure the held-out pre-step reverse KL.
3. Capture the complete restorable pre-step state needed by training: distributed
   student parameters, optimizer state, CPU and CUDA RNG state, and any mutable
   scaler/scheduler state that exists.
4. Apply one uniform OPD optimizer step with the same training PPO group and
   normal OPD loss implementation.
5. Measure held-out reverse KL and record `delta_uniform`.
6. Restore and verify the pre-step state. The uniform branch must leave no
   parameter, optimizer, RNG, counter, checkpoint, evaluation, or logging side
   effect in the real run.
7. Apply the normal CMT optimizer step once.
8. Measure held-out reverse KL and record `delta_cmt`.
9. Continue normal training from the CMT-updated state.

Callbacks that save checkpoints or launch benchmark evaluation run only for the
real CMT update. The shadow update does not increment the optimizer-step counter.

If exact state restoration cannot be verified, fail the probe before the real
update rather than emit an unpaired measurement or continue from the shadow
branch.

## Distributed and Numerical Semantics

- The probe is collective across the same FSDP ranks as training.
- KL numerators and valid-token counts are reduced globally; no rank computes an
  independently normalized metric.
- Scoring runs under inference mode in FP32 reductions with finite checks.
- Padding and tokens after EOS are excluded.
- A probe pair is written only after both post-update measurements succeed.
- A failed probe aborts by default. An explicit `failure_policy: warn` may skip
  the pair only after the pre-step state has been restored and verified.

## Configuration

Add a dedicated configuration block with the following production defaults:

```yaml
one_step_kl_probe:
  enabled: true
  benchmark: Competition-MATH
  subset_size: 64
  seed: 20260922
  interval_steps: 1
  top_k: 16
  metric: conditional_reverse_kl
  failure_policy: error
  artifact_subdir: one_step_kl_probe
```

Probe rollout batch size, scoring micro-batch size, response length, temperature,
and top-p remain explicit configuration fields, with memory-safe defaults derived
from the training rollout configuration. The resolved values are stored in the
artifact manifest and each log row.

Existing non-CMT configurations remain disabled unless explicitly enabled. The
shipped CMT configuration enables the probe as requested.

## Outputs

Write append-safe, resume-safe files below
`<output_dir>/one_step_kl_probe/`:

- `heldout_manifest.json`: dataset identity, selected IDs, generation settings,
  tokenizer/model identity, tensor artifact path, and content hash.
- `heldout_prefixes.pt`: frozen prompt-response token tensors and masks.
- `one_step_kl_probe.jsonl`: canonical machine-readable paired measurements.
- `one_step_kl_probe.csv`: flat convenience export for later plotting.

Each paired measurement contains at least:

```text
run_id, optimizer_step, rollout_id, ppo_group_index,
benchmark, subset_size, heldout_example_hash, heldout_prefix_hash,
valid_prefix_token_count, top_k, metric,
kl_before, kl_after_uniform, kl_after_cmt,
delta_uniform, delta_cmt, paired_gap,
uniform_update_loss, cmt_update_loss,
probe_time_sec, uniform_branch_time_sec, cmt_eval_time_sec
```

Rows use `optimizer_step` as the unique resume key. A retry replaces an incomplete
or duplicate row atomically rather than appending a second observation.

## Isolation Boundaries

The implementation should separate four responsibilities:

1. `HeldoutPrefixStore`: deterministic selection, initial rollout persistence,
   hash verification, and resume loading.
2. `FiniteSupportKLEvaluator`: build a pre-step frozen Top-K reference and score
   current student parameters against it.
3. `TrainingStateSnapshot`: exact capture, restore, and equality verification of
   state mutated by a shadow optimizer step.
4. `OneStepKLProbeLogger`: validate paired records and write JSONL/CSV atomically.

Trainer integration orchestrates these units around one existing global PPO
optimizer group. It must not duplicate the OPD loss formula or CMT allocator.

## Tests and Verification

Follow test-first development. Cheap deterministic tests must cover:

- fixed subset selection is seed-stable and excludes training data;
- held-out artifact hash detects changed tokens or masks;
- conditional reverse KL is zero for identical distributions and has the correct
  direction for a known example;
- post-update scoring reuses pre-step candidate IDs and teacher targets;
- improvement and paired-gap signs match their definitions;
- a uniform shadow branch followed by restore reproduces parameters, optimizer
  tensors, RNG state, and counters exactly on a tiny model;
- the real CMT branch starts from the same pre-step state as uniform;
- distributed reduction uses global numerator and token count;
- shadow updates do not fire real-step callbacks or increment step counters;
- resume replaces duplicate step rows and reuses the held-out artifact;
- disabled mode leaves the existing training path unchanged.

Before a full B200 run, pass the unit suite and a tiny one-step CPU/CUDA smoke
probe. A green test suite establishes path correctness only; the logged paired
results determine whether CMT improves held-out KL.

## Cost and Interpretation

With `interval_steps=1`, every real step adds one shadow backward update and three
held-out student scoring points (`before`, `after uniform`, and `after CMT`). The
teacher target for a step is computed once and reused by both branches. This is
intentionally expensive because the requested quantity is realized one-step
improvement, not a first-order approximation.

The resulting metric measures teacher transfer on fixed student-generated
prefixes and a frozen finite Student Top-K support. It is not task accuracy,
causal credit, or full-vocabulary KL. No claim of CMT superiority is made until
paired logs are analyzed.
