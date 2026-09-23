# Experiment Card: On-Policy Full-Vocabulary Trajectory KL

## Question

After one matched optimizer update, does CMT produce near-future trajectories
that are closer to the teacher than Uniform OPD trajectories?

## Locked design

- Population: 64 fixed problems sampled once from Competition-MATH test.
- Subset seed: `20260923`; selected IDs, rendered token roots, and hashes are
  persisted and reused on resume.
- Comparison: one restorable Uniform OPD shadow update versus the real CMT
  update on the same PPO group and pre-update training state.
- Rollouts: two independent samples per problem per branch, maximum horizon 64,
  `temperature=1.0`, `top_p=1.0`, and matched sample seeds across branches.
- Schedule: every 50 optimizer steps.
- Scoring: exact full-vocabulary reverse KL
  `KL(pi_branch(.|s) || q_teacher(.|s))` at every visited state.
- Normalization: sum state KL over each trajectory, divide by the fixed horizon
  64 (post-EOS positions contribute zero), average rollouts within problem, then
  average problems with equal weight.
- Primary statistic: `trajectory_gap = D_H(Uniform,q) - D_H(CMT,q)`.
- Positive direction: positive gap favors CMT.

## Artifacts

All files are under `one_step_trajectory_kl_probe/`:

- `heldout_root_manifest.json` and `heldout_roots.rank-*.pt`
- `trajectory_kl_probe.jsonl` and `.csv` (one record per probed step)
- `trajectory_kl_per_problem.jsonl` and `.csv` (64 records per probed step)

The step record includes both branch KLs, gap, valid-state count, mean length,
early-EOS rate, trajectory hashes, update losses, and timing. Resume rewinds
both log granularities to the selected checkpoint without changing fixed roots.

## Analysis lock

Report the across-step mean gap and a paired bootstrap confidence interval.
Also report per-step gaps, branch length/EOS diagnostics, and probe runtime. Do
not tune this run from observed probe outcomes. Replicate the final comparison
across at least three training seeds and retain benchmark accuracy as the main
outcome measure.

## Interpretation boundary

A positive gap supports the specific consequence that the one-step CMT branch
visits trajectories with lower student-to-teacher KL than the Uniform branch.
It does not by itself establish better mathematical correctness, long-horizon
learning, or final benchmark performance. A null or negative result rejects
this diagnostic consequence at the chosen horizon and sampling distribution;
it does not alone prove that every possible CMT benefit is absent.

The fixed 64 Competition-MATH test problems are diagnostic data after first
use, not an untouched final test set.
