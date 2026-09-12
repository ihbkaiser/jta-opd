# CMT ablation and hyperparameter runs

This directory adds an experiment layer around the production BellmanOPD
implementation.  It does not replace the OPD loss, optimizer, rollout,
Gibbs/KL allocator, or evaluator.  At each token CMT computes its existing
detached diagnostics and only changes the score passed to the same allocator:

* `g`: `S_t = gain_t` (local PGT/CMT gain);
* `g_x`: `S_t = gain_t + successor_excess_t`;
* `g_d`: `S_t = learning_value_t`, the canonical CMT score
  `gain_t + sequential_gain_t`.

`g_d` is therefore numerically identical to an ordinary CMT run when all
configuration values are equal.  `successor_excess` is the CMT semantic
future-excess diagnostic; it is not `R`, `M`, `V`, or `H`.  Scores are detached
before the global KL-constrained Gibbs allocation and weighted OPD loss.

## Configure and run

All launchers resolve paths from their own location, so they can be called from
any working directory.  The most commonly edited exports are at the top of
`scripts/train.sh`:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export STORAGE_ROOT=/workspace/storage-shared
export STUDENT_MODEL='nlp/tungdd11/stable-on-policy-distillation/OPD/model/Qwen3-1.7B-Base'
export TEACHER_MODEL='models/Qwen3-8B'
export TRAIN_DATASET=competition_math       # or dapo_math/custom
export GLOBAL_BATCH_SIZE=64
export PPO_MINI_BATCH_SIZE=16
export MICRO_BATCH_SIZE_PER_GPU=8
export SEED=42
```

Training generation and evaluation generation are intentionally independent:
`MAX_RESPONSE_LEN` is set from `TRAIN_MAX_NEW_TOKENS` (default **4096**), while
`TRAIN_EVAL_MAX_NEW_TOKENS` is set from `EVAL_MAX_NEW_TOKENS` (default **7168**).
Training rollouts use `temperature=1.0, top_p=1.0`, as required by the exact
on-policy CMT estimator.  Periodic evaluation is enabled by default.

Run one arm:

```bash
bash ablation/scripts/train.sh g
bash ablation/scripts/train.sh g_x
bash ablation/scripts/train.sh g_d
```

Run all three arms (separate output directories, same seed and protocol):

```bash
bash ablation/scripts/run_ablation.sh
```

Override any export without editing files, for example a four-GPU run with a
short smoke budget:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 MAX_STEPS=2 \
  bash ablation/scripts/train.sh g_d
```

Every run is written to `ablation/outputs/<run-name>/` and refuses to append to
an existing run.  The trainer stores `resolved_config.yaml`, checkpoints,
`metrics.jsonl`, `eval_history.jsonl`, TensorBoard logs, and detached
`token_score_stats/`.  `ablation_spec.json` records arm, seed, hyperparameters,
token budgets, and git commit.

## Hyperparameter sweeps

The main search is on full CMT (`g_d`) only.  Keep the canonical
`successor_lambda=1.0`, `gamma=1.0`, and `rollout_top_p=1.0`; after selecting a
configuration, reuse it unchanged for all three arms.

```bash
bash ablation/scripts/sweep_epsilon.sh       # 0.1 0.25 0.5 1.0
bash ablation/scripts/sweep_topk.sh          # 8 16 32
bash ablation/scripts/sweep_lr.sh            # 5e-7 1e-6 2e-6
```

Values can be replaced without changing code:

```bash
EPSILON_VALUES='0.25 0.5' SEED=7 bash ablation/scripts/sweep.sh epsilon
TOP_K_VALUES='8 32' bash ablation/scripts/sweep.sh top_k
LR_VALUES='5e-7 2e-6' bash ablation/scripts/sweep.sh lr
```

`sweep.sh` is deliberately sequential.  To train independent values
concurrently, assign one physical GPU to each job:

```bash
GPU_LIST=0,1,2 \
EPSILON_VALUES='0.25 0.5 0.75' \
bash ablation/scripts/sweep_parallel.sh epsilon
```

This starts three separate `g_d` runs at the same time, with one GPU per run,
and waits for all of them.  Logs are saved under
`ablation/outputs/_sweep_logs/`.  The same interface works for `top_k` and
`lr`:

```bash
GPU_LIST=0,1,2 TOP_K_VALUES='8 16 32' \
  bash ablation/scripts/sweep_parallel.sh top_k
GPU_LIST=0,1,2 LR_VALUES='5e-7 1e-6 2e-6' \
  bash ablation/scripts/sweep_parallel.sh lr
```

Do not run two jobs on overlapping GPUs.  If one run itself needs multiple
GPUs, keep `sweep.sh` sequential or provide disjoint multi-GPU groups and
launch separate commands manually.

Each sweep run has a name such as
`analysis_epsilon_0.5_seed42`; an existing name is never overwritten.

## Resume and standalone evaluation

Resume uses the production launcher and preserves optimizer/checkpoint state:

```bash
RESUME_FROM_CHECKPOINT=/abs/path/ablation/outputs/ablation_g_d_seed42/checkpoint-000100 \
  RUN_NAME=ablation_g_d_seed42 \
  OUTPUT_DIR=/abs/path/ablation/outputs/ablation_g_d_seed42 \
  MAX_STEPS=200 bash ablation/scripts/train.sh g_d
```

Evaluate a final checkpoint or any numbered checkpoint.  The evaluator writes
details under `checkpoint_eval/` and upserts the corresponding row in the
run's `eval_history.jsonl` (repeating the same checkpoint/metric replaces only
that row):

```bash
EVAL_NUM_RESPONSES=8 EVAL_METRIC=pass@8 \
  bash scripts/eval_checkpoint_b200.sh cmt \
  ablation/outputs/ablation_g_d_seed42/final
```

For multi-GPU evaluation set `EVAL_WORLD_SIZE` and expose the same number of
GPUs; the existing evaluator launches one TP=1 vLLM replica per GPU and merges
rank shards.  `EVAL_VLLM_GPU_MEMORY_UTILIZATION=auto` is useful when training
processes still occupy device memory.

## Plotting

After evaluations have produced history rows:

```bash
python ablation/plots/plot_ablation.py \
  --input-root ablation/outputs --output-dir ablation/figures \
  --benchmark MATH-500 --metric accuracy

python ablation/plots/plot_hparam.py \
  --parameter epsilon --input-root ablation/outputs \
  --output-dir ablation/figures --benchmark MATH-500

python ablation/plots/plot_hparam.py \
  --parameter top_k --input-root ablation/outputs \
  --output-dir ablation/figures --aggregate auc

python ablation/plots/plot_diagnostics.py \
  --input-root ablation/outputs --output-dir ablation/figures
```

Each plot command writes both PNG and PDF.  The standard BellmanOPD progress
plotter can still be used directly on a particular run output with
`bash scripts/plot_training_progress.sh`.

## Scope of the core edit

Only two production files are touched.  `CMTSelector` accepts an optional
`ablation_arm` (default `canonical`) and selects one of its already-computed
`gain`, `successor_excess`, or canonical `learning_value` tensors; `trainer.py`
passes `selector.cmt_ablation_arm`.  With the default config, no numerical or
training behavior changes.  All ablation scripts, configuration, plots, and
tests live under this directory.
