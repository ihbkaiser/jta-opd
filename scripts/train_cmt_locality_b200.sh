#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Locality diagnostics are a CMT-scored, uniformly allocated measurement run.
export CMT_CONFIG="${CMT_CONFIG:-${SCRIPT_DIR}/../configs/qwen3_b200_cmt_locality.yaml}"
export TEACHER_MODEL="${TEACHER_MODEL:-${TEACHER_MODEL_PATH:-models/Qwen3-4B}}"
export STUDENT_MODEL="${STUDENT_MODEL:-${STUDENT_MODEL_PATH:-nlp/tungdd11/stable-on-policy-distillation/OPD/model/Qwen3-1.7B-Base}}"
export TRAIN_DATASET="${TRAIN_DATASET:-competition_math}"
export NUM_EPOCHS="${NUM_EPOCHS:-${EPOCHS:-3}}"
export NUM_RESPONSES=1
export GLOBAL_BATCH_SIZE="${BATCH_SIZE:-${GLOBAL_BATCH_SIZE:-64}}"
export BATCH_SIZE="${GLOBAL_BATCH_SIZE}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-${GLOBAL_BATCH_SIZE}}"
export RUN_NAME="${RUN_NAME:-cmt_locality_$(date +%Y%m%d_%H%M%S_%N)}"
export OUTPUT_DIR="${OUTPUT_DIR:-outputs/${RUN_NAME}/cmt_locality}"
export TOKEN_SCORE_INTERVAL=1

export LOCALITY_PROBE_PROMPTS="${LOCALITY_PROBE_PROMPTS:-32}"
export LOCALITY_PROBE_MAX_NEW_TOKENS="${LOCALITY_PROBE_MAX_NEW_TOKENS:-2048}"
export LOCALITY_FUTURE_HORIZONS="${LOCALITY_FUTURE_HORIZONS:-[8,16,32]}"
export LOCALITY_TOKEN_SAMPLE_SIZE="${LOCALITY_TOKEN_SAMPLE_SIZE:-2048}"
export LOCALITY_MATCHED_PAIRS="${LOCALITY_MATCHED_PAIRS:-8}"

exec bash "${SCRIPT_DIR}/train_cmt_b200.sh" \
  --set "analysis.locality_probe.other_id.num_prompts=${LOCALITY_PROBE_PROMPTS}" \
  --set "analysis.locality_probe.other_id.max_new_tokens=${LOCALITY_PROBE_MAX_NEW_TOKENS}" \
  --set "analysis.locality_probe.future_horizons=${LOCALITY_FUTURE_HORIZONS}" \
  --set "analysis.locality_probe.token_sample_size=${LOCALITY_TOKEN_SAMPLE_SIZE}" \
  --set "analysis.locality_probe.matched_pairs_per_step=${LOCALITY_MATCHED_PAIRS}" \
  "$@"
