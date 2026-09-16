#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ================= USER CONFIG: JTA-OPD =====================
export STUDENT_MODEL="${STUDENT_MODEL:-${STUDENT_MODEL_PATH:-nlp/tungdd11/stable-on-policy-distillation/OPD/model/Qwen3-1.7B-Base}}"
export TEACHER_MODEL="${TEACHER_MODEL:-${TEACHER_MODEL_PATH:-models/Qwen3-8B}}"
export TRAIN_DATASET="${TRAIN_DATASET:-competition_math}"
case "${TRAIN_DATASET,,}" in
  dapo_math|dapo-math|dapo)
    _DEFAULT_TRAIN_DATA="nlp/minhpn19/data/DAPO-Math-17k-Processed"
    _DEFAULT_PROMPT_KEY="prompt"
    _DEFAULT_TRAIN_SPLIT="all"
    ;;
  competition_math|competition-math|math)
    _DEFAULT_TRAIN_DATA="nlp/minhpn19/data/competition_math/data/train-00000-of-00001.parquet"
    _DEFAULT_PROMPT_KEY="problem"
    _DEFAULT_TRAIN_SPLIT="null"
    ;;
  custom)
    if [[ -z "${TRAIN_DATA_PATH:-}" || -z "${TRAIN_PROMPT_KEY:-}" ]]; then
      echo "TRAIN_DATASET=custom requires TRAIN_DATA_PATH and TRAIN_PROMPT_KEY" >&2
      exit 2
    fi
    _DEFAULT_TRAIN_DATA="${TRAIN_DATA_PATH}"
    _DEFAULT_PROMPT_KEY="${TRAIN_PROMPT_KEY}"
    _DEFAULT_TRAIN_SPLIT="${TRAIN_DATA_SPLIT:-null}"
    ;;
  *)
    echo "Unknown TRAIN_DATASET=${TRAIN_DATASET}; use competition_math, dapo_math, or custom" >&2
    exit 2
    ;;
esac
export TRAIN_DATA="${TRAIN_DATA:-${TRAIN_DATA_PATH:-${_DEFAULT_TRAIN_DATA}}}"
export PROMPT_KEY="${PROMPT_KEY:-${TRAIN_PROMPT_KEY:-${_DEFAULT_PROMPT_KEY}}}"
export TRAIN_DATA_SPLIT="${TRAIN_DATA_SPLIT:-${_DEFAULT_TRAIN_SPLIT}}"
export STUDENT_MODEL_PATH="${STUDENT_MODEL}"
export TEACHER_MODEL_PATH="${TEACHER_MODEL}"
export TRAIN_DATA_PATH="${TRAIN_DATA}"
export TRAIN_PROMPT_KEY="${PROMPT_KEY}"
export RUN_NAME="${RUN_NAME:-${JTA_RUN_NAME:-jta_$(date +%Y%m%d_%H%M%S_%N)}}"
export JTA_RUN_NAME="${RUN_NAME}"
export STORAGE_ROOT="${STORAGE_ROOT:-/workspace/storage-shared}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export SEED="${SEED:-42}"
export ROLLOUT_SEED="${ROLLOUT_SEED:-42}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}"
export BATCH_SIZE="${BATCH_SIZE:-${GLOBAL_BATCH_SIZE}}"
export NUM_RESPONSES="${NUM_RESPONSES:-1}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-16}"
export MICRO_BATCH_SIZE_PER_GPU="${MICRO_BATCH_SIZE_PER_GPU:-8}"
export NUM_EPOCHS="${NUM_EPOCHS:-1}"
export MAX_STEPS="${MAX_STEPS:--1}"
export LR="${LR:-${LEARNING_RATE:-1.0e-6}}"
export MAX_PROMPT_LEN="${MAX_PROMPT_LENGTH:-${MAX_PROMPT_LEN:-1024}}"
export OVERLONG_PROMPT_POLICY="${OVERLONG_PROMPT_POLICY:-filter}"
export MAX_RESPONSE_LEN="${MAX_RESPONSE_LENGTH:-${MAX_RESPONSE_LEN:-4096}}"
export TOP_K="${TOP_K:-16}"
export ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1.0}"
export ROLLOUT_TOP_P="${ROLLOUT_TOP_P:-1.0}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-50}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-50}"
export LOG_INTERVAL="${LOG_INTERVAL:-1}"
export ROLLOUT_BACKEND="${ROLLOUT_BACKEND:-vllm}"
export TRAIN_EVAL_ENABLED="${TRAIN_EVAL_ENABLED:-true}"
export TRAIN_EVAL_BACKEND="${TRAIN_EVAL_BACKEND:-vllm}"
export TRAIN_EVAL_NUM_RESPONSES="${TRAIN_EVAL_NUM_RESPONSES:-8}"
export TRAIN_EVAL_MAX_NEW_TOKENS="${TRAIN_EVAL_MAX_NEW_TOKENS:-4096}"
export DISTRIBUTED_STRATEGY="${DISTRIBUTED_STRATEGY:-fsdp}"
export GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-true}"
export FSDP_TEACHER_CPU_OFFLOAD="${FSDP_TEACHER_CPU_OFFLOAD:-false}"
export FSDP_USE_NO_SYNC="${FSDP_USE_NO_SYNC:-false}"
export JTA_EPSILON="${JTA_EPSILON:-0.1}"
export JTA_SKETCH_DIM="${JTA_SKETCH_DIM:-512}"
export JTA_SKETCH_SEED="${JTA_SKETCH_SEED:-42}"
export JTA_HIDDEN_SKETCH_SEED="${JTA_HIDDEN_SKETCH_SEED:-1729}"
export JTA_EMBEDDING_BACKEND="${JTA_EMBEDDING_BACKEND:-topk_context_tensorsketch}"
export JTA_FW_MAX_ITERATIONS="${JTA_FW_MAX_ITERATIONS:-20}"
export JTA_FW_GAP_TOLERANCE="${JTA_FW_GAP_TOLERANCE:-1.0e-5}"
export JTA_KL_BISECTION_ITERATIONS="${JTA_KL_BISECTION_ITERATIONS:-50}"
export JTA_CONSTRAINT_TOLERANCE="${JTA_CONSTRAINT_TOLERANCE:-1.0e-6}"
export JTA_TOKEN_CHUNK_SIZE="${JTA_TOKEN_CHUNK_SIZE:-4096}"
export RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-${RESUME:-}}"

source "${SCRIPT_DIR}/common_b200.sh"
export B200_METHOD="jta"
print_asset_selection
resolve_run_paths
if [[ -n "${RESUME_FROM_CHECKPOINT}" && "${RESUME_FROM_CHECKPOINT}" != "auto" && -z "${OUTPUT_DIR:-}" ]]; then
  OUTPUT_DIR="$(dirname -- "${RESUME_FROM_CHECKPOINT}")"
else
  OUTPUT_DIR="${OUTPUT_DIR:-${JTA_RUN_OUTPUT}}"
fi
build_training_args "${OUTPUT_DIR}"
echo "JTA-OPD run: ${RUN_NAME}"
echo "JTA-OPD output: ${OUTPUT_DIR}"
echo "JTA-OPD settings: backend=${JTA_EMBEDDING_BACKEND}, epsilon=${JTA_EPSILON}, sketch_dim=${JTA_SKETCH_DIM}, vocab_seed=${JTA_SKETCH_SEED}, hidden_seed=${JTA_HIDDEN_SKETCH_SEED}, responses=${NUM_RESPONSES}, ppo_batch=${PPO_MINI_BATCH_SIZE}"
if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
  echo "Resume checkpoint: ${RESUME_FROM_CHECKPOINT}"
fi
cd "${REPO_DIR}"
run_training_cli train \
  --config "${JTA_CONFIG}" \
  "${COMMON_TRAIN_ARGS[@]}" \
  "$@"
echo "JTA-OPD completed. Keep this identifier: JTA_RUN_NAME=${RUN_NAME}"
