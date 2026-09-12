#!/usr/bin/env bash
set -euo pipefail

# ===================== PLOT CONFIG =========================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export INPUT_ROOT="${INPUT_ROOT:-${SCRIPT_DIR}/../outputs}"
export OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/../figures}"
# modes: arms, epsilon, top_k, lr, diagnostics
export PLOT_MODE="${PLOT_MODE:-arms}"
export BENCHMARK="${BENCHMARK:-MATH-500}"
export METRIC="${METRIC:-accuracy}"
export AGGREGATE="${AGGREGATE:-final}"
# Either set RUN_NAMES as a space-separated list, or fill these slots below.
export RUN_NAMES="${RUN_NAMES:-}"
export RUN_NAME_1="${RUN_NAME_1:-}"
export RUN_NAME_2="${RUN_NAME_2:-}"
export RUN_NAME_3="${RUN_NAME_3:-}"
export RUN_NAME_4="${RUN_NAME_4:-}"
# ============================================================

if [[ -z "${RUN_NAMES}" ]]; then
  RUN_NAMES="${RUN_NAME_1} ${RUN_NAME_2} ${RUN_NAME_3} ${RUN_NAME_4}"
fi
RUN_ARGS=()
for name in ${RUN_NAMES}; do
  [[ -z "${name}" ]] || RUN_ARGS+=(--run-name "${name}")
done

case "${PLOT_MODE}" in
  arms|ablation)
    COMMAND=("${PYTHON_BIN}" "${SCRIPT_DIR}/../plots/plot_ablation.py"
      --input-root "${INPUT_ROOT}" --output-dir "${OUTPUT_DIR}"
      --benchmark "${BENCHMARK}" --metric "${METRIC}")
    ;;
  epsilon|top_k|lr)
    COMMAND=("${PYTHON_BIN}" "${SCRIPT_DIR}/../plots/plot_hparam.py"
      --parameter "${PLOT_MODE}" --input-root "${INPUT_ROOT}"
      --output-dir "${OUTPUT_DIR}" --benchmark "${BENCHMARK}"
      --metric "${METRIC}" --aggregate "${AGGREGATE}")
    ;;
  diagnostics)
    COMMAND=("${PYTHON_BIN}" "${SCRIPT_DIR}/../plots/plot_diagnostics.py"
      --input-root "${INPUT_ROOT}" --output-dir "${OUTPUT_DIR}")
    ;;
  *)
    echo "PLOT_MODE must be arms, epsilon, top_k, lr, or diagnostics" >&2
    exit 2
    ;;
esac

cd "${REPO_DIR}"
echo "Plot mode: ${PLOT_MODE}"
echo "Input: ${INPUT_ROOT}"
echo "Output: ${OUTPUT_DIR}"
if (( ${#RUN_ARGS[@]} )); then
  echo "Selected runs: ${RUN_NAMES}"
fi
exec "${COMMAND[@]}" "${RUN_ARGS[@]}"
