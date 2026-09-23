#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Every-step pilot for the on-policy full-vocabulary trajectory-KL probe.
# Callers may still override any value on the command line/environment.
export ONE_STEP_KL_PROBE_INTERVAL="${ONE_STEP_KL_PROBE_INTERVAL:-1}"
export ONE_STEP_KL_PROBE_SUBSET_SIZE="${ONE_STEP_KL_PROBE_SUBSET_SIZE:-32}"
export ONE_STEP_KL_PROBE_NUM_ROLLOUTS_PER_PROBLEM="${ONE_STEP_KL_PROBE_NUM_ROLLOUTS_PER_PROBLEM:-1}"
export ONE_STEP_KL_PROBE_HORIZON="${ONE_STEP_KL_PROBE_HORIZON:-32}"

exec bash "${SCRIPT_DIR}/train_cmt_b200.sh" "$@"
