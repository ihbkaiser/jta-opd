#!/usr/bin/env bash
set -euo pipefail
if (( $# == 0 )); then
  echo "Usage: OUTPUT_DIR=figures/locality $0 RUN_DIR [RUN_DIR ...]" >&2
  exit 2
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${REPO_DIR}/.venv/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="${PYTHON:-python3}"
fi
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/figures/locality}"
cd "${REPO_DIR}"
MPLBACKEND=Agg "${PYTHON_BIN}" -m b200_experiment.locality_plotting \
  --output-dir "${OUTPUT_DIR}" "$@"
