#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

CONFIG="${1:-}"
GPU="${2:-${GPU:-0}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DRY_RUN="${DRY_RUN:-0}"

if [[ -z "${CONFIG}" ]]; then
  echo "Usage: bash scripts/train_one_setting.sh <config.py> [gpu]" >&2
  echo "Example: bash scripts/train_one_setting.sh configs/best_ckpt/AutoSSVH_activitynet_16bit_BPMS_PAHVL_FNSC.py 0" >&2
  exit 2
fi

if [[ ! -f "${CONFIG}" ]]; then
  echo "Missing config: ${CONFIG}" >&2
  exit 1
fi

cmd=("${PYTHON_BIN}" train.py --config "${CONFIG}" --gpu "${GPU}")

if [[ "${DRY_RUN}" == "1" ]]; then
  printf 'DRY_RUN:'
  printf ' %q' "${cmd[@]}"
  printf '\n'
else
  "${cmd[@]}"
fi
