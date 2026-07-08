#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU="${GPU:-0}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT="${OUTPUT:-eval_results/best_ckpt_BPMS_PAHVL_FNSC_${RUN_ID}.csv}"
DATASETS="${DATASETS:-activitynet fcv hmdb ucf}"
BITS="${BITS:-16 32 64}"
DRY_RUN="${DRY_RUN:-0}"

mkdir -p eval_results run_logs/eval_best_ckpt_BPMS_PAHVL_FNSC

contains_word() {
  local list="$1"
  local item="$2"
  [[ " ${list} " == *" ${item} "* ]]
}

run_one() {
  local dataset="$1"
  local bit="$2"
  local config="$3"
  local checkpoint="$4"

  if ! contains_word "${DATASETS}" "${dataset}"; then
    return 0
  fi
  if ! contains_word "${BITS}" "${bit}"; then
    return 0
  fi

  if [[ ! -f "${config}" ]]; then
    echo "Missing config: ${config}" >&2
    return 1
  fi
  if [[ ! -f "${checkpoint}" ]]; then
    echo "Missing checkpoint: ${checkpoint}" >&2
    return 1
  fi

  local cmd=(
    "${PYTHON_BIN}" scripts/eval_checkpoint_maps.py
    --config "${config}"
    --checkpoint "${checkpoint}"
    --gpu "${GPU}"
    --output "${OUTPUT}"
  )

  echo
  echo "[${dataset} ${bit}-bit] ${checkpoint}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf 'DRY_RUN:'
    printf ' %q' "${cmd[@]}"
    printf '\n'
  else
    "${cmd[@]}"
  fi
}

echo "Evaluating BPMS + PA-HVL + FNSC best checkpoints"
echo "repo: ${REPO_ROOT}"
echo "gpu: ${GPU}"
echo "datasets: ${DATASETS}"
echo "bits: ${BITS}"
echo "output: ${OUTPUT}"

run_one activitynet 16 configs/best_ckpt/AutoSSVH_activitynet_16bit_BPMS_PAHVL_FNSC.py best_ckpt/activitynet_16bit_r06_p045_metric_majority.pth
run_one activitynet 32 configs/best_ckpt/AutoSSVH_activitynet_32bit_BPMS_PAHVL_FNSC.py best_ckpt/activitynet_32bit_r07_p035_metric_majority.pth
run_one activitynet 64 configs/best_ckpt/AutoSSVH_activitynet_64bit_BPMS_PAHVL_FNSC.py best_ckpt/activitynet_64bit_r03_p025_metric_majority.pth

run_one fcv 16 configs/best_ckpt/AutoSSVH_fcv_16bit_BPMS_PAHVL_FNSC.py best_ckpt/fcv_16bit_r05_p035_metric_majority.pth
run_one fcv 32 configs/best_ckpt/AutoSSVH_fcv_32bit_BPMS_PAHVL_FNSC.py best_ckpt/fcv_32bit_r05_p035_metric_majority.pth
run_one fcv 64 configs/best_ckpt/AutoSSVH_fcv_64bit_BPMS_PAHVL_FNSC.py best_ckpt/fcv_64bit_r05_p035_metric_majority.pth

run_one hmdb 16 configs/best_ckpt/AutoSSVH_hmdb_16bit_BPMS_PAHVL_FNSC.py best_ckpt/hmdb_16bit_r06_p045_metric_majority.pth
run_one hmdb 32 configs/best_ckpt/AutoSSVH_hmdb_32bit_BPMS_PAHVL_FNSC.py best_ckpt/hmdb_32bit_r05_p045_metric_majority.pth
run_one hmdb 64 configs/best_ckpt/AutoSSVH_hmdb_64bit_BPMS_PAHVL_FNSC.py best_ckpt/hmdb_64bit_r05_p055_metric_majority.pth

run_one ucf 16 configs/best_ckpt/AutoSSVH_ucf_16bit_BPMS_PAHVL_FNSC.py best_ckpt/ucf_16bit_r06_p055_metric_majority.pth
run_one ucf 32 configs/best_ckpt/AutoSSVH_ucf_32bit_BPMS_PAHVL_FNSC.py best_ckpt/ucf_32bit_r06_p025_metric_majority.pth
run_one ucf 64 configs/best_ckpt/AutoSSVH_ucf_64bit_BPMS_PAHVL_FNSC.py best_ckpt/ucf_64bit_r06_p055_metric_majority.pth

echo
if [[ "${DRY_RUN}" == "1" ]]; then
  echo "Dry run complete. No inference/evaluation was started."
else
  echo "Done. Results written to: ${OUTPUT}"
fi
