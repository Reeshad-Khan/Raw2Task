#!/usr/bin/env bash
set -euo pipefail

# End-to-end automation:
#   1. Wait for the currently running fast per-task workflow, if present.
#   2. Otherwise launch the fast per-task workflow.
#   3. Run unified-only fast multi-task feasibility.
#   4. Run the full workflow for per-task + unified experiments.

GPUS="${GPUS:-${GPU_ID:-0,1}}"
MAX_BATCH="${MAX_BATCH:-16}"
PYTHON_BIN="${PYTHON_BIN:-/home/rk010/.conda/envs/raw2task/bin/python}"
KITTI_ROOT="${KITTI_ROOT:-/home/rk010/Desktop/Research/NurIPS/KITTI-360}"
OCCUPANCY_OUT_ROOT="${OCCUPANCY_OUT_ROOT:-data_external/kitti360_occupancy}"
FAST_OCCUPANCY_MAX_SAMPLES="${FAST_OCCUPANCY_MAX_SAMPLES:-20}"
FAST_OCCUPANCY_EPOCHS="${FAST_OCCUPANCY_EPOCHS:-2}"
FULL_OCCUPANCY_MAX_SAMPLES="${FULL_OCCUPANCY_MAX_SAMPLES:-0}"
FULL_OCCUPANCY_EPOCHS="${FULL_OCCUPANCY_EPOCHS:-0}"
FRESH_FAST=0
FRESH_UNIFIED=1
FRESH_FULL=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus|--gpu)
      GPUS="$2"
      shift 2
      ;;
    --max-batch)
      MAX_BATCH="$2"
      shift 2
      ;;
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --kitti-root)
      KITTI_ROOT="$2"
      shift 2
      ;;
    --occupancy-out-root)
      OCCUPANCY_OUT_ROOT="$2"
      shift 2
      ;;
    --fast-occupancy-max-samples)
      FAST_OCCUPANCY_MAX_SAMPLES="$2"
      shift 2
      ;;
    --fast-occupancy-epochs)
      FAST_OCCUPANCY_EPOCHS="$2"
      shift 2
      ;;
    --full-occupancy-max-samples)
      FULL_OCCUPANCY_MAX_SAMPLES="$2"
      shift 2
      ;;
    --full-occupancy-epochs)
      FULL_OCCUPANCY_EPOCHS="$2"
      shift 2
      ;;
    --fresh-fast)
      FRESH_FAST=1
      shift
      ;;
    --resume-unified)
      FRESH_UNIFIED=0
      shift
      ;;
    --resume-full)
      FRESH_FULL=0
      shift
      ;;
    --help|-h)
      cat <<'EOF'
Usage: deeplens/projects/raw2task/scripts/run_fast_then_unified_then_full_all.sh [options]

Options:
  --gpus <csv>                         GPUs for concurrent jobs (default: 0,1)
  --max-batch <n>                      Max probed batch size (default: 16)
  --python <path>                      Python executable
  --kitti-root <path>                  KITTI-360 root
  --occupancy-out-root <path>          Observed voxel artifact root
  --fast-occupancy-max-samples <n>     Fast voxel build cap (default: 20)
  --fast-occupancy-epochs <n>          Fast per-task occupancy epochs (default: 2)
  --full-occupancy-max-samples <n>     Full voxel build cap, 0 = all (default: 0)
  --full-occupancy-epochs <n>          Full per-task occupancy epoch override, 0 = config default
  --fresh-fast                         Force a new fast per-task run instead of waiting/reusing
  --resume-unified                     Do not delete unified runs before unified-fast/full
  --resume-full                        Do not delete full per-task outputs before full run
EOF
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd -P)"
cd "${REPO_ROOT}"

mkdir -p runs
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export MPLBACKEND="${MPLBACKEND:-Agg}"
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export PYTHON_BIN

wait_for_pids() {
  local label="$1"
  shift
  local pids=("$@")
  if [[ ${#pids[@]} -eq 0 ]]; then
    return 0
  fi
  echo "[sequence] waiting for ${label}: ${pids[*]}"
  while true; do
    local alive=0
    for pid in "${pids[@]}"; do
      if kill -0 "${pid}" 2>/dev/null; then
        alive=1
      fi
    done
    if [[ ${alive} -eq 0 ]]; then
      break
    fi
    sleep 60
  done
}

find_current_fast_pids() {
  pgrep -f "run_complete_claim_experiments.sh .*--mode fast" || true
}

run_complete() {
  local log="$1"
  shift
  deeplens/projects/raw2task/scripts/run_complete_claim_experiments.sh "$@" \
    2>&1 | tee "${log}"
}

mapfile -t current_fast < <(find_current_fast_pids)
if [[ ${FRESH_FAST} -eq 0 && ${#current_fast[@]} -gt 0 ]]; then
  wait_for_pids "current fast per-task workflow" "${current_fast[@]}"
else
  fast_args=(--mode fast --gpus "${GPUS}" --max-batch "${MAX_BATCH}" --occupancy-max-samples "${FAST_OCCUPANCY_MAX_SAMPLES}" --occupancy-epochs "${FAST_OCCUPANCY_EPOCHS}" --python "${PYTHON_BIN}" --kitti-root "${KITTI_ROOT}" --occupancy-out-root "${OCCUPANCY_OUT_ROOT}")
  if [[ ${FRESH_FAST} -eq 1 ]]; then
    fast_args+=(--fresh)
  fi
  echo "[sequence] running fast per-task workflow"
  run_complete runs/sequence_fast_per_task_live.log "${fast_args[@]}"
fi

unified_fast_args=(--unified-only --mode fast --gpus "${GPUS}" --max-batch "${MAX_BATCH}" --occupancy-max-samples "${FAST_OCCUPANCY_MAX_SAMPLES}" --python "${PYTHON_BIN}" --kitti-root "${KITTI_ROOT}" --occupancy-out-root "${OCCUPANCY_OUT_ROOT}")
if [[ ${FRESH_UNIFIED} -eq 1 ]]; then
  unified_fast_args+=(--fresh)
fi
echo "[sequence] running unified-only fast workflow"
run_complete runs/sequence_unified_fast_live.log "${unified_fast_args[@]}"

full_args=(--mode full --with-unified --gpus "${GPUS}" --max-batch "${MAX_BATCH}" --occupancy-max-samples "${FULL_OCCUPANCY_MAX_SAMPLES}" --occupancy-epochs "${FULL_OCCUPANCY_EPOCHS}" --python "${PYTHON_BIN}" --kitti-root "${KITTI_ROOT}" --occupancy-out-root "${OCCUPANCY_OUT_ROOT}")
if [[ ${FRESH_FULL} -eq 1 ]]; then
  full_args+=(--fresh)
fi
echo "[sequence] running full workflow for all per-task and unified experiments"
run_complete runs/sequence_full_all_live.log "${full_args[@]}"

echo "[sequence] complete"

