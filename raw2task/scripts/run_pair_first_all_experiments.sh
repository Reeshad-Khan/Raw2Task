#!/usr/bin/env bash
set -euo pipefail

# One-command paper workflow with the requested ordering:
#   FAST:
#     1. latest unified co-design task-token vs unified RGB baseline, concurrently
#     2. remaining unified baseline(s)
#     3. full fast per-task matrices: 2D, stress, 3D segmentation, occupancy
#   FULL:
#     4. latest unified co-design task-token vs unified RGB baseline, concurrently
#     5. remaining unified baseline(s)
#     6. full per-task matrices: 2D, stress, 3D segmentation, occupancy

GPUS="${GPUS:-${GPU_ID:-0,1}}"
MAX_BATCH="${MAX_BATCH:-16}"
PYTHON_BIN="${PYTHON_BIN:-/home/rk010/.conda/envs/raw2task/bin/python}"
KITTI_ROOT="${KITTI_ROOT:-/home/rk010/Desktop/Research/NurIPS/KITTI-360}"
OCCUPANCY_OUT_ROOT="${OCCUPANCY_OUT_ROOT:-data_external/kitti360_occupancy}"
FAST_OCCUPANCY_MAX_SAMPLES="${FAST_OCCUPANCY_MAX_SAMPLES:-400}"
FAST_OCCUPANCY_EPOCHS="${FAST_OCCUPANCY_EPOCHS:-2}"
FULL_OCCUPANCY_MAX_SAMPLES="${FULL_OCCUPANCY_MAX_SAMPLES:-0}"
FULL_OCCUPANCY_EPOCHS="${FULL_OCCUPANCY_EPOCHS:-0}"
FRESH=0
SKIP_FAST=0
SKIP_FULL=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus|--gpu)
      GPUS="$2"; shift 2 ;;
    --max-batch)
      MAX_BATCH="$2"; shift 2 ;;
    --python)
      PYTHON_BIN="$2"; shift 2 ;;
    --kitti-root)
      KITTI_ROOT="$2"; shift 2 ;;
    --occupancy-out-root)
      OCCUPANCY_OUT_ROOT="$2"; shift 2 ;;
    --fast-occupancy-max-samples)
      FAST_OCCUPANCY_MAX_SAMPLES="$2"; shift 2 ;;
    --fast-occupancy-epochs)
      FAST_OCCUPANCY_EPOCHS="$2"; shift 2 ;;
    --full-occupancy-max-samples)
      FULL_OCCUPANCY_MAX_SAMPLES="$2"; shift 2 ;;
    --full-occupancy-epochs)
      FULL_OCCUPANCY_EPOCHS="$2"; shift 2 ;;
    --fresh)
      FRESH=1; shift ;;
    --skip-fast)
      SKIP_FAST=1; shift ;;
    --skip-full)
      SKIP_FULL=1; shift ;;
    --help|-h)
      cat <<'EOF'
Usage: deeplens/projects/raw2task/scripts/run_pair_first_all_experiments.sh [options]

Options:
  --gpus <csv>                         Two GPUs, e.g. 0,1
  --max-batch <n>                      Max batch size for preflight probes
  --python <path>                      Python executable
  --kitti-root <path>                  KITTI-360 root
  --occupancy-out-root <path>          Observed voxel artifact root
  --fast-occupancy-max-samples <n>     Fast voxel cap
  --fast-occupancy-epochs <n>          Fast per-task occupancy epochs
  --full-occupancy-max-samples <n>     Full voxel cap, 0 = all
  --full-occupancy-epochs <n>          Full per-task occupancy epochs, 0 = config
  --fresh                              Delete generated run outputs first
  --skip-fast                          Run only full protocol
  --skip-full                          Run only fast protocol
EOF
      exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2 ;;
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

if [[ ${FRESH} -eq 1 ]]; then
  echo "[pair-first] fresh requested: wiping generated run outputs"
  rm -rf runs/*
  rm -f imgs/raw2task_*.gif
  mkdir -p runs
fi

run_complete() {
  local log="$1"; shift
  deeplens/projects/raw2task/scripts/run_complete_claim_experiments.sh "$@" 2>&1 | tee "${log}"
}

run_latest_pair() {
  local mode="$1"
  local occ_max="$2"
  echo "[pair-first] ${mode}: latest idea vs competitive RGB baseline"
  deeplens/projects/raw2task/scripts/run_latest_vs_competitive.sh \
    --mode "${mode}" \
    --gpus "${GPUS}" \
    --python "${PYTHON_BIN}" \
    --kitti-root "${KITTI_ROOT}" \
    --occupancy-out-root "${OCCUPANCY_OUT_ROOT}" \
    --occupancy-max-samples "${occ_max}" \
    2>&1 | tee "runs/pair_first_${mode}_latest_vs_rgb_live.log"
}

run_remaining_unified() {
  local mode="$1"
  local occ_max="$2"
  echo "[pair-first] ${mode}: remaining unified baseline(s)"
  deeplens/projects/raw2task/scripts/run_unified_multitask_fast.sh \
    --mode "${mode}" \
    --only fixed_camera \
    --gpus "${GPUS}" \
    --python "${PYTHON_BIN}" \
    --kitti-root "${KITTI_ROOT}" \
    --occupancy-out-root "${OCCUPANCY_OUT_ROOT}" \
    --occupancy-max-samples "${occ_max}" \
    2>&1 | tee "runs/pair_first_${mode}_remaining_unified_live.log"
}

run_per_task_all() {
  local mode="$1"
  local occ_max="$2"
  local occ_epochs="$3"
  local log="$4"
  echo "[pair-first] ${mode}: all per-task experiments"
  run_complete "${log}" \
    --mode "${mode}" \
    --gpus "${GPUS}" \
    --max-batch "${MAX_BATCH}" \
    --occupancy-max-samples "${occ_max}" \
    --occupancy-epochs "${occ_epochs}" \
    --python "${PYTHON_BIN}" \
    --kitti-root "${KITTI_ROOT}" \
    --occupancy-out-root "${OCCUPANCY_OUT_ROOT}"
}

if [[ ${SKIP_FAST} -eq 0 ]]; then
  run_latest_pair fast "${FAST_OCCUPANCY_MAX_SAMPLES}"
  run_remaining_unified fast "${FAST_OCCUPANCY_MAX_SAMPLES}"
  run_per_task_all fast "${FAST_OCCUPANCY_MAX_SAMPLES}" "${FAST_OCCUPANCY_EPOCHS}" "runs/pair_first_fast_per_task_all_live.log"
fi

if [[ ${SKIP_FULL} -eq 0 ]]; then
  run_latest_pair full "${FULL_OCCUPANCY_MAX_SAMPLES}"
  run_remaining_unified full "${FULL_OCCUPANCY_MAX_SAMPLES}"
  run_per_task_all full "${FULL_OCCUPANCY_MAX_SAMPLES}" "${FULL_OCCUPANCY_EPOCHS}" "runs/pair_first_full_per_task_all_live.log"
fi

"${PYTHON_BIN}" -m raw2task.analyze_unified_results \
  --runs-root runs/unified_multitask \
  --out-dir runs/unified_multitask/paper_tables || true

echo "[pair-first] complete"
