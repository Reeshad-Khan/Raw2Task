#!/usr/bin/env bash
set -euo pipefail

GPUS="${GPUS:-${GPU_ID:-0,1}}"
MAX_BATCH="${MAX_BATCH:-16}"
SEEDS="${SEEDS:-0}"
PYTHON_BIN="${PYTHON_BIN:-/home/rk010/.conda/envs/raw2task/bin/python}"
ROBUSTNESS_BATCHES="${ROBUSTNESS_BATCHES:-80}"
FRESH=0
STOP_EXISTING=0
ALLOW_CONCURRENT=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus|--gpu)
      GPUS="$2"; shift 2 ;;
    --max-batch)
      MAX_BATCH="$2"; shift 2 ;;
    --seeds)
      SEEDS="$2"; shift 2 ;;
    --robustness-batches)
      ROBUSTNESS_BATCHES="$2"; shift 2 ;;
    --python)
      PYTHON_BIN="$2"; shift 2 ;;
    --fresh)
      FRESH=1; shift ;;
    --stop-existing)
      STOP_EXISTING=1; shift ;;
    --allow-concurrent)
      ALLOW_CONCURRENT=1; shift ;;
    --help|-h)
      cat <<'EOF'
Usage: deeplens/projects/raw2task/scripts/run_cross_dataset_comparison_fast.sh [options]

Runs matched fast modern-backbone experiments on KITTI-360 and Cityscapes:
  newest co-design hypotheses first (SegFormer-B1, DDR/PID-style, task-token),
  then continuity co-design rows, RGB references, and fixed camera rows.

Options:
  --fresh            Delete runs/dataset_compare_fast first
  --stop-existing    Stop active raw2task jobs first
  --allow-concurrent Allow running beside existing raw2task jobs
  --gpus <csv>       First GPU for KITTI, second GPU for Cityscapes, default 0,1
  --max-batch <n>    Max batch size for fixed preflight probe
  --seeds <csv>      Seed override, default 0
  --python <path>    Python executable
  --robustness-batches <n>
                     Validation batches for fast robustness/viewpoint sweeps
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
export PYTHON_BIN
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

KITTI_MATRIX="deeplens/projects/raw2task/configs/dataset_compare_kitti360_fast.yaml"
CITY_MATRIX="deeplens/projects/raw2task/configs/dataset_compare_cityscapes_fast.yaml"
KITTI_PLAN="runs/dataset_compare_fast/batch_size_plan_kitti360.yaml"
CITY_PLAN="runs/dataset_compare_fast/batch_size_plan_cityscapes.yaml"

first_gpu() {
  echo "${GPUS}" | cut -d, -f1 | xargs
}

second_gpu_or_first() {
  local second
  second="$(echo "${GPUS}" | cut -d, -f2 | xargs)"
  if [[ -n "${second}" && "${second}" != "$(first_gpu)" ]]; then
    echo "${second}"
  else
    first_gpu
  fi
}

find_active_jobs() {
  ps -eo pid=,ppid=,stat=,etime=,cmd= \
    | grep -E "deeplens\.projects\.raw2task\.(train_extended|run_paper_experiments|train_unified_multitask)|run_pair_first_all_experiments|run_complete_claim_experiments|run_pivot_constrained_experiments" \
    | grep -v grep \
    | grep -v "run_cross_dataset_comparison_fast" || true
}

ACTIVE_JOBS="$(find_active_jobs)"
if [[ -n "${ACTIVE_JOBS}" ]]; then
  if [[ ${STOP_EXISTING} -eq 1 ]]; then
    echo "[dataset-compare] stopping active raw2task jobs"
    echo "${ACTIVE_JOBS}"
    echo "${ACTIVE_JOBS}" | awk '{print $1}' | xargs -r kill
    sleep 5
    ACTIVE_JOBS="$(find_active_jobs)"
    if [[ -n "${ACTIVE_JOBS}" ]]; then
      echo "[dataset-compare] some jobs did not stop after SIGTERM; sending SIGKILL"
      echo "${ACTIVE_JOBS}" | awk '{print $1}' | xargs -r kill -9
      sleep 2
    fi
  elif [[ ${ALLOW_CONCURRENT} -eq 0 ]]; then
    echo "[dataset-compare] active raw2task jobs are still running; refusing to start concurrently." >&2
    echo "${ACTIVE_JOBS}" >&2
    echo "[dataset-compare] rerun with --stop-existing or --allow-concurrent." >&2
    exit 3
  fi
fi

if [[ ${FRESH} -eq 1 ]]; then
  echo "[dataset-compare] fresh requested: deleting runs/dataset_compare_fast"
  rm -rf runs/dataset_compare_fast
fi
mkdir -p runs/dataset_compare_fast

probe_and_run() {
  local label="$1"
  local matrix="$2"
  local plan="$3"
  local gpu="$4"
  local log="$5"
  local robust_experiments=""

  echo "[dataset-compare] ${label}: probing fixed batch sizes on gpu=${gpu}"
  deeplens/projects/raw2task/scripts/probe_batch_size.sh \
    --matrix "${matrix}" \
    --seeds "${SEEDS}" \
    --gpu "${gpu}" \
    --max-batch "${MAX_BATCH}" \
    --out "${plan}" \
    2>&1 | tee "runs/dataset_compare_fast/${label}_preflight.log"

  echo "[dataset-compare] ${label}: training on gpu=${gpu}"
  if [[ "${label}" == "kitti360" ]]; then
    robust_experiments="kitti360_clean_codesign_segformer_b0,kitti360_clean_rgb_segformer_b0,kitti360_clean_fixed_segformer_b0,kitti360_lowlight_codesign_segformer_b0,kitti360_lowlight_fixed_segformer_b0,kitti360_lowbit4_codesign_segformer_b0,kitti360_lowbit4_fixed_segformer_b0"
  else
    robust_experiments="cityscapes_clean_codesign_segformer_b0,cityscapes_clean_rgb_segformer_b0,cityscapes_clean_fixed_segformer_b0,cityscapes_lowlight_codesign_segformer_b0,cityscapes_lowlight_fixed_segformer_b0,cityscapes_lowbit4_codesign_segformer_b0,cityscapes_lowbit4_fixed_segformer_b0"
  fi
  "${PYTHON_BIN}" -u -m deeplens.projects.raw2task.run_paper_experiments \
    --matrix "${matrix}" \
    --seeds "${SEEDS}" \
    --batch-plan "${plan}" \
    --gpus "${gpu}" \
    --max-parallel 1 \
    --order-policy new_idea_first \
    --skip-existing \
    --robustness-experiments "${robust_experiments}" \
    --max-robustness-batches "${ROBUSTNESS_BATCHES}" \
    2>&1 | tee "${log}"
}

KITTI_GPU="$(first_gpu)"
CITY_GPU="$(second_gpu_or_first)"
echo "[dataset-compare] schedule: KITTI-360 gpu=${KITTI_GPU}, Cityscapes gpu=${CITY_GPU}"

probe_and_run "kitti360" "${KITTI_MATRIX}" "${KITTI_PLAN}" "${KITTI_GPU}" "runs/dataset_compare_fast/kitti360_live.log" &
KITTI_PID=$!
probe_and_run "cityscapes" "${CITY_MATRIX}" "${CITY_PLAN}" "${CITY_GPU}" "runs/dataset_compare_fast/cityscapes_live.log" &
CITY_PID=$!

wait "${KITTI_PID}"
wait "${CITY_PID}"

"${PYTHON_BIN}" -u -m deeplens.projects.raw2task.summarize_dataset_comparison \
  --kitti-summary runs/dataset_compare_fast/kitti360/summary.csv \
  --city-summary runs/dataset_compare_fast/cityscapes/summary.csv \
  --out-dir runs/dataset_compare_fast/paper_tables

echo "[dataset-compare] complete"
echo "[dataset-compare] table: runs/dataset_compare_fast/paper_tables/cross_dataset_comparison.csv"
echo "[dataset-compare] plot : runs/dataset_compare_fast/paper_tables/cross_dataset_comparison_miou.png"
