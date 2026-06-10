#!/usr/bin/env bash
set -euo pipefail

# Revised paper workflow:
#   1. Fast constrained-sensing matrix on KITTI-360:
#      co-design/fixed/RGB paired controls under clean, low-light, and low-bit.
#   2. Matching fast constrained-sensing matrix on Cityscapes.
#   3. Optional full constrained stress matrix after fast evidence looks viable.

MODE="${MODE:-fast}"
GPUS="${GPUS:-${GPU_ID:-0,1}}"
MAX_BATCH="${MAX_BATCH:-16}"
SEEDS="${SEEDS:-0}"
PYTHON_BIN="${PYTHON_BIN:-/home/rk010/.conda/envs/raw2task/bin/python}"
FRESH=0
RUN_ACCURACY=0
RUN_CONSTRAINED=1
RUN_CITYSCAPES=1
RUN_FULL_STRESS=0
STOP_EXISTING=0
ALLOW_CONCURRENT=0
SPLIT_DATASET_GPUS=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="$2"; shift 2 ;;
    --gpus|--gpu)
      GPUS="$2"; shift 2 ;;
    --max-batch)
      MAX_BATCH="$2"; shift 2 ;;
    --seeds)
      SEEDS="$2"; shift 2 ;;
    --python)
      PYTHON_BIN="$2"; shift 2 ;;
    --fresh)
      FRESH=1; shift ;;
    --accuracy-only)
      RUN_ACCURACY=1; RUN_CONSTRAINED=0; RUN_CITYSCAPES=0; RUN_FULL_STRESS=0; shift ;;
    --constrained-only)
      RUN_ACCURACY=0; RUN_CONSTRAINED=1; RUN_CITYSCAPES=0; RUN_FULL_STRESS=0; shift ;;
    --cityscapes-only)
      RUN_ACCURACY=0; RUN_CONSTRAINED=0; RUN_CITYSCAPES=1; RUN_FULL_STRESS=0; shift ;;
    --with-cityscapes)
      RUN_CITYSCAPES=1; shift ;;
    --skip-cityscapes)
      RUN_CITYSCAPES=0; shift ;;
    --with-full-stress)
      RUN_FULL_STRESS=1; shift ;;
    --stop-existing)
      STOP_EXISTING=1; shift ;;
    --allow-concurrent)
      ALLOW_CONCURRENT=1; shift ;;
    --split-dataset-gpus)
      SPLIT_DATASET_GPUS=1; shift ;;
    --shared-gpus)
      SPLIT_DATASET_GPUS=0; shift ;;
    --help|-h)
      cat <<'EOF'
Usage: deeplens/projects/raw2task/scripts/run_pivot_constrained_experiments.sh [options]

Options:
  --fresh                 Delete revised fast/full outputs first
  --gpus <csv>            GPUs for probing and training, e.g. 0,1
  --max-batch <n>         Highest batch size for preflight probe
  --seeds <csv>           Seed override, default 0
  --python <path>         Python executable
  --accuracy-only         Run only the legacy accuracy-recovery fast matrix
  --constrained-only      Run only constrained-sensing fast matrix
  --cityscapes-only       Run only Cityscapes fast matrix
  --with-cityscapes       Run Cityscapes fast matrix after KITTI-360 fast matrices
  --skip-cityscapes       Skip Cityscapes fast matrix
  --with-full-stress      After fast matrices, run full sensor_stress_matrix.yaml
  --stop-existing         Stop active raw2task training/experiment jobs first
  --allow-concurrent      Do not block when old raw2task jobs are still active
  --split-dataset-gpus    Run KITTI on first GPU and Cityscapes on second GPU
  --shared-gpus           Use all listed GPUs for each matrix sequentially
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

ACC_MATRIX="deeplens/projects/raw2task/configs/accuracy_recovery_matrix_fast.yaml"
CLAIM_MATRIX="deeplens/projects/raw2task/configs/constrained_codesign_matrix_fast.yaml"
CITY_MATRIX="deeplens/projects/raw2task/configs/cityscapes_codesign_matrix_fast.yaml"
FULL_STRESS_MATRIX="deeplens/projects/raw2task/configs/sensor_stress_matrix.yaml"

find_active_jobs() {
  ps -eo pid=,ppid=,stat=,etime=,cmd= \
    | grep -E "deeplens\.projects\.raw2task\.(train_extended|run_paper_experiments|train_unified_multitask)|run_pair_first_all_experiments|run_complete_claim_experiments" \
    | grep -v grep \
    | grep -v "run_pivot_constrained_experiments" || true
}

ACTIVE_JOBS="$(find_active_jobs)"
if [[ -n "${ACTIVE_JOBS}" ]]; then
  if [[ ${STOP_EXISTING} -eq 1 ]]; then
    echo "[pivot] stopping active raw2task jobs before starting revised run"
    echo "${ACTIVE_JOBS}"
    echo "${ACTIVE_JOBS}" | awk '{print $1}' | xargs -r kill
    sleep 5
    ACTIVE_JOBS="$(find_active_jobs)"
    if [[ -n "${ACTIVE_JOBS}" ]]; then
      echo "[pivot] some jobs did not stop after SIGTERM; sending SIGKILL"
      echo "${ACTIVE_JOBS}" | awk '{print $1}' | xargs -r kill -9
      sleep 2
    fi
  elif [[ ${ALLOW_CONCURRENT} -eq 0 ]]; then
    echo "[pivot] active raw2task jobs are still running; refusing to start/wipe concurrently." >&2
    echo "${ACTIVE_JOBS}" >&2
    echo "[pivot] rerun with --stop-existing to stop them, or --allow-concurrent if you intentionally want overlap." >&2
    exit 3
  fi
fi

if [[ ${FRESH} -eq 1 ]]; then
  echo "[pivot] fresh requested: deleting revised experiment outputs"
  if [[ ${RUN_ACCURACY} -eq 1 ]]; then
    rm -rf runs/accuracy_recovery_fast runs/batch_size_plan_accuracy_fast.yaml
  fi
  if [[ ${RUN_CONSTRAINED} -eq 1 ]]; then
    rm -rf runs/constrained_codesign_fast runs/batch_size_plan_constrained_fast.yaml
  fi
  if [[ ${RUN_CITYSCAPES} -eq 1 ]]; then
    rm -rf runs/cityscapes_codesign_fast runs/batch_size_plan_cityscapes_fast.yaml
  fi
  if [[ ${RUN_FULL_STRESS} -eq 1 || "${MODE}" == "full" || "${MODE}" == "both" ]]; then
    rm -rf runs/sensor_stress_matrix runs/batch_size_plan_full_stress.yaml
  fi
fi

probe_and_run() {
  local label="$1"
  local matrix="$2"
  local out_plan="$3"
  local log="$4"
  local run_gpus="${5:-${GPUS}}"
  local probe_gpu
  probe_gpu="$(echo "${run_gpus}" | cut -d, -f1 | xargs)"
  echo "[pivot] ${label}: probing fixed batch sizes"
  deeplens/projects/raw2task/scripts/probe_batch_size.sh \
    --matrix "${matrix}" \
    --seeds "${SEEDS}" \
    --gpu "${probe_gpu}" \
    --max-batch "${MAX_BATCH}" \
    --out "${out_plan}" \
    2>&1 | tee "runs/pivot_${label}_preflight.log"

  echo "[pivot] ${label}: training matrix=${matrix} gpus=${run_gpus}"
  "${PYTHON_BIN}" -u -m raw2task.run_paper_experiments \
    --matrix "${matrix}" \
    --seeds "${SEEDS}" \
    --batch-plan "${out_plan}" \
    --gpus "${run_gpus}" \
    --max-parallel 1 \
    --order-policy matrix \
    --skip-existing \
    --robustness-experiments "" \
    2>&1 | tee "${log}"
}

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

run_kitti_fast() {
  local gpu="$1"
  if [[ ${RUN_ACCURACY} -eq 1 ]]; then
    probe_and_run "accuracy_fast" "${ACC_MATRIX}" "runs/batch_size_plan_accuracy_fast.yaml" "runs/pivot_accuracy_fast_live.log" "${gpu}"
  fi
  if [[ ${RUN_CONSTRAINED} -eq 1 ]]; then
    probe_and_run "constrained_fast" "${CLAIM_MATRIX}" "runs/batch_size_plan_constrained_fast.yaml" "runs/pivot_constrained_fast_live.log" "${gpu}"
  fi
}

run_cityscapes_fast() {
  local gpu="$1"
  if [[ ${RUN_CITYSCAPES} -eq 1 ]]; then
    probe_and_run "cityscapes_fast" "${CITY_MATRIX}" "runs/batch_size_plan_cityscapes_fast.yaml" "runs/pivot_cityscapes_fast_live.log" "${gpu}"
  fi
}

if [[ "${MODE}" != "fast" && "${MODE}" != "full" && "${MODE}" != "both" ]]; then
  echo "Unsupported --mode '${MODE}'. Use fast, full, or both." >&2
  exit 2
fi

if [[ ${SPLIT_DATASET_GPUS} -eq 1 && ${RUN_CITYSCAPES} -eq 1 && ( ${RUN_ACCURACY} -eq 1 || ${RUN_CONSTRAINED} -eq 1 ) ]]; then
  KITTI_GPU="$(first_gpu)"
  CITY_GPU="$(second_gpu_or_first)"
  if [[ "${KITTI_GPU}" == "${CITY_GPU}" ]]; then
    echo "[pivot] only one GPU supplied; running KITTI then Cityscapes sequentially on gpu=${KITTI_GPU}"
    run_kitti_fast "${KITTI_GPU}"
    run_cityscapes_fast "${CITY_GPU}"
  else
    echo "[pivot] split-dataset schedule: KITTI gpu=${KITTI_GPU}, Cityscapes gpu=${CITY_GPU}"
    run_kitti_fast "${KITTI_GPU}" &
    KITTI_PID=$!
    run_cityscapes_fast "${CITY_GPU}" &
    CITY_PID=$!
    wait "${KITTI_PID}"
    wait "${CITY_PID}"
  fi
else
  if [[ ${SPLIT_DATASET_GPUS} -eq 1 ]]; then
    run_kitti_fast "$(first_gpu)"
    run_cityscapes_fast "$(second_gpu_or_first)"
  else
    run_kitti_fast "${GPUS}"
    run_cityscapes_fast "${GPUS}"
  fi
fi

if [[ ${RUN_FULL_STRESS} -eq 1 || "${MODE}" == "full" || "${MODE}" == "both" ]]; then
  probe_and_run "full_stress" "${FULL_STRESS_MATRIX}" "runs/batch_size_plan_full_stress.yaml" "runs/pivot_full_stress_live.log" "$(first_gpu)"
fi

echo "[pivot] complete"
echo "[pivot] accuracy artifacts: runs/accuracy_recovery_fast/paper_tables"
echo "[pivot] constrained artifacts: runs/constrained_codesign_fast/paper_tables"
echo "[pivot] Cityscapes artifacts: runs/cityscapes_codesign_fast/paper_tables"
echo "[pivot] full stress artifacts: runs/sensor_stress_matrix/paper_tables"
