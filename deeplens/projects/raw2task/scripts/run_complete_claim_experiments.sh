#!/usr/bin/env bash
set -euo pipefail

# One-command workflow for the defensible paper claims:
#   1. Probe fixed batch sizes on each requested GPU.
#   2. Run the fast sanity matrix.
#   3. Run the full matched segmentation matrix.
#   4. Run the fast and full optics/sensor stress matrices.
#   5. Refresh paper tables, camera measurements, GIFs, and claim-readiness files.
#
# Example:
#   deeplens/projects/raw2task/scripts/run_complete_claim_experiments.sh \
#     2>&1 | tee runs/complete_claim_experiments_live.log

MODE="${MODE:-both}"
GPUS="${GPUS:-${GPU_ID:-0,1}}"
MAX_BATCH="${MAX_BATCH:-16}"
FAST_SEEDS="${FAST_SEEDS:-0}"
FULL_SEEDS="${FULL_SEEDS:-0,1,2}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
SKIP_EXISTING=1
RUN_STANDARD=1
RUN_STRESS=1
RUN_OCCUPANCY=1
RUN_SEG3D=1
RUN_UNIFIED="${RUN_UNIFIED:-0}"
SKIP_OCCUPANCY_BUILD=0
OCCUPANCY_EPOCHS="${OCCUPANCY_EPOCHS:-0}"
OCCUPANCY_STRIDE="${OCCUPANCY_STRIDE:-20}"
OCCUPANCY_MAX_SAMPLES="${OCCUPANCY_MAX_SAMPLES:-0}"
OCCUPANCY_OUT_ROOT="${OCCUPANCY_OUT_ROOT:-data_external/kitti360_occupancy}"
KITTI_ROOT="${KITTI_ROOT:-/home/rk010/Desktop/Research/NurIPS/KITTI-360}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="$2"
      shift 2
      ;;
    --gpus|--gpu)
      GPUS="$2"
      shift 2
      ;;
    --max-batch)
      MAX_BATCH="$2"
      shift 2
      ;;
    --fast-seeds)
      FAST_SEEDS="$2"
      shift 2
      ;;
    --full-seeds)
      FULL_SEEDS="$2"
      shift 2
      ;;
    --fresh)
      SKIP_EXISTING=0
      shift
      ;;
    --standard-only)
      RUN_STANDARD=1
      RUN_STRESS=0
      RUN_OCCUPANCY=0
      shift
      ;;
    --stress-only)
      RUN_STANDARD=0
      RUN_STRESS=1
      RUN_OCCUPANCY=0
      shift
      ;;
    --occupancy-only)
      RUN_STANDARD=0
      RUN_STRESS=0
      RUN_OCCUPANCY=1
      RUN_SEG3D=1
      shift
      ;;
    --unified-only)
      RUN_STANDARD=0
      RUN_STRESS=0
      RUN_OCCUPANCY=0
      RUN_SEG3D=0
      RUN_UNIFIED=1
      shift
      ;;
    --with-unified)
      RUN_UNIFIED=1
      shift
      ;;
    --skip-occupancy)
      RUN_OCCUPANCY=0
      shift
      ;;
    --skip-3d-segmentation)
      RUN_SEG3D=0
      shift
      ;;
    --skip-occupancy-build)
      SKIP_OCCUPANCY_BUILD=1
      shift
      ;;
    --occupancy-epochs)
      OCCUPANCY_EPOCHS="$2"
      shift 2
      ;;
    --occupancy-stride)
      OCCUPANCY_STRIDE="$2"
      shift 2
      ;;
    --occupancy-max-samples)
      OCCUPANCY_MAX_SAMPLES="$2"
      shift 2
      ;;
    --occupancy-out-root)
      OCCUPANCY_OUT_ROOT="$2"
      shift 2
      ;;
    --kitti-root)
      KITTI_ROOT="$2"
      shift 2
      ;;
    --no-stress)
      RUN_STRESS=0
      shift
      ;;
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --help|-h)
      cat <<'EOF'
Usage: deeplens/projects/raw2task/scripts/run_complete_claim_experiments.sh [options]

Options:
  --mode fast|full|both    fast sanity, full paper, or both (default: both)
  --gpus <csv>             GPU ids for concurrent jobs (default: 0,1)
  --max-batch <n>          Highest batch size to probe (default: 16)
  --fast-seeds <csv>       Seeds for fast matrices (default: 0)
  --full-seeds <csv>       Seeds for full matrices (default: 0,1,2)
  --fresh                  Re-run existing jobs instead of resuming/skipping
  --standard-only          Run matched RGB/fixed/co-design segmentation matrices only
  --stress-only            Run optics/sensor stress matrices only
  --occupancy-only         Run KITTI-360 observed occupancy only
  --unified-only           Run one shared 2D+3D+occupancy model only
  --with-unified           Also run the unified multi-task model after per-task jobs
  --skip-occupancy         Do not run observed occupancy
  --skip-3d-segmentation   Do not run observed 3D semantic segmentation
  --skip-occupancy-build   Reuse data_external/kitti360_occupancy manifests
  --occupancy-epochs <n>   Override occupancy epochs
  --occupancy-stride <n>   KITTI frame stride for observed occupancy build
  --occupancy-max-samples <n>
                           Debug cap for occupancy manifest build
  --occupancy-out-root <p> Observed occupancy artifact root
  --kitti-root <p>         KITTI-360 root
  --no-stress              Skip sensor stress matrices
  --python <path>          Python executable

Outputs:
  runs/fast_paper_experiments_live.log
  runs/paper_experiments_live.log
  runs/sensor_stress_fast_live.log
  runs/sensor_stress_full_live.log
  runs/*/paper_tables/claim_readiness.md
  imgs/raw2task_*.gif
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

mkdir -p runs imgs

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export MPLBACKEND="${MPLBACKEND:-Agg}"
export PYTHONUNBUFFERED=1
export GPUS MAX_BATCH FAST_SEEDS FULL_SEEDS PYTHON_BIN

skip_flag=()
if [[ ${SKIP_EXISTING} -eq 0 ]]; then
  skip_flag+=(--no-skip-existing)
fi

echo "[complete] repo=${REPO_ROOT}"
echo "[complete] mode=${MODE} gpus=${GPUS} max_batch=${MAX_BATCH} fast_seeds=${FAST_SEEDS} full_seeds=${FULL_SEEDS}"
echo "[complete] standard=${RUN_STANDARD} stress=${RUN_STRESS} occupancy=${RUN_OCCUPANCY} seg3d=${RUN_SEG3D} unified=${RUN_UNIFIED} skip_existing=${SKIP_EXISTING}"

if [[ ${SKIP_EXISTING} -eq 0 ]]; then
  echo "[complete] fresh requested: deleting selected experiment outputs before launch"
  fresh_targets=()
  if [[ ${RUN_STANDARD} -eq 1 ]]; then
    if [[ "${MODE}" == "fast" || "${MODE}" == "both" ]]; then
      fresh_targets+=(runs/industry_paper_matrix_fast runs/batch_size_plan_fast.yaml runs/batch_size_plan_fast_gpu*.yaml runs/batch_size_probe_fast.log runs/fast_paper_experiments_live.log)
    fi
    if [[ "${MODE}" == "full" || "${MODE}" == "both" ]]; then
      fresh_targets+=(runs/industry_paper_matrix runs/batch_size_plan_full.yaml runs/batch_size_plan_full_gpu*.yaml runs/batch_size_probe_full.log runs/paper_experiments_live.log)
    fi
  fi
  if [[ ${RUN_STRESS} -eq 1 ]]; then
    if [[ "${MODE}" == "fast" || "${MODE}" == "both" ]]; then
      fresh_targets+=(runs/sensor_stress_matrix_fast runs/batch_size_plan_stress_fast.yaml runs/batch_size_plan_stress_fast_gpu*.yaml runs/stress_fast_preflight.log runs/sensor_stress_fast_live.log)
    fi
    if [[ "${MODE}" == "full" || "${MODE}" == "both" ]]; then
      fresh_targets+=(runs/sensor_stress_matrix runs/batch_size_plan_stress_full.yaml runs/batch_size_plan_stress_full_gpu*.yaml runs/stress_full_preflight.log runs/sensor_stress_full_live.log)
    fi
  fi
  if [[ ${RUN_OCCUPANCY} -eq 1 ]]; then
    fresh_targets+=(runs/occupancy)
  fi
  if [[ ${RUN_SEG3D} -eq 1 ]]; then
    fresh_targets+=(runs/seg3d)
  fi
  if [[ ${RUN_UNIFIED} -eq 1 ]]; then
    fresh_targets+=(runs/unified_multitask)
  fi
  for target in "${fresh_targets[@]}"; do
    rm -rf ${target}
  done
fi

if [[ ${RUN_STANDARD} -eq 1 ]]; then
  echo "[complete] running matched segmentation workflow"
  deeplens/projects/raw2task/scripts/run_preflight_then_experiments.sh \
    --mode "${MODE}" \
    --gpus "${GPUS}" \
    --max-batch "${MAX_BATCH}" \
    --fast-seeds "${FAST_SEEDS}" \
    --full-seeds "${FULL_SEEDS}" \
    --python "${PYTHON_BIN}" \
    "${skip_flag[@]}"
fi

if [[ ${RUN_STRESS} -eq 1 ]]; then
  stress_mode="${MODE}"
  if [[ "${MODE}" == "both" ]]; then
    echo "[complete] running fast stress workflow"
    deeplens/projects/raw2task/scripts/run_sensor_stress_experiments.sh \
      --mode fast \
      --gpus "${GPUS}" \
      --max-batch "${MAX_BATCH}" \
      --fast-seeds "${FAST_SEEDS}" \
      --python "${PYTHON_BIN}" \
      "${skip_flag[@]}" \
      2>&1 | tee runs/sensor_stress_fast_live.log
    echo "[complete] running full stress workflow"
    deeplens/projects/raw2task/scripts/run_sensor_stress_experiments.sh \
      --mode full \
      --gpus "${GPUS}" \
      --max-batch "${MAX_BATCH}" \
      --full-seeds "${FULL_SEEDS}" \
      --python "${PYTHON_BIN}" \
      "${skip_flag[@]}" \
      2>&1 | tee runs/sensor_stress_full_live.log
  else
    echo "[complete] running ${stress_mode} stress workflow"
    deeplens/projects/raw2task/scripts/run_sensor_stress_experiments.sh \
      --mode "${stress_mode}" \
      --gpus "${GPUS}" \
      --max-batch "${MAX_BATCH}" \
      --fast-seeds "${FAST_SEEDS}" \
      --full-seeds "${FULL_SEEDS}" \
      --python "${PYTHON_BIN}" \
      "${skip_flag[@]}" \
      2>&1 | tee "runs/sensor_stress_${stress_mode}_live.log"
  fi
fi

if [[ ${RUN_OCCUPANCY} -eq 1 || ${RUN_SEG3D} -eq 1 ]]; then
  echo "[complete] running KITTI-360 observed 3D voxel workflows"
  occ_args=(
    --skip-segmentation
    --gpus "${GPUS}"
    --max-parallel 0
    --kitti-root "${KITTI_ROOT}"
    --occupancy-out-root "${OCCUPANCY_OUT_ROOT}"
    --occupancy-stride "${OCCUPANCY_STRIDE}"
    --occupancy-max-samples "${OCCUPANCY_MAX_SAMPLES}"
    --python "${PYTHON_BIN}"
  )
  if [[ "${OCCUPANCY_EPOCHS}" != "0" ]]; then
    occ_args+=(--occupancy-epochs "${OCCUPANCY_EPOCHS}")
  fi
  if [[ ${SKIP_OCCUPANCY_BUILD} -eq 1 ]]; then
    occ_args+=(--skip-occupancy-build)
  fi
  if [[ ${RUN_OCCUPANCY} -eq 0 ]]; then
    occ_args+=(--skip-occupancy)
  fi
  if [[ ${RUN_SEG3D} -eq 0 ]]; then
    occ_args+=(--skip-3d-segmentation)
  fi
  if [[ ${SKIP_EXISTING} -eq 1 ]]; then
    occ_args+=(--skip-existing)
  fi
  deeplens/projects/raw2task/scripts/run_full_paper_experiments.sh "${occ_args[@]}" \
    2>&1 | tee runs/occupancy_live.log
fi

if [[ ${RUN_UNIFIED} -eq 1 ]]; then
  echo "[complete] running unified 2D+3D+occupancy multi-task workflow"
  unified_args=(--mode "${MODE}" --gpus "${GPUS}" --python "${PYTHON_BIN}" --kitti-root "${KITTI_ROOT}" --occupancy-out-root "${OCCUPANCY_OUT_ROOT}" --occupancy-stride "${OCCUPANCY_STRIDE}" --occupancy-max-samples "${OCCUPANCY_MAX_SAMPLES}")
  if [[ ${SKIP_EXISTING} -eq 0 ]]; then
    unified_args+=(--fresh)
  fi
  deeplens/projects/raw2task/scripts/run_unified_multitask_fast.sh "${unified_args[@]}" \
    2>&1 | tee runs/unified_multitask_live.log
fi

"${PYTHON_BIN}" -m deeplens.projects.raw2task.analyze_unified_results \
  --runs-root runs/unified_multitask \
  --out-dir runs/unified_multitask/paper_tables || true

echo "[complete] done"
echo "[complete] standard artifacts: runs/industry_paper_matrix_fast/paper_tables and runs/industry_paper_matrix/paper_tables"
echo "[complete] stress artifacts: runs/sensor_stress_matrix_fast/paper_tables and runs/sensor_stress_matrix/paper_tables"
echo "[complete] occupancy artifacts: runs/occupancy/summary.csv and data_external/kitti360_occupancy"
echo "[complete] 3D segmentation artifacts: runs/seg3d/summary.csv and runs/seg3d/paper_tables"
echo "[complete] unified multi-task artifacts: runs/unified_multitask"
echo "[complete] camera GIFs: imgs/raw2task_*.gif"
