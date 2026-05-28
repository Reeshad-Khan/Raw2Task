#!/usr/bin/env bash
set -euo pipefail

GPUS="${GPUS:-${GPU_ID:-0,1}}"
PYTHON_BIN="${PYTHON_BIN:-/home/rk010/.conda/envs/raw2task/bin/python}"
NUM_CANDIDATES="${NUM_CANDIDATES:-8}"
CANDIDATE_PRIOR="${CANDIDATE_PRIOR:-physics}"
RUN_ROOT="${RUN_ROOT:-runs/hybrid_camera_search_fast}"
MAX_PARALLEL="${MAX_PARALLEL:-1}"
MAX_BATCH="${MAX_BATCH:-16}"
SEED="${SEED:-0}"
EPOCHS="${EPOCHS:-5}"
DATASET="${DATASET:-both}"
FRESH=0
STOP_EXISTING=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus|--gpu)
      GPUS="$2"; shift 2 ;;
    --python)
      PYTHON_BIN="$2"; shift 2 ;;
    --num-candidates)
      NUM_CANDIDATES="$2"; shift 2 ;;
    --candidate-prior)
      CANDIDATE_PRIOR="$2"; shift 2 ;;
    --run-root|--out-root)
      RUN_ROOT="$2"; shift 2 ;;
    --max-parallel)
      MAX_PARALLEL="$2"; shift 2 ;;
    --max-batch)
      MAX_BATCH="$2"; shift 2 ;;
    --seed)
      SEED="$2"; shift 2 ;;
    --epochs)
      EPOCHS="$2"; shift 2 ;;
    --dataset)
      DATASET="$2"; shift 2 ;;
    --fresh)
      FRESH=1; shift ;;
    --stop-existing)
      STOP_EXISTING=1; shift ;;
    --help|-h)
      cat <<'EOF'
Usage: deeplens/projects/raw2task/scripts/run_hybrid_physical_ai_search.sh [options]

Runs a hybrid physical-AI camera search:
  - deterministic candidate camera generation
  - fixed preflight batch probing
  - short co-design training per candidate
  - ranked camera_search_ranked.csv and best_camera_design.json export

Options:
  --dataset <kitti|cityscapes|both>  Default: both
  --fresh                            Delete hybrid search outputs first
  --stop-existing                    Stop active raw2task jobs first
  --gpus <csv>                       First GPU for KITTI, second for Cityscapes
  --num-candidates <n>               Candidate cameras per dataset
  --candidate-prior <physics|physics_local|heuristic>
                                      Physics-latent search by default; physics_local searches near broad winners
  --run-root <path>                   Output root for dataset folders
  --epochs <n>                       Short training epochs per candidate
  --max-batch <n>                    Probe maximum batch size
  --max-parallel <n>                 Jobs per dataset runner
  --seed <n>                         Candidate/search seed
  --python <path>                    Python executable
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
    | grep -E "deeplens\.projects\.raw2task\.(train_extended|run_paper_experiments|hybrid_camera_search)|run_hybrid_physical_ai_search|run_pivot_constrained_experiments|run_cross_dataset_comparison_fast" \
    | grep -v grep \
    | grep -v "run_hybrid_physical_ai_search" || true
}

ACTIVE_JOBS="$(find_active_jobs)"
if [[ -n "${ACTIVE_JOBS}" ]]; then
  if [[ ${STOP_EXISTING} -eq 1 ]]; then
    echo "[hybrid-search] stopping active raw2task jobs"
    echo "${ACTIVE_JOBS}"
    echo "${ACTIVE_JOBS}" | awk '{print $1}' | xargs -r kill
    sleep 5
    ACTIVE_JOBS="$(find_active_jobs)"
    if [[ -n "${ACTIVE_JOBS}" ]]; then
      echo "[hybrid-search] some jobs did not stop after SIGTERM; sending SIGKILL"
      echo "${ACTIVE_JOBS}" | awk '{print $1}' | xargs -r kill -9
      sleep 2
    fi
  else
    echo "[hybrid-search] active raw2task jobs found; rerun with --stop-existing or wait." >&2
    echo "${ACTIVE_JOBS}" >&2
    exit 3
  fi
fi

if [[ ${FRESH} -eq 1 ]]; then
  echo "[hybrid-search] fresh requested: deleting ${RUN_ROOT}"
  rm -rf "${RUN_ROOT}"
fi

run_one() {
  local label="$1"
  local matrix="$2"
  local template="$3"
  local gpu="$4"
  local out_root="$5"
  local plan="${out_root}/batch_size_plan.yaml"
  local candidate_matrix="${out_root}/candidate_matrix.yaml"

  mkdir -p "${out_root}"
  echo "[hybrid-search] ${label}: generating candidate matrix"
  "${PYTHON_BIN}" -u -m deeplens.projects.raw2task.hybrid_camera_search \
    --base-matrix "${matrix}" \
    --template "${template}" \
    --out-root "${out_root}" \
    --num-candidates "${NUM_CANDIDATES}" \
    --candidate-prior "${CANDIDATE_PRIOR}" \
    --seed "${SEED}" \
    --epochs "${EPOCHS}" \
    --generate-only

  echo "[hybrid-search] ${label}: probing fixed batch sizes on gpu=${gpu}"
  deeplens/projects/raw2task/scripts/probe_batch_size.sh \
    --matrix "${candidate_matrix}" \
    --seeds "${SEED}" \
    --gpu "${gpu}" \
    --max-batch "${MAX_BATCH}" \
    --out "${plan}" \
    2>&1 | tee "${out_root}/preflight.log"

  echo "[hybrid-search] ${label}: running candidate search on gpu=${gpu}"
  "${PYTHON_BIN}" -u -m deeplens.projects.raw2task.hybrid_camera_search \
    --base-matrix "${matrix}" \
    --template "${template}" \
    --out-root "${out_root}" \
    --num-candidates "${NUM_CANDIDATES}" \
    --candidate-prior "${CANDIDATE_PRIOR}" \
    --seed "${SEED}" \
    --epochs "${EPOCHS}" \
    --gpus "${gpu}" \
    --max-parallel "${MAX_PARALLEL}" \
    --batch-plan "${plan}" \
    2>&1 | tee "${out_root}/live.log"
}

KITTI_MATRIX="deeplens/projects/raw2task/configs/constrained_codesign_matrix_fast.yaml"
CITY_MATRIX="deeplens/projects/raw2task/configs/cityscapes_codesign_matrix_fast.yaml"

KITTI_GPU="$(first_gpu)"
CITY_GPU="$(second_gpu_or_first)"

case "${DATASET}" in
  kitti|kitti360)
    run_one "kitti360" "${KITTI_MATRIX}" "constrained_lowlight_codesign_segformer_b1" "${KITTI_GPU}" "${RUN_ROOT}/kitti360"
    ;;
  city|cityscapes)
    run_one "cityscapes" "${CITY_MATRIX}" "city_constrained_lowlight_codesign_segformer_b1" "${CITY_GPU}" "${RUN_ROOT}/cityscapes"
    ;;
  both)
    run_one "kitti360" "${KITTI_MATRIX}" "constrained_lowlight_codesign_segformer_b1" "${KITTI_GPU}" "${RUN_ROOT}/kitti360" &
    KITTI_PID=$!
    run_one "cityscapes" "${CITY_MATRIX}" "city_constrained_lowlight_codesign_segformer_b1" "${CITY_GPU}" "${RUN_ROOT}/cityscapes" &
    CITY_PID=$!
    wait "${KITTI_PID}"
    wait "${CITY_PID}"
    ;;
  *)
    echo "Unsupported --dataset '${DATASET}'. Use kitti, cityscapes, or both." >&2
    exit 2
    ;;
esac

echo "[hybrid-search] complete"
echo "[hybrid-search] KITTI ranked: ${RUN_ROOT}/kitti360/camera_search_ranked.csv"
echo "[hybrid-search] City ranked : ${RUN_ROOT}/cityscapes/camera_search_ranked.csv"
