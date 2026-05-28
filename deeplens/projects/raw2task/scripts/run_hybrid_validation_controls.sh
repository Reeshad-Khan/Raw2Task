#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/rk010/.conda/envs/raw2task/bin/python}"
GPUS="${GPUS:-${GPU_ID:-0,1}}"
EPOCHS="${EPOCHS:-24}"
MODEL="${MODEL:-segformer_b1}"
RGB_MODEL="${RGB_MODEL:-${MODEL}}"
RANK_METRIC="${RANK_METRIC:-regular}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:--1}"
MAX_VAL_SAMPLES="${MAX_VAL_SAMPLES:--1}"
IMG_HEIGHT="${IMG_HEIGHT:-0}"
IMG_WIDTH="${IMG_WIDTH:-0}"
BATCH_SIZE="${BATCH_SIZE:-0}"
ACCUM_STEPS="${ACCUM_STEPS:-0}"
DATASET="${DATASET:-both}"
FRESH=0
STOP_EXISTING=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --gpus|--gpu) GPUS="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --rgb-model) RGB_MODEL="$2"; shift 2 ;;
    --rank-metric) RANK_METRIC="$2"; shift 2 ;;
    --max-train-samples) MAX_TRAIN_SAMPLES="$2"; shift 2 ;;
    --max-val-samples) MAX_VAL_SAMPLES="$2"; shift 2 ;;
    --img-height) IMG_HEIGHT="$2"; shift 2 ;;
    --img-width) IMG_WIDTH="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --accum-steps) ACCUM_STEPS="$2"; shift 2 ;;
    --dataset) DATASET="$2"; shift 2 ;;
    --fresh) FRESH=1; shift ;;
    --stop-existing) STOP_EXISTING=1; shift ;;
    --help|-h)
      cat <<'EOF'
Usage: deeplens/projects/raw2task/scripts/run_hybrid_validation_controls.sh [options]

Builds and runs fair follow-up controls from completed hybrid search results:
  - top regular/avg-ranked candidate co-design
  - same candidate fixed-camera control
  - RGB control

Useful knobs:
  --rank-metric regular|avg_bestk   Which ranked CSV to draw candidates from.
  --model segformer_b1|segformer_b2 Main model for co-design/fixed controls.
  --rgb-model MODEL                 Optional stronger RGB baseline.
  --max-train-samples 0             Full train split; -1 keeps template cap.
  --max-val-samples 0               Full val split; -1 keeps template cap.
  --img-height H --img-width W      Override resolution.
EOF
      exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd -P)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export MPLBACKEND="${MPLBACKEND:-Agg}"
export PYTHONUNBUFFERED=1
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
    | grep -E "deeplens\.projects\.raw2task\.(train_extended|run_paper_experiments|hybrid_camera_search)|run_hybrid_validation_controls" \
    | grep -v grep \
    | grep -v "run_hybrid_validation_controls" || true
}

ACTIVE_JOBS="$(find_active_jobs)"
if [[ -n "${ACTIVE_JOBS}" ]]; then
  if [[ ${STOP_EXISTING} -eq 1 ]]; then
    echo "[hybrid-validation] stopping active raw2task jobs"
    echo "${ACTIVE_JOBS}"
    echo "${ACTIVE_JOBS}" | awk '{print $1}' | xargs -r kill
    sleep 5
    ACTIVE_JOBS="$(find_active_jobs)"
    if [[ -n "${ACTIVE_JOBS}" ]]; then
      echo "${ACTIVE_JOBS}" | awk '{print $1}' | xargs -r kill -9
      sleep 2
    fi
  else
    echo "[hybrid-validation] active raw2task jobs found; rerun with --stop-existing or wait." >&2
    echo "${ACTIVE_JOBS}" >&2
    exit 3
  fi
fi

if [[ "${RANK_METRIC}" == "avg_bestk" ]]; then
  RANK_FILE="camera_search_ranked_avg_bestk.csv"
else
  RANK_FILE="camera_search_ranked_regular.csv"
fi

if [[ ${FRESH} -eq 1 ]]; then
  rm -rf "runs/hybrid_validation_${RANK_METRIC}_${MODEL}"
fi

build_one() {
  local dataset="$1"
  local base_matrix="$2"
  local ranked_csv="$3"
  local prefix="$4"
  local codesign_template="$5"
  local fixed_template="$6"
  local rgb_template="$7"
  local matrix_out="$8"
  local out_root="$9"

  "${PYTHON_BIN}" -m deeplens.projects.raw2task.build_hybrid_validation_matrix \
    --base-matrix "${base_matrix}" \
    --ranked-csv "${ranked_csv}" \
    --output "${matrix_out}" \
    --out-root "${out_root}" \
    --prefix "${prefix}" \
    --codesign-template "${codesign_template}" \
    --fixed-template "${fixed_template}" \
    --rgb-template "${rgb_template}" \
    --top-k 2 \
    --model "${MODEL}" \
    --rgb-model "${RGB_MODEL}" \
    --epochs "${EPOCHS}" \
    --max-train-samples "${MAX_TRAIN_SAMPLES}" \
    --max-val-samples "${MAX_VAL_SAMPLES}" \
    --img-height "${IMG_HEIGHT}" \
    --img-width "${IMG_WIDTH}" \
    --batch-size "${BATCH_SIZE}" \
    --accum-steps "${ACCUM_STEPS}"
}

run_one() {
  local label="$1"
  local matrix="$2"
  local gpu="$3"
  echo "[hybrid-validation] running ${label} on gpu=${gpu}"
  "${PYTHON_BIN}" -u -m deeplens.projects.raw2task.run_paper_experiments \
    --matrix "${matrix}" \
    --seeds 0 \
    --gpus "${gpu}" \
    --max-parallel 1 \
    --order-policy matrix \
    --skip-existing \
    --fresh-summary \
    --summary-only \
    --robustness-experiments ""
}

ROOT="runs/hybrid_validation_${RANK_METRIC}_${MODEL}"
mkdir -p "${ROOT}"

KITTI_MATRIX="${ROOT}/kitti360_validation_matrix.yaml"
CITY_MATRIX="${ROOT}/cityscapes_validation_matrix.yaml"

case "${DATASET}" in
  kitti|kitti360|both)
    build_one \
      kitti360 \
      deeplens/projects/raw2task/configs/constrained_codesign_matrix_fast.yaml \
      "runs/hybrid_camera_search_fast/kitti360/${RANK_FILE}" \
      "kitti360_hybrid_${RANK_METRIC}" \
      constrained_lowlight_codesign_segformer_b1 \
      constrained_lowlight_fixed_segformer_b1 \
      constrained_clean_rgb_segformer_b1 \
      "${KITTI_MATRIX}" \
      "${ROOT}/kitti360"
    ;;
esac

case "${DATASET}" in
  city|cityscapes|both)
    build_one \
      cityscapes \
      deeplens/projects/raw2task/configs/cityscapes_codesign_matrix_fast.yaml \
      "runs/hybrid_camera_search_fast/cityscapes/${RANK_FILE}" \
      "cityscapes_hybrid_${RANK_METRIC}" \
      city_constrained_lowlight_codesign_segformer_b1 \
      city_constrained_lowlight_fixed_segformer_b1 \
      city_constrained_clean_rgb_segformer_b1 \
      "${CITY_MATRIX}" \
      "${ROOT}/cityscapes"
    ;;
esac

case "${DATASET}" in
  kitti|kitti360)
    run_one kitti360 "${KITTI_MATRIX}" "$(first_gpu)" ;;
  city|cityscapes)
    run_one cityscapes "${CITY_MATRIX}" "$(second_gpu_or_first)" ;;
  both)
    run_one kitti360 "${KITTI_MATRIX}" "$(first_gpu)" &
    KP=$!
    run_one cityscapes "${CITY_MATRIX}" "$(second_gpu_or_first)" &
    CP=$!
    wait "${KP}"
    wait "${CP}"
    ;;
  *) echo "Unsupported --dataset '${DATASET}'" >&2; exit 2 ;;
esac

echo "[hybrid-validation] complete"
