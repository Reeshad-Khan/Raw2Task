#!/usr/bin/env bash
set -euo pipefail

# --- Resolve REPO_ROOT as the top-level 'DeepLens' directory ---
# scripts is: DeepLens/deeplens/projects/raw2task/scripts
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd -P)"   # four levels up -> DeepLens
LOG_ROOT="${REPO_ROOT}/logs"
mkdir -p "${LOG_ROOT}"

# --- Python path (safe if PYTHONPATH unset) ---
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# --- Configs are stored relative to REPO_ROOT ---
CONFIGS=(
  deeplens/projects/raw2task/configs/kitti360_seg_cellphone80.yaml
  deeplens/projects/raw2task/configs/kitti360_seg_identity.yaml
  deeplens/projects/raw2task/configs/kitti360_seg_bayer_richlens.yaml
  deeplens/projects/raw2task/configs/kitti360_seg_lowbit_fast.yaml
  deeplens/projects/raw2task/configs/kitti360_seg_robust_noise.yaml
  deeplens/projects/raw2task/configs/kitti360_seg_nosensor_ablation.yaml
  deeplens/projects/raw2task/configs/kitti360_seg_wide_model.yaml
)

# --- GPU detection (fallback to 1 if nvidia-smi missing/returns 0) ---
if command -v nvidia-smi >/dev/null 2>&1; then
  NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')
  [[ "${NUM_GPUS}" -gt 0 ]] || NUM_GPUS=1
else
  NUM_GPUS=1
fi

echo "Repo root: ${REPO_ROOT}"
echo "Detected ${NUM_GPUS} GPU(s). Will run ${NUM_GPUS} job(s) at a time (one per GPU)."
echo "Total jobs: ${#CONFIGS[@]}"

# Kill background jobs only on INT/TERM/ERR (not on normal exit after wait)
cleanup() {
  echo "Stopping background jobs..."
  jobs -pr | xargs -r kill || true
}
trap cleanup INT TERM ERR

# Sanity-check configs
for cfg in "${CONFIGS[@]}"; do
  if [[ ! -f "${REPO_ROOT}/${cfg}" ]]; then
    echo "ERROR: Missing config: ${REPO_ROOT}/${cfg}" >&2
    exit 1
  fi
done

total=${#CONFIGS[@]}
start=0
batch_idx=0

while [[ $start -lt $total ]]; do
  end=$(( start + NUM_GPUS ))
  if [[ $end -gt $total ]]; then end=$total; fi

  echo "=== Batch $((batch_idx+1)) : launching jobs $start..$((end-1)) ==="
  pids=()

  # Launch one job per GPU in this batch
  gpu=0
  for (( i=start; i<end; i++ )); do
    cfg="${CONFIGS[$i]}"
    name="$(basename "${cfg%.yaml}")"
    logdir="${LOG_ROOT}/${name}"
    mkdir -p "${logdir}"

    echo "-> GPU ${gpu}: ${cfg}"
    (
      CUDA_VISIBLE_DEVICES=${gpu} \
      python -u -m deeplens.projects.raw2task.train_extended \
        --config "${REPO_ROOT}/${cfg}" \
        > "${logdir}/train.out" 2>&1
    ) &
    pids+=("$!")
    gpu=$(( (gpu + 1) % NUM_GPUS ))
    sleep 2
  done

  echo "Launched ${#pids[@]} job(s). Tail logs with:"
  echo "  tail -f ${LOG_ROOT}/*/train.out"
  echo "Waiting for this batch to finish..."
  wait   # wait for the whole batch
  echo "Batch $((batch_idx+1)) finished."
  echo

  start=$end
  batch_idx=$((batch_idx+1))
done

echo "All jobs completed."