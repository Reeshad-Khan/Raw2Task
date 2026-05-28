#!/usr/bin/env bash
set -euo pipefail

# --- Resolve REPO_ROOT as the top-level 'DeepLens' directory ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd -P)"   # four levels up -> DeepLens
echo "Repo root: ${REPO_ROOT}"

# --- Python path (safe if PYTHONPATH unset) ---
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# --- Configs to evaluate (same as training script) ---
CONFIGS=(
  deeplens/projects/raw2task/configs/kitti360_seg_cellphone80.yaml
  deeplens/projects/raw2task/configs/kitti360_seg_identity.yaml
  deeplens/projects/raw2task/configs/kitti360_seg_bayer_richlens.yaml
  deeplens/projects/raw2task/configs/kitti360_seg_lowbit_fast.yaml
  deeplens/projects/raw2task/configs/kitti360_seg_robust_noise.yaml
  deeplens/projects/raw2task/configs/kitti360_seg_nosensor_ablation.yaml
  deeplens/projects/raw2task/configs/kitti360_seg_wide_model.yaml
)

# --- Where to write evaluation metrics + comparison images ---
OUT_ROOT="${REPO_ROOT}/test_results"
mkdir -p "${OUT_ROOT}"

# --- Check that configs exist ---
for cfg in "${CONFIGS[@]}"; do
  if [[ ! -f "${REPO_ROOT}/${cfg}" ]]; then
    echo "ERROR: Missing config: ${REPO_ROOT}/${cfg}" >&2
    exit 1
  fi
done

# --- Helper: get ckpt_dir from YAML (relative to REPO_ROOT if not absolute) ---
get_ckpt_dir() {
  local cfg_path="$1"

  # Pass REPO_ROOT via env; pass cfg_path via argv
  REPO_ROOT="${REPO_ROOT}" python - "$cfg_path" <<'PY'
import os, sys, yaml

repo_root = os.environ["REPO_ROOT"]
cfg_path = sys.argv[1]

with open(cfg_path, "r") as f:
    y = yaml.safe_load(f)

train = y.get("train", {}) or {}
ckpt = train.get("ckpt_dir", "./checkpoints")

if not os.path.isabs(ckpt):
    ckpt = os.path.join(repo_root, ckpt)

print(ckpt)
PY
}

BEST_CKPTS=()

echo "=== Discovering best checkpoints for all configs ==="
for cfg in "${CONFIGS[@]}"; do
  full_cfg="${REPO_ROOT}/${cfg}"
  name="$(basename "${cfg%.yaml}")"

  ckpt_dir="$(get_ckpt_dir "${full_cfg}")"
  echo "Config: ${cfg}"
  echo "  -> ckpt_dir: ${ckpt_dir}"

  if [[ ! -d "${ckpt_dir}" ]]; then
    echo "  !! WARNING: ckpt_dir does not exist, skipping: ${ckpt_dir}"
    continue
  fi

  # Prefer best_ep*.pt, fall back to last.pt if needed
  best_pt="$(ls -t "${ckpt_dir}"/best_ep*.pt 2>/dev/null | head -n 1 || true)"
  if [[ -z "${best_pt}" ]]; then
    if [[ -f "${ckpt_dir}/last.pt" ]]; then
      best_pt="${ckpt_dir}/last.pt"
      echo "  -> Using last.pt (no best_ep*.pt found)."
    else
      echo "  !! WARNING: No best_ep*.pt or last.pt in ${ckpt_dir}, skipping."
      continue
    fi
  else
    echo "  -> Using best checkpoint: ${best_pt}"
  fi

  BEST_CKPTS+=("${best_pt}")
done

if [[ ${#BEST_CKPTS[@]} -eq 0 ]]; then
  echo "ERROR: No checkpoints found for any config. Did you run training first?" >&2
  exit 1
fi

echo
echo "=== Will evaluate the following checkpoints ==="
for ck in "${BEST_CKPTS[@]}"; do
  echo "  ${ck}"
done
echo

# --- Optional: choose a GPU for evaluation (default: 0 if CUDA exists) ---
GPU_ID=0
if command -v nvidia-smi >/dev/null 2>&1; then
  NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')
  if [[ "${NUM_GPUS}" -gt 0 ]]; then
    echo "Detected ${NUM_GPUS} GPU(s). Using GPU ${GPU_ID} for evaluation."
  else
    echo "No visible GPUs; running on CPU."
  fi
else
  echo "nvidia-smi not found; running on CPU."
fi

# --- Run the Python test script once with all checkpoints ---
echo "=== Running test_seg_models.py ==="
CUDA_VISIBLE_DEVICES=${GPU_ID} \
python -u -m deeplens.projects.raw2task.test_seg_models \
  --ckpts "${BEST_CKPTS[@]}" \
  --out-root "${OUT_ROOT}" \
  --num-vis 6

echo "=== All evaluations completed. Results in: ${OUT_ROOT} ==="