#!/usr/bin/env bash
set -euo pipefail

# Unified workflow for a single optics-sensor-model stack:
# 2D semantic segmentation + observed 3D semantic segmentation + occupancy.

MODE="${MODE:-fast}"
GPUS="${GPUS:-${GPU_ID:-0,1}}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
KITTI_ROOT="${KITTI_ROOT:-/home/rk010/Desktop/Research/NurIPS/KITTI-360}"
OCCUPANCY_OUT_ROOT="${OCCUPANCY_OUT_ROOT:-data_external/kitti360_occupancy}"
OCCUPANCY_STRIDE="${OCCUPANCY_STRIDE:-20}"
OCCUPANCY_MAX_SAMPLES="${OCCUPANCY_MAX_SAMPLES:-400}"
FRESH=0
ONLY=""

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
    --occupancy-stride)
      OCCUPANCY_STRIDE="$2"
      shift 2
      ;;
    --occupancy-max-samples)
      OCCUPANCY_MAX_SAMPLES="$2"
      shift 2
      ;;
    --fresh)
      FRESH=1
      shift
      ;;
    --only)
      ONLY="$2"
      shift 2
      ;;
    --help|-h)
      cat <<'EOF'
Usage: deeplens/projects/raw2task/scripts/run_unified_multitask_fast.sh [options]

Options:
  --mode fast|full|both          Unified protocol to run (default: fast)
  --gpus <csv>                   GPUs for concurrent jobs (default: 0,1)
  --python <path>                Python executable
  --kitti-root <path>            KITTI-360 root
  --occupancy-out-root <path>    Observed voxel artifact root
  --occupancy-stride <n>         KITTI frame stride when building voxel data
  --occupancy-max-samples <n>    Debug cap for voxel data build
  --fresh                        Delete previous unified fast runs first
  --only <csv>                   Run only configs whose basename/name contains one of these tokens
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

mkdir -p runs/unified_multitask
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export MPLBACKEND="${MPLBACKEND:-Agg}"
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

if [[ ${FRESH} -eq 1 ]]; then
  rm -rf runs/unified_multitask
  mkdir -p runs/unified_multitask
fi

if [[ ! -f "${OCCUPANCY_OUT_ROOT}/manifests/train.jsonl" || ! -f "${OCCUPANCY_OUT_ROOT}/manifests/val.jsonl" ]]; then
  echo "[unified] building observed KITTI-360 voxel assets -> ${OCCUPANCY_OUT_ROOT}"
  build_cmd=("${PYTHON_BIN}" -m raw2task.occupancy.kitti360_builder \
    --root "${KITTI_ROOT}" \
    --out-root "${OCCUPANCY_OUT_ROOT}" \
    --stride "${OCCUPANCY_STRIDE}" \
    --grid-shape 16 64 128 \
    --min-valid-voxels 20)
  if [[ "${OCCUPANCY_MAX_SAMPLES}" != "0" ]]; then
    build_cmd+=(--max-samples "${OCCUPANCY_MAX_SAMPLES}")
  fi
  "${build_cmd[@]}"
fi

fast_configs=(
  deeplens/projects/raw2task/configs/unified/kitti360_unified_codesign_tasktokens_fast.yaml
  deeplens/projects/raw2task/configs/unified/kitti360_unified_fixed_camera_tasktokens_fast.yaml
  deeplens/projects/raw2task/configs/unified/kitti360_unified_rgb_fast.yaml
)

make_full_configs() {
  local out_dir="runs/unified_multitask/generated_full_configs"
  mkdir -p "${out_dir}"
  "${PYTHON_BIN}" - "${out_dir}" "${fast_configs[@]}" <<'PY'
import sys, yaml
from pathlib import Path
out = Path(sys.argv[1])
for src in map(Path, sys.argv[2:]):
    with open(src) as f:
        cfg = yaml.safe_load(f) or {}
    cfg["name"] = str(cfg.get("name", src.stem)).replace("_fast", "_full")
    d2 = cfg.setdefault("data_2d", {})
    d2["img_size"] = [384, 1280]
    d2["batch_size"] = 2
    d2["train_stride"] = 1
    d2["val_stride"] = 1
    d2.pop("max_train_samples", None)
    d2.pop("max_val_samples", None)
    dv = cfg.setdefault("data_voxel", {})
    dv["image_size"] = [192, 640]
    dv["batch_size"] = 1
    dv.pop("max_train_samples", None)
    dv.pop("max_val_samples", None)
    tr = cfg.setdefault("train", {})
    tr["epochs"] = 40
    tr["steps_per_epoch"] = 1000
    tr["log_interval"] = 50
    tr["ckpt_dir"] = str(tr.get("ckpt_dir", "./runs/unified_multitask/" + cfg["name"])).replace("_fast", "_full")
    dst = out / src.name.replace("_fast.yaml", "_full.yaml")
    with open(dst, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(dst)
PY
}

configs=()
if [[ "${MODE}" == "fast" || "${MODE}" == "both" ]]; then
  configs+=("${fast_configs[@]}")
fi
if [[ "${MODE}" == "full" || "${MODE}" == "both" ]]; then
  while IFS= read -r cfg; do
    configs+=("${cfg}")
  done < <(make_full_configs)
fi
if [[ ${#configs[@]} -eq 0 ]]; then
  echo "Unsupported --mode '${MODE}'. Expected fast, full, or both." >&2
  exit 2
fi
if [[ -n "${ONLY}" ]]; then
  IFS=',' read -r -a only_tokens <<< "${ONLY}"
  filtered=()
  for cfg in "${configs[@]}"; do
    base="$(basename "${cfg}" .yaml)"
    for token in "${only_tokens[@]}"; do
      token="$(echo "${token}" | xargs)"
      if [[ -n "${token}" && "${base}" == *"${token}"* ]]; then
        filtered+=("${cfg}")
        break
      fi
    done
  done
  configs=("${filtered[@]}")
  if [[ ${#configs[@]} -eq 0 ]]; then
    echo "No unified configs matched --only '${ONLY}'." >&2
    exit 2
  fi
fi

IFS=',' read -r -a gpu_array <<< "${GPUS}"
pids=()
idx=0
for cfg in "${configs[@]}"; do
  gpu="${gpu_array[$((idx % ${#gpu_array[@]}))]}"
  gpu="$(echo "${gpu}" | xargs)"
  name="$(basename "${cfg}" .yaml)"
  log="runs/unified_multitask/${name}.log"
  echo "[unified] launch ${name} on GPU ${gpu}; log=${log}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" -u -m raw2task.train_unified_multitask \
    --config "${cfg}" 2>&1 | tee "${log}" &
  pids+=("$!")
  idx=$((idx + 1))
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

"${PYTHON_BIN}" -m raw2task.analyze_unified_results \
  --runs-root runs/unified_multitask \
  --out-dir runs/unified_multitask/paper_tables || true

exit "${status}"
