#!/usr/bin/env bash
set -euo pipefail

# Focused clean comparison:
#   GPU 0: latest idea, unified co-designed optics/sensor/task-token model.
#   GPU 1: strongest fair unified RGB baseline.
#
# This intentionally skips the larger ablation matrix so the first fresh
# evidence answers the core question: does the latest co-design idea work?

MODE="${MODE:-fast}"
GPUS="${GPUS:-${GPU_ID:-0,1}}"
PYTHON_BIN="${PYTHON_BIN:-/home/rk010/.conda/envs/raw2task/bin/python}"
KITTI_ROOT="${KITTI_ROOT:-/home/rk010/Desktop/Research/NurIPS/KITTI-360}"
OCCUPANCY_OUT_ROOT="${OCCUPANCY_OUT_ROOT:-data_external/kitti360_occupancy}"
OCCUPANCY_STRIDE="${OCCUPANCY_STRIDE:-20}"
OCCUPANCY_MAX_SAMPLES="${OCCUPANCY_MAX_SAMPLES:-400}"
FRESH=0

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
    --help|-h)
      cat <<'EOF'
Usage: deeplens/projects/raw2task/scripts/run_latest_vs_competitive.sh [options]

Options:
  --mode fast|full          Protocol to run (default: fast)
  --gpus <a,b>              Two GPUs: latest model uses first, baseline second
  --python <path>           Python executable
  --kitti-root <path>       KITTI-360 root
  --occupancy-out-root <p>  Observed voxel artifact root
  --occupancy-stride <n>    KITTI frame stride if voxel assets must be built
  --occupancy-max-samples <n>
                            Voxel build cap, 0 = all
  --fresh                   Delete latest-vs-competitive outputs first
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

mkdir -p runs/latest_vs_competitive
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export MPLBACKEND="${MPLBACKEND:-Agg}"
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

if [[ "${MODE}" != "fast" && "${MODE}" != "full" ]]; then
  echo "Unsupported --mode '${MODE}'. Expected fast or full." >&2
  exit 2
fi

if [[ ${FRESH} -eq 1 ]]; then
  rm -rf runs/latest_vs_competitive runs/unified_multitask
  mkdir -p runs/latest_vs_competitive runs/unified_multitask
fi

if [[ ! -f "${OCCUPANCY_OUT_ROOT}/manifests/train.jsonl" || ! -f "${OCCUPANCY_OUT_ROOT}/manifests/val.jsonl" ]]; then
  echo "[latest-vs-competitive] building observed KITTI-360 voxel assets -> ${OCCUPANCY_OUT_ROOT}"
  build_cmd=("${PYTHON_BIN}" -m raw2task.occupancy.kitti360_builder
    --root "${KITTI_ROOT}"
    --out-root "${OCCUPANCY_OUT_ROOT}"
    --stride "${OCCUPANCY_STRIDE}"
    --grid-shape 16 64 128
    --min-valid-voxels 20)
  if [[ "${OCCUPANCY_MAX_SAMPLES}" != "0" ]]; then
    build_cmd+=(--max-samples "${OCCUPANCY_MAX_SAMPLES}")
  fi
  "${build_cmd[@]}"
fi

IFS=',' read -r -a gpu_array <<< "${GPUS}"
if [[ ${#gpu_array[@]} -lt 2 ]]; then
  echo "Please provide two GPUs, e.g. --gpus 0,1" >&2
  exit 2
fi
GPU_LATEST="$(echo "${gpu_array[0]}" | xargs)"
GPU_BASELINE="$(echo "${gpu_array[1]}" | xargs)"

latest_cfg="deeplens/projects/raw2task/configs/unified/kitti360_unified_codesign_tasktokens_fast.yaml"
baseline_cfg="deeplens/projects/raw2task/configs/unified/kitti360_unified_rgb_fast.yaml"

if [[ "${MODE}" == "full" ]]; then
  gen_dir="runs/latest_vs_competitive/generated_full_configs"
  mkdir -p "${gen_dir}"
  mapfile -t generated < <("${PYTHON_BIN}" - "${gen_dir}" "${latest_cfg}" "${baseline_cfg}" <<'PY'
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
)
  latest_cfg="${generated[0]}"
  baseline_cfg="${generated[1]}"
fi

echo "[latest-vs-competitive] latest idea on GPU ${GPU_LATEST}: ${latest_cfg}"
echo "[latest-vs-competitive] competitive baseline on GPU ${GPU_BASELINE}: ${baseline_cfg}"

CUDA_VISIBLE_DEVICES="${GPU_LATEST}" "${PYTHON_BIN}" -u -m raw2task.train_unified_multitask \
  --config "${latest_cfg}" 2>&1 | tee "runs/latest_vs_competitive/latest_${MODE}.log" &
pid_latest="$!"

CUDA_VISIBLE_DEVICES="${GPU_BASELINE}" "${PYTHON_BIN}" -u -m raw2task.train_unified_multitask \
  --config "${baseline_cfg}" 2>&1 | tee "runs/latest_vs_competitive/baseline_rgb_${MODE}.log" &
pid_baseline="$!"

status=0
wait "${pid_latest}" || status=1
wait "${pid_baseline}" || status=1

"${PYTHON_BIN}" -m raw2task.analyze_unified_results \
  --runs-root runs/unified_multitask \
  --out-dir "runs/latest_vs_competitive/paper_tables_${MODE}" || true

exit "${status}"
