#!/usr/bin/env bash
set -euo pipefail

# One-command launcher for:
#   1. segmentation paper matrix, resumed from saved checkpoints
#   2. KITTI-360 observed occupancy preprocessing/validation
#   3. occupancy co-design/fixed/RGB experiments, GPU-scheduled and resumable

SKIP_EXISTING=1
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
PASSTHROUGH=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-existing)
      SKIP_EXISTING=1
      shift
      ;;
    --no-skip-existing)
      SKIP_EXISTING=0
      shift
      ;;
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --help|-h)
      cat <<'EOF'
Usage: deeplens/projects/raw2task/scripts/run_full_paper_experiments.sh [options]

Common options:
  --skip-existing              Reuse completed runs and resume partial ones (default)
  --no-skip-existing           Launch even completed jobs
  --python <path>              Python executable
  --dry-run                    Print segmentation and occupancy jobs
  --max-parallel <n>           Max concurrent jobs; default one per visible GPU
  --gpus <csv>                 GPU ids, e.g. 0,1,2 or cpu

Segmentation options:
  --skip-segmentation
  --segmentation-matrix <yaml>
  --segmentation-seeds <csv>
  --segmentation-only <csv>

Occupancy options:
  --skip-occupancy
  --skip-occupancy-build       Reuse data_external/kitti360_occupancy manifests
  --occupancy-stride <n>       KITTI frame stride for voxel build, default 20
  --occupancy-max-samples <n>  Debug cap for train and val manifest build
  --occupancy-epochs <n>       Override occupancy epochs
EOF
      exit 0
      ;;
    *)
      PASSTHROUGH+=("$1")
      shift
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd -P)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export MPLBACKEND="${MPLBACKEND:-Agg}"
export PYTHONUNBUFFERED=1

python - <<'PY'
try:
    import torch
    print(f"[env] torch={torch.__version__}")
except Exception as exc:
    raise SystemExit(f"Environment check failed: PyTorch cannot be imported. Original error: {exc}")
PY

CMD=("${PYTHON_BIN}" -u -m raw2task.run_full_paper_experiments --python "${PYTHON_BIN}")
if [[ ${SKIP_EXISTING} -eq 1 ]]; then
  CMD+=(--skip-existing)
fi

echo "[run] ${CMD[*]} ${PASSTHROUGH[*]}"
exec "${CMD[@]}" "${PASSTHROUGH[@]}"

