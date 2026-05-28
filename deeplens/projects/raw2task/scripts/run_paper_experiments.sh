#!/usr/bin/env bash
set -euo pipefail

# Production launcher for the paper experiment matrix.
# Defaults to reusing completed checkpoints instead of training from scratch.

SKIP_EXISTING=1
ONLY=""
SEEDS=""
MATRIX="${MATRIX:-deeplens/projects/raw2task/configs/industry_paper_matrix.yaml}"
BATCH_PLAN="${BATCH_PLAN:-}"
GPUS_ARG="${GPUS:-}"
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
    --only)
      ONLY="$2"
      shift 2
      ;;
    --seeds)
      SEEDS="$2"
      shift 2
      ;;
    --matrix)
      MATRIX="$2"
      shift 2
      ;;
    --batch-plan)
      BATCH_PLAN="$2"
      shift 2
      ;;
    --gpus)
      GPUS_ARG="$2"
      shift 2
      ;;
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --help|-h)
      cat <<'EOF'
Usage: deeplens/projects/raw2task/scripts/run_paper_experiments.sh [options]

Options:
  --skip-existing      Reuse completed run dirs containing last.pt (default)
  --no-skip-existing   Force all jobs to launch even if completed
  --only <csv>         Run only selected experiment names
  --seeds <csv>        Override seeds (e.g. 0,1,2)
  --matrix <path>      Override matrix YAML
  --batch-plan <path>  Fixed batch-size plan from probe_batch_size.py
  --gpus <csv>         GPU ids for concurrent job slots, e.g. 0,1
  --python <path>      Override Python executable
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
    raise SystemExit(
        "Environment check failed: PyTorch cannot be imported. "
        f"Original error: {exc}"
    )
PY

CMD=("${PYTHON_BIN}" -u -m deeplens.projects.raw2task.run_paper_experiments --matrix "${MATRIX}")
if [[ -n "${ONLY}" ]]; then
  CMD+=(--only "${ONLY}")
fi
if [[ -n "${SEEDS}" ]]; then
  CMD+=(--seeds "${SEEDS}")
fi
if [[ -n "${BATCH_PLAN}" ]]; then
  CMD+=(--batch-plan "${BATCH_PLAN}")
fi
if [[ -n "${GPUS_ARG}" ]]; then
  CMD+=(--gpus "${GPUS_ARG}")
fi
if [[ ${SKIP_EXISTING} -eq 1 ]]; then
  CMD+=(--skip-existing)
fi

echo "[run] ${CMD[*]}"
exec "${CMD[@]}" "${PASSTHROUGH[@]}"
