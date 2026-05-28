#!/usr/bin/env bash
set -euo pipefail

# Fast, isolated experiment pass for iteration. It uses a reduced KITTI-360
# split/resolution and writes to runs/industry_paper_matrix_fast, leaving the
# final paper matrix untouched.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd -P)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export MPLBACKEND="${MPLBACKEND:-Agg}"
export PYTHONUNBUFFERED=1

PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"

python - <<'PY'
try:
    import torch
    print(f"[env] torch={torch.__version__}")
except Exception as exc:
    raise SystemExit(f"Environment check failed: PyTorch cannot be imported. Original error: {exc}")
PY

CMD=(
  "${PYTHON_BIN}" -u -m deeplens.projects.raw2task.run_paper_experiments
  --matrix deeplens/projects/raw2task/configs/industry_paper_matrix_fast.yaml
  --skip-existing
)

if [[ -n "${BATCH_PLAN:-}" ]]; then
  CMD+=(--batch-plan "${BATCH_PLAN}")
fi

echo "[run] ${CMD[*]} $*"
exec "${CMD[@]}" "$@"
