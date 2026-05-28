#!/usr/bin/env bash
set -euo pipefail

# DeepLens workspace cleanup helper.
# Default mode is dry-run and prints what would be removed.

ROOT_DEFAULT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
ROOT="$ROOT_DEFAULT"
DRY_RUN=1
PRUNE_LEGACY=0
REMOVE_LOCAL_DATASETS=0
KEEP_RUNS=1
QUIET=0

usage() {
  cat <<'EOF'
Usage:
  cleanup_workspace.sh [options]

Options:
  --root <path>               Workspace root (default: DeepLens repo root)
  --execute                   Actually delete files/directories
  --prune-legacy              Remove raw2task legacy/old code artifacts
  --remove-local-datasets     Remove local data/, data_eval/, datasets/BSDS300/, deeplens/data/
  --drop-runs                 Remove runs/ content (except active service artifacts)
  --quiet                     Less verbose output
  -h, --help                  Show this help

Examples:
  # Preview cleanup safely
  bash deeplens/projects/raw2task/scripts/cleanup_workspace.sh

  # Apply cleanup but keep runs and datasets
  bash deeplens/projects/raw2task/scripts/cleanup_workspace.sh --execute

  # Aggressive cleanup
  bash deeplens/projects/raw2task/scripts/cleanup_workspace.sh --execute --prune-legacy --drop-runs
EOF
}

log() {
  if [[ "$QUIET" -eq 0 ]]; then
    echo "$@"
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root) ROOT="$2"; shift 2 ;;
    --execute) DRY_RUN=0; shift ;;
    --prune-legacy) PRUNE_LEGACY=1; shift ;;
    --remove-local-datasets) REMOVE_LOCAL_DATASETS=1; shift ;;
    --drop-runs) KEEP_RUNS=0; shift ;;
    --quiet) QUIET=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ ! -d "$ROOT/.git" ]]; then
  echo "Error: '$ROOT' is not a git workspace root." >&2
  exit 2
fi

MODE="DRY-RUN"
if [[ "$DRY_RUN" -eq 0 ]]; then
  MODE="EXECUTE"
fi
log "[cleanup] root: $ROOT"
log "[cleanup] mode: $MODE"

ACTIVE_SERVICE=0
if command -v systemctl >/dev/null 2>&1; then
  if systemctl --user is-active deeplens-complete-experiments.service >/dev/null 2>&1; then
    ACTIVE_SERVICE=1
    log "[cleanup] detected active service: deeplens-complete-experiments.service"
  fi
fi

if [[ "$ACTIVE_SERVICE" -eq 1 && "$KEEP_RUNS" -eq 0 ]]; then
  echo "Refusing to drop runs while experiment service is active." >&2
  echo "Stop it first: systemctl --user stop deeplens-complete-experiments.service" >&2
  exit 3
fi

declare -a TARGETS=()

# Always-safe generated artifacts
while IFS= read -r p; do TARGETS+=("$p"); done < <(find "$ROOT" -type d \( -name "__pycache__" -o -name ".pytest_cache" -o -name ".ipynb_checkpoints" -o -name ".mypy_cache" -o -name ".ruff_cache" \) 2>/dev/null || true)
while IFS= read -r p; do TARGETS+=("$p"); done < <(find "$ROOT" -type f \( -name "*.pyc" -o -name "*.pyo" -o -name "*.bak" -o -name "*~" \) 2>/dev/null || true)

# Run/log cleanup
RUNS_DIR="$ROOT/runs"
if [[ -d "$RUNS_DIR" ]]; then
  while IFS= read -r p; do TARGETS+=("$p"); done < <(find "$RUNS_DIR" -maxdepth 1 -type f \( -name "complete_experiments_*.log" -o -name "nohup_test*" -o -name "systemd_test.log" \) 2>/dev/null || true)
  if [[ "$KEEP_RUNS" -eq 0 ]]; then
    while IFS= read -r p; do TARGETS+=("$p"); done < <(find "$RUNS_DIR" -mindepth 1 -maxdepth 1 -type d 2>/dev/null || true)
  fi
fi

# Optional legacy pruning for raw2task
RAW2TASK="$ROOT/deeplens/projects/raw2task"
if [[ "$PRUNE_LEGACY" -eq 1 && -d "$RAW2TASK" ]]; then
  [[ -d "$RAW2TASK/legacy" ]] && TARGETS+=("$RAW2TASK/legacy")
  [[ -f "$RAW2TASK/configs/raw2task_full_legacy.yml" ]] && TARGETS+=("$RAW2TASK/configs/raw2task_full_legacy.yml")
  [[ -f "$RAW2TASK/run_all_ablations_safe_serial.sh" ]] && TARGETS+=("$RAW2TASK/run_all_ablations_safe_serial.sh")
fi

# Optional local datasets purge
if [[ "$REMOVE_LOCAL_DATASETS" -eq 1 ]]; then
  [[ -d "$ROOT/data" ]] && TARGETS+=("$ROOT/data")
  [[ -d "$ROOT/data_eval" ]] && TARGETS+=("$ROOT/data_eval")
  [[ -d "$ROOT/deeplens/data" ]] && TARGETS+=("$ROOT/deeplens/data")
  [[ -d "$ROOT/datasets/BSDS300" ]] && TARGETS+=("$ROOT/datasets/BSDS300")
fi

# Deduplicate while preserving order
declare -A SEEN=()
declare -a UNIQUE=()
for p in "${TARGETS[@]}"; do
  [[ -e "$p" ]] || continue
  if [[ -z "${SEEN[$p]+x}" ]]; then
    SEEN["$p"]=1
    UNIQUE+=("$p")
  fi
done

if [[ ${#UNIQUE[@]} -eq 0 ]]; then
  log "[cleanup] nothing to remove."
  exit 0
fi

log "[cleanup] candidates: ${#UNIQUE[@]}"
if [[ "$QUIET" -eq 0 ]]; then
  for p in "${UNIQUE[@]}"; do
    echo "  - $p"
  done
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
  log "[cleanup] dry-run complete. Re-run with --execute to apply."
  exit 0
fi

for p in "${UNIQUE[@]}"; do
  rm -rf "$p"
done

# Remove empty directories left behind in raw2task tree.
if [[ -d "$RAW2TASK" ]]; then
  find "$RAW2TASK" -type d -empty -delete || true
fi

log "[cleanup] done."
