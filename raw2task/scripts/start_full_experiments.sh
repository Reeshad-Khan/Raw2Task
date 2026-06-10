#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$ROOT_DIR"

MATRIX="${1:-deeplens/projects/raw2task/configs/kitti360_review_matrix.yaml}"
PYTHON_BIN="${PYTHON_BIN:-/opt/anaconda3/bin/python}"
RUNS_DIR="${RUNS_DIR:-runs}"
mkdir -p "$RUNS_DIR"

TS="$(date +%Y%m%d_%H%M%S)"
LOG_PATH="$RUNS_DIR/complete_experiments_${TS}.log"
echo "$LOG_PATH" > "$RUNS_DIR/complete_experiments.logpath"

systemd-run --user --unit=deeplens-complete-experiments --same-dir --collect \
  --setenv=CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  --setenv=MPLBACKEND=Agg \
  --setenv=PYTHONUNBUFFERED=1 \
  /bin/bash -lc "echo \$\$ > $RUNS_DIR/complete_experiments.pid; exec \"$PYTHON_BIN\" -u -m raw2task.run_end2end_pipeline --matrix \"$MATRIX\" --skip-existing --force-robustness > \"$LOG_PATH\" 2>&1"

MAIN_PID="$(systemctl --user show -p MainPID --value deeplens-complete-experiments.service)"
echo "$MAIN_PID" > "$RUNS_DIR/complete_experiments.pid"

cat <<EOF
Started experiment service.
Unit: deeplens-complete-experiments.service
Main PID: $MAIN_PID
Log: $LOG_PATH

Monitor:
  systemctl --user status deeplens-complete-experiments.service --no-pager
  tail -f "$LOG_PATH"
  watch -n 20 nvidia-smi
EOF
