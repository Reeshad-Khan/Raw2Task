#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# generate_all_paper_elements.sh
#
# Run this on Newton AFTER all experiments finish.
# Generates every figure, table, and qualitative result needed for the paper.
#
# Usage:
#   cd ~/Raw2Task
#   bash paper/generate_all_paper_elements.sh
#
# All outputs land in paper/figures/
# ─────────────────────────────────────────────────────────────────────────────
set -e

REPO=~/Raw2Task
PYTHON=$(which python3)

# ── Run dirs ──────────────────────────────────────────────────────────────────
KITTI_SPEC="$REPO/runs/kitti360_sfb4_spectral"
KITTI_BASE="$REPO/runs/kitti360_sfb4"
ACDC_SPEC="$REPO/runs/acdc_sfb4_spectral"
ACDC_BASE="$REPO/runs/acdc_sfb4"

# ── CFA weight dirs (best.pt checkpoints) ─────────────────────────────────────
TILE2_DIR="$KITTI_SPEC/ablate_no_optics_sfb4_seed0"
TILE3_DIR="$KITTI_SPEC/cfa_tile3_sfb4_seed0"
TILE4_DIR="$KITTI_SPEC/cfa_tile4_sfb4_seed0"

echo "======================================================================"
echo "Raw2Task — Paper Element Generator"
echo "$(date)"
echo "======================================================================"

# ── Step 1: Figures + LaTeX table ─────────────────────────────────────────────
echo ""
echo "── Step 1: Generating all matplotlib figures + LaTeX table ─────────────"
$PYTHON paper/make_figures.py --all \
    --kitti-spectral-dir "$KITTI_SPEC" \
    --kitti-base-dir     "$KITTI_BASE" \
    --acdc-spectral-dir  "$ACDC_SPEC"  \
    --acdc-base-dir      "$ACDC_BASE"  \
    --tile2-dir          "$TILE2_DIR"  \
    --tile3-dir          "$TILE3_DIR"  \
    --tile4-dir          "$TILE4_DIR"  \
    --latex

echo ""
echo "── Step 2: Qualitative figures — KITTI-360 ─────────────────────────────"
$PYTHON paper/generate_qualitative.py \
    --spectral-dir "$KITTI_SPEC" \
    --base-dir     "$KITTI_BASE" \
    --out-dir      "$REPO/paper/figures/qualitative_kitti" \
    --dataset kitti360 \
    --n-figures 8 \
    --max-scan-batches 120

echo ""
echo "── Step 3: Qualitative figures — ACDC ──────────────────────────────────"
$PYTHON paper/generate_qualitative.py \
    --spectral-dir "$ACDC_SPEC" \
    --base-dir     "$ACDC_BASE" \
    --out-dir      "$REPO/paper/figures/qualitative_acdc" \
    --dataset acdc \
    --n-figures 8 \
    --max-scan-batches 120

echo ""
echo "======================================================================"
echo "DONE — all paper elements saved to paper/figures/"
ls -la "$REPO/paper/figures/"
echo "======================================================================"
