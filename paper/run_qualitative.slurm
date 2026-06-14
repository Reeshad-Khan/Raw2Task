#!/usr/bin/env bash
#SBATCH --job-name=r2t_qual
#SBATCH --output=runs/logs/r2t_qual_%j.out
#SBATCH --error=runs/logs/r2t_qual_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00

set -e
cd ~/Raw2Task
source /apps/anaconda/anaconda-2024.10/etc/profile.d/conda.sh
conda activate deeplens

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONPATH=/home/re141872/Raw2Task:$PYTHONPATH
export PYTHONUNBUFFERED=1

REPO=~/Raw2Task
KITTI_SPEC="$REPO/runs/kitti360_sfb4_spectral"
KITTI_BASE="$REPO/runs/kitti360_sfb4"
ACDC_SPEC="$REPO/runs/acdc_sfb4_spectral"
ACDC_BASE="$REPO/runs/acdc_sfb4"

echo "── Qualitative figures — KITTI-360"
python3 paper/generate_qualitative.py \
    --spectral-dir "$KITTI_SPEC" --base-dir "$KITTI_BASE" \
    --out-dir      "$REPO/paper/figures/qualitative_kitti" \
    --dataset kitti360 --n-figures 8 --max-scan-batches 120

echo "── Qualitative figures — ACDC"
python3 paper/generate_qualitative.py \
    --spectral-dir "$ACDC_SPEC" --base-dir "$ACDC_BASE" \
    --out-dir      "$REPO/paper/figures/qualitative_acdc" \
    --dataset acdc --n-figures 8 --max-scan-batches 120

echo "DONE — paper/figures/qualitative_*/"
ls "$REPO/paper/figures/"
