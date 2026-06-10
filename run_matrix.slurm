#!/bin/bash
#SBATCH --job-name=raw2task_matrix
#SBATCH --output=/home/re141872/Raw2Task/runs/raw2task_matrix_%j.out
#SBATCH --error=/home/re141872/Raw2Task/runs/raw2task_matrix_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:2
#SBATCH --constraint=gpu80

source /apps/anaconda/anaconda-2024.10/etc/profile.d/conda.sh
conda activate deeplens

cd /home/re141872/Raw2Task
mkdir -p runs

export PYTHONUNBUFFERED=1

MATRIX="${1:-raw2task/configs/kitti360_review_matrix.yaml}"

echo "=== Job $SLURM_JOB_ID | Matrix: $MATRIX ==="
echo "=== GPUs: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) ==="

python -u -m raw2task.run_review_matrix \
  --matrix "$MATRIX" \
  --max-parallel 2
