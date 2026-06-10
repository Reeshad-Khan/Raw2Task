#!/bin/bash
#SBATCH --job-name=r2t
#SBATCH --output=/home/re141872/Raw2Task/runs/logs/%x_%j.out
#SBATCH --error=/home/re141872/Raw2Task/runs/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --exclude=evc[22,26,28-29,31,34,38-39,42,43,46,50]

source /apps/anaconda/anaconda-2024.10/etc/profile.d/conda.sh
conda activate deeplens

cd /home/re141872/Raw2Task
mkdir -p runs/logs

export PYTHONUNBUFFERED=1

EXP="${1:?Usage: sbatch run_experiment.slurm <experiment_name>}"
MATRIX="${2:-raw2task/configs/kitti360_review_matrix.yaml}"

echo "=== Job $SLURM_JOB_ID | Experiment: $EXP | Node: $SLURMD_NODENAME ==="
echo "=== GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) ==="
echo "=== Matrix: $MATRIX ==="

python -u -m raw2task.run_review_matrix \
  --matrix "$MATRIX" \
  --only "$EXP" \
  --skip-existing
