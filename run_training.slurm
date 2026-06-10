#!/bin/bash
#SBATCH --job-name=raw2task
#SBATCH --output=/home/re141872/Raw2Task/runs/raw2task_%j.out
#SBATCH --error=/home/re141872/Raw2Task/runs/raw2task_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80

source /apps/anaconda/anaconda-2024.10/etc/profile.d/conda.sh
conda activate deeplens

cd /home/re141872/Raw2Task

export PYTHONUNBUFFERED=1

CONFIG="${1:-raw2task/configs/kitti360_review_base.yaml}"

echo "=== Job $SLURM_JOB_ID | Config: $CONFIG | Node: $SLURMD_NODENAME ==="
echo "=== GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) ==="

python -u -m raw2task.train_extended --config "$CONFIG"
