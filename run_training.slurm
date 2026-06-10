#!/bin/bash
#SBATCH --job-name=raw2task_train
#SBATCH --output=/home/re141872/Raw2Task/runs/raw2task_train_%j.out
#SBATCH --error=/home/re141872/Raw2Task/runs/raw2task_train_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1

source /apps/anaconda/anaconda-2024.10/etc/profile.d/conda.sh
conda activate deeplens

cd /home/re141872/Raw2Task

export PYTHONUNBUFFERED=1

python -u -m raw2task.train_extended \
  --config raw2task/configs/kitti360_review_base.yaml
