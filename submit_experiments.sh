#!/bin/bash
# Submit one Slurm job per experiment. Each job runs all 3 seeds on 1 H100.
# Usage:
#   bash submit_experiments.sh                         # all 13 experiments
#   bash submit_experiments.sh codesign ablate_no_optics  # specific ones

cd "$(dirname "$0")"

ALL_EXPERIMENTS=(
  codesign_full_liteseg
  ablate_no_optics
  ablate_fixed_optics
  ablate_fixed_cfa
  ablate_no_noise_quant
  ablate_fixed_noise_quant
  ablate_no_learned_exposure
  ablate_fixed_camera_frontend
  ablate_ce_only
  rgb_liteseg
  rgb_lraspp_mobilenetv3
  rgb_deeplabv3_mobilenetv3
  rgb_segformer_b0_scratch
)

MATRIX="${MATRIX:-raw2task/configs/kitti360_review_matrix.yaml}"

if [ $# -gt 0 ]; then
  EXPERIMENTS=("$@")
else
  EXPERIMENTS=("${ALL_EXPERIMENTS[@]}")
fi

echo "Submitting ${#EXPERIMENTS[@]} experiment(s) on H100 (gpu80)..."
echo "Matrix: $MATRIX"
echo ""

for exp in "${EXPERIMENTS[@]}"; do
  job_id=$(sbatch --parsable run_experiment.slurm "$exp" "$MATRIX")
  echo "  Submitted $exp → job $job_id"
done

echo ""
echo "Monitor with: watch -n 30 squeue -u re141872"
