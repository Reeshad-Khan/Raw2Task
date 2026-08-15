#!/bin/bash
# Launches the scoped WACV-resubmission experiment set:
#   - 6 jobs: seed-variance (seeds 1,2) on the 3 headline KITTI-360 configs
#     (codesign_sfb4, ablate_fixed_camera_sfb4, ablate_no_optics_sfb4)
#   - 2 jobs: seed-variance (seeds 1,2) on cfa_only_sfb4 (the RKV7/dMx8
#     fixed-camera-vs-CFA-only isolation ask)
#   - 2 jobs: backbone transfer on SegFormer-B2 (ablate_fixed_camera_frontend_sfb2,
#     ablate_no_optics_sfb2) -- the two configs that matter most for the paper's
#     model-agnostic claim
#
# Each job is a single seed on one H100 (~11h observed for SFB4 full runs),
# so all 10 jobs are independent and can run fully in parallel given capacity.
#
# Usage: bash submit_wacv_experiments.sh

cd "$(dirname "$0")"

echo "Submitting seed-variance jobs (SFB4 core, seed1)..."
for exp in codesign_sfb4 ablate_fixed_camera_sfb4 ablate_no_optics_sfb4; do
  job_id=$(sbatch --parsable run_experiment.slurm "$exp" raw2task/configs/kitti360_sfb4_v3_seed1_matrix.yaml)
  echo "  $exp (seed1) -> job $job_id"
done

echo "Submitting seed-variance jobs (SFB4 core, seed2)..."
for exp in codesign_sfb4 ablate_fixed_camera_sfb4 ablate_no_optics_sfb4; do
  job_id=$(sbatch --parsable run_experiment.slurm "$exp" raw2task/configs/kitti360_sfb4_v3_seed2_matrix.yaml)
  echo "  $exp (seed2) -> job $job_id"
done

echo "Submitting seed-variance jobs (cfa_only, seed1 + seed2)..."
job_id=$(sbatch --parsable run_experiment.slurm cfa_only_sfb4 raw2task/configs/kitti360_sfb4_spectral_seed1_matrix.yaml)
echo "  cfa_only_sfb4 (seed1) -> job $job_id"
job_id=$(sbatch --parsable run_experiment.slurm cfa_only_sfb4 raw2task/configs/kitti360_sfb4_spectral_seed2_matrix.yaml)
echo "  cfa_only_sfb4 (seed2) -> job $job_id"

echo "Submitting backbone-transfer jobs (SegFormer-B2)..."
for exp in ablate_fixed_camera_frontend_sfb2 ablate_no_optics_sfb2; do
  job_id=$(sbatch --parsable run_experiment.slurm "$exp" raw2task/configs/kitti360_sfb2_matrix.yaml)
  echo "  $exp -> job $job_id"
done

echo ""
echo "All 10 jobs submitted. Monitor with: watch -n 30 squeue -u re141872"
