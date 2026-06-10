# RAW-to-Task Driving Segmentation Experiments

This project is the reviewer-facing experiment stack for task-driven
optics-sensor-model co-design on KITTI-360 style driving segmentation.

## What Is Actually Evaluated

KITTI-360 `data_2d_raw` images are rectified 8-bit RGB images, not genuine
sensor RAW. In this codebase the default `pipeline.input: sensor` experiment is
therefore a camera-simulation frontend applied to processed RGB used as a proxy
scene signal:

`processed RGB -> constrained trainable PSF -> exposure -> CFA -> noise/quantization -> segmentation`

Use this wording in the paper. Do not claim validation on a manufactured
lens+CFA+ADC camera unless you add real RAW/calibrated hardware experiments.

## What Changed For The Reviews

- Same split and same metric: all internal comparisons use KITTI-360 val mIoU
  from the same frame files, not leaderboard confidence-weighted mIoU.
- Quantitative ablations: optics, CFA, exposure, noise/quantization, frontend
  learning, and loss terms are separate named rows in the matrix.
- Real co-design claim: the default camera path jointly learns bounded
  field-dependent PSFs, exposure, CFA responses, read/shot noise magnitudes,
  and a bounded differentiable ADC bit-depth proxy, then freezes/export them
  to `camera_design_best.json` for inference.
- Robustness is numeric: blur, Gaussian noise, exposure shifts, bit depth,
  read noise, and shot noise are swept into CSV/JSON tables.
- Modern baselines: RGB baselines include LiteSeg, LR-ASPP MobileNetV3,
  DeepLabV3 MobileNetV3, and SegFormer-style models.
- Seeds and variance: matrix summaries are aggregated as mean/std across seeds.

## Main Commands

Smoke test:

```bash
python -m py_compile \
  raw2task/data/kitti360_seg.py \
  raw2task/models/segmentation.py \
  raw2task/train_extended.py \
  raw2task/eval_robustness.py \
  raw2task/run_review_matrix.py \
  raw2task/run_end2end_pipeline.py
```

Single full co-design run:

```bash
python -m raw2task.train_extended \
  --config raw2task/configs/kitti360_review_base.yaml
```

Reviewer matrix with same split, same metric, and multiple seeds:

```bash
python -m raw2task.run_review_matrix \
  --matrix raw2task/configs/kitti360_review_matrix.yaml
```

Complete pipeline: train matrix, aggregate paper tables, run selected
robustness sweeps, and collect camera design exports:

```bash
python -m raw2task.run_end2end_pipeline \
  --matrix raw2task/configs/kitti360_review_matrix.yaml \
  --skip-existing
```

Industry paper matrix with KITTI-360 component ablations, modern RGB baselines,
and read-only Cityscapes/ACDC experiments through the local WorldFactor mirror:

```bash
raw2task/scripts/run_paper_experiments.sh
```

Full paper run with resumable segmentation plus KITTI-360 observed 3D occupancy:

```bash
raw2task/scripts/run_full_paper_experiments.sh \
  2>&1 | tee runs/full_paper_experiments_live.log
```

This script resumes segmentation from saved checkpoints, builds/validates
KITTI-360 observed occupancy assets under `data_external/kitti360_occupancy`,
then runs co-designed, fixed-camera, and RGB occupancy experiments across
available GPUs.

Structured perception matrix for the second-task claim:

```bash
raw2task/scripts/run_paper_experiments.sh \
  --matrix raw2task/configs/structured_perception_matrix.yaml
```

This matrix adds an image-plane driving occupancy proxy derived from semantic
labels: `free_space`, `static_occupied`, and `dynamic_occupied`. It is useful
second-task evidence for optics-sensor co-design, but it should be reported as a
proxy structured-perception task, not as a calibrated 3D Occ3D-style benchmark.

Dry-run the full job layout before spending GPU time:

```bash
raw2task/scripts/run_paper_experiments.sh --dry-run --seeds 0
```

Run only a focused claim subset:

```bash
raw2task/scripts/run_paper_experiments.sh \
  --only kitti360_codesign_liteseg,kitti360_fixed_camera_frontend,kitti360_rgb_liteseg \
  --seeds 0,1,2
```

Force strictly sequential execution even on a multi-GPU machine:

```bash
raw2task/scripts/run_paper_experiments.sh --max-parallel 1
```

Robustness table for a trained checkpoint:

```bash
python -m raw2task.eval_robustness \
  --ckpt runs/kitti360_review_matrix/codesign_full_liteseg_seed0/best_epXX_miouY.pt \
  --out-dir runs/kitti360_review_matrix/codesign_full_liteseg_seed0/robustness
```

## Reviewer-Proof Experiment Matrix

The matrix config includes:

- `codesign_full_liteseg`: simulated optics + learned exposure + learned CFA + noise/quantization.
- `ablate_no_optics`: no lens/PSF frontend.
- `ablate_fixed_optics`: constrained PSF frontend present but frozen.
- `ablate_fixed_cfa`: CFA initialized from Bayer and frozen.
- `ablate_no_noise_quant`: no noise or quantization during training.
- `ablate_fixed_noise_quant`: read/shot noise and ADC bit depth present but frozen.
- `ablate_no_learned_exposure`: exposure gain frozen.
- `ablate_fixed_camera_frontend`: PSF, CFA, exposure, noise, and bit depth all frozen.
- `ablate_ce_only`: removes OHEM, Lovasz/Dice overlap, and smoothness terms.
- `rgb_liteseg`: same compact model family on processed RGB.
- `rgb_lraspp_mobilenetv3`: TorchVision LR-ASPP MobileNetV3 baseline.
- `rgb_deeplabv3_mobilenetv3`: TorchVision DeepLabV3 MobileNetV3 baseline.
- `rgb_segformer_b0_scratch`: local SegFormer-B0 style baseline through Transformers.

All rows use the same KITTI-360 train/val frame files when present:

- `data_2d_semantics/train/2013_05_28_drive_train_frames.txt`
- `data_2d_semantics/train/2013_05_28_drive_val_frames.txt`

This avoids mixing validation mIoU with leaderboard confidence-weighted mIoU.

## Multi-Dataset Evidence

`configs/industry_paper_matrix.yaml` adds Cityscapes and ACDC rows using
`data_external/worldfactor`, a symlink to `/home/rk010/projects/WorldFactor/data`.
The loader only reads the split files and images; it does not modify the
WorldFactor project. Cityscapes label IDs and ACDC train IDs are normalized into
the same 19-class Cityscapes trainId space used by KITTI-360.

`configs/structured_perception_matrix.yaml` reuses the same read-only data
mirror and camera frontend for a second task. It compares co-designed, fixed
camera, no-optics, and RGB frontends on KITTI-360 occupancy proxy labels, plus
Cityscapes/ACDC occupancy-proxy transfer rows.

The end-to-end runner writes paper-facing artifacts under
`runs/industry_paper_matrix/paper_tables/`:

- `main_results.csv` / `.md`: seed mean/std for mIoU, accuracy, params, latency.
- `robustness_summary.csv` / `.md`: quantitative corruption sweeps.
- `design_inventory.csv`: exported fixed camera designs for learned frontends.
- `main_results_miou.png`, `accuracy_vs_params.png`, `accuracy_vs_latency.png`.
- `pipeline_diagram.png`: camera-to-task schematic with fixed-at-inference note.
- `qualitative/`: curated validation panels showing RGB, camera stages, GT, prediction.

## Reporting Rules

Report dataset-level mIoU from `metrics_log.csv` or the matrix `summary.csv`.
If you include external leaderboard results, put them in a separate table and
label their split and metric explicitly.

For the paper table, aggregate seeds as mean and standard deviation:

```text
method | input | backbone | params | latency | mIoU mean +/- std
```

For robustness, use `robustness.csv` and plot mIoU versus corruption level for
blur, Gaussian noise, exposure shifts, bit depth, read noise, and shot noise.

For the optics/sensor claim, include `camera_design_best.json` in the artifact
bundle. It contains the fixed PSF coefficients/kernels, CFA spectral weights,
exposure gain, learned noise magnitudes, and learned deployable ADC bit depth
that were optimized during training and then frozen for inference.

## Claim Discipline For The Paper

Strong claim supported by this code:

```text
Under a processed-RGB proxy scene signal, a constrained optics-sensor frontend
with learned PSF, exposure, CFA, noise, and ADC parameters improves the
accuracy/efficiency trade-off over fixed-camera and RGB baselines when the
dataset split, metric, training loop, and lightweight task head are controlled.
The learned camera is exported as one fixed-at-inference design.
```

Do not claim real manufactured-camera validation from KITTI-360 alone. KITTI-360
and Cityscapes frames are already ISP-processed RGB. The correct phrase is
`physically constrained differentiable camera-simulation frontend` unless a real
RAW/calibrated hardware dataset is added.

The optics-sensor evidence should come from these artifacts:

- `main_results.csv`: compare full co-design against fixed-camera/frontend rows
  using the same model.
- `design_inventory.csv`: show nonzero initial-to-best deltas for PSF, CFA,
  exposure, noise, and ADC bit depth.
- `camera_design_initial.json` and `camera_design_best.json`: prove the learned
  parameters are frozen into a deployable design after training.
- `accuracy_vs_params.png` and `accuracy_vs_latency.png`: show the trade-off is
  not just a larger backbone effect.
- `qualitative/`: show visible camera-stage outputs and segmentation changes.

## Modern Backbone Hooks

`models/segmentation.py` supports:

- `unet_tiny`
- `liteseg`
- `lraspp_mobilenet_v3_large`
- `deeplabv3_mobilenet_v3_large`
- `deeplabv3_resnet50`
- `fcn_resnet50`
- `segformer`
- `hf_auto` for current Hugging Face semantic-segmentation checkpoints such as
  Mask2Former/OneFormer-style models

The registry is intentionally lazy: optional model-zoo dependencies do not
break the core raw-to-task experiments. Add newer 2025-2026 backbones behind
the same `build_segmentation_model()` interface so ablations and RGB baselines
remain apples-to-apples.
