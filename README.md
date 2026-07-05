# Raw2Task: Task-Driven Optics-Sensor-Model Co-Design

[![arXiv](https://img.shields.io/badge/arXiv-2606.24096-b31b1b.svg)](https://arxiv.org/abs/2606.24096)

**Raw2Task** is a research pipeline for jointly optimizing camera optics, sensor parameters, and perception models for downstream vision tasks (segmentation, occupancy, etc.) on driving data.

The core idea: instead of designing a camera and training a model separately, Raw2Task co-designs both end-to-end — the camera learns what to encode, the model learns what to decode, with the task metric as the objective.

## Pipeline

```
Scene → [trainable PSF] → [exposure] → [CFA] → [noise/quant] → Perception Model → Task Loss
              ↑                ↑           ↑           ↑
              └───────────────── jointly optimized ────┘
```

## Repository Structure

```
Raw2Task/
│
├── raw2task/               ← project code
│   ├── train_extended.py   (main training loop with co-design)
│   ├── run_end2end_pipeline.py
│   ├── run_review_matrix.py
│   ├── models/             (segmentation, occupancy, unified multitask)
│   ├── sensors/            (differentiable PSF, CFA, noise, optics)
│   ├── occupancy/          (3D occupancy prediction)
│   ├── configs/            (experiment configs)
│   └── scripts/            (experiment launch scripts)
│
├── deeplens/               ← differentiable optics library (dependency)
│   ├── geolens.py          (ray-tracing lens)
│   ├── diffraclens.py      (wave-optics lens)
│   ├── optics/
│   └── ...
│
└── deeplens_examples/      ← DeepLens standalone demos (not our project)
    ├── 0_hello_deeplens.py
    ├── lenses/
    └── ...
```

## Setup

```bash
conda env create -f environment.yml -n raw2task
conda activate raw2task
```

or

```bash
pip install -r requirements.txt
```

## Quick Start

Smoke test (verify imports):

```bash
python -m py_compile \
  raw2task/models/segmentation.py \
  raw2task/train_extended.py \
  raw2task/eval_robustness.py \
  raw2task/run_review_matrix.py \
  raw2task/run_end2end_pipeline.py
```

Single co-design run:

```bash
python -m raw2task.train_extended \
  --config raw2task/configs/kitti360_review_base.yaml
```

Reviewer matrix (multiple seeds, same split/metric):

```bash
python -m raw2task.run_review_matrix \
  --matrix raw2task/configs/kitti360_review_matrix.yaml
```

Full end-to-end pipeline (train → aggregate → robustness sweep → camera export):

```bash
python -m raw2task.run_end2end_pipeline \
  --matrix raw2task/configs/kitti360_review_matrix.yaml \
  --skip-existing
```

Full paper experiments:

```bash
raw2task/scripts/run_paper_experiments.sh
```

See [raw2task/README.md](raw2task/README.md) for the complete experiment guide.

## DeepLens

This project uses [DeepLens](https://github.com/vccimaging/AutoLens) as a differentiable optics backend. The `deeplens/` directory contains the library. Standalone DeepLens demos are in `deeplens_examples/`.

## Related Work

For broader context on end-to-end autonomous driving systems, see our survey:

> Reeshad Khan. *From Pixels to Policies: A Unified Survey of End-to-End Autonomous Driving Systems*. arXiv:2606.24096, 2026. [[paper]](https://arxiv.org/abs/2606.24096)
