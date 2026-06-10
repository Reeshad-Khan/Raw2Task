# Real Occupancy Assets

These configs are for real voxel occupancy supervision. They intentionally do
not derive occupancy from 2D semantic masks.

## KITTI-360 Only Path

For the current project scope, use KITTI-360 observed semantic occupancy. This
voxelizes the official KITTI-360 3D semantic PLY windows into camera-frame voxel
labels. This is real 3D supervision, but it is **observed occupancy**, not
hidden-space semantic scene completion.

Build KITTI-360 manifests and voxel labels:

```bash
python -m raw2task.occupancy.kitti360_builder \
  --root /home/rk010/Desktop/Research/NurIPS/KITTI-360 \
  --out-root data_external/kitti360_occupancy \
  --stride 20
```

Validate:

```bash
python -m raw2task.occupancy.validate_assets \
  --root data_external/kitti360_occupancy \
  --manifest data_external/kitti360_occupancy/manifests/val.jsonl \
  --out runs/occupancy/kitti360_asset_report_val.json
```

Train co-design and RGB baselines:

```bash
python -m raw2task.occupancy.train_occupancy \
  --config deeplens/projects/raw2task/configs/occupancy/kitti360_observed_codesign.yaml

python -m raw2task.occupancy.train_occupancy \
  --config deeplens/projects/raw2task/configs/occupancy/kitti360_observed_rgb.yaml
```

## Required Manifest Contract

Create `manifests/train.jsonl` and `manifests/val.jsonl` under the dataset root.
Each line must look like:

```json
{
  "sample_id": "scene-0001__token",
  "cameras": {
    "CAM_FRONT": {
      "image": "samples/CAM_FRONT/xxx.jpg",
      "intrinsics": [[...], [...], [...]],
      "extrinsics": [[...], [...], [...], [...]]
    }
  },
  "occupancy": "gts/scene-0001/token/labels.npz",
  "camera_mask": "gts/scene-0001/token/labels.npz",
  "meta": {"dataset": "occ3d-nuscenes"}
}
```

The occupancy `.npz` must contain one of:

- `semantics`
- `semantic`
- `labels`
- `label`
- `occupancy`
- `occ`

The visibility mask `.npz` must contain one of:

- `mask_camera`
- `camera_mask`
- `mask_lidar`
- `lidar_mask`
- `valid_mask`
- `valid`

Validate before training:

```bash
python -m raw2task.occupancy.validate_assets \
  --root /data/occ3d_nuscenes \
  --manifest /data/occ3d_nuscenes/manifests/val.jsonl \
  --out runs/occupancy/asset_report_val.json
```

Run co-design:

```bash
python -m raw2task.occupancy.train_occupancy \
  --config deeplens/projects/raw2task/configs/occupancy/occ3d_nuscenes_codesign.yaml
```

Run RGB baseline:

```bash
python -m raw2task.occupancy.train_occupancy \
  --config deeplens/projects/raw2task/configs/occupancy/occ3d_nuscenes_rgb.yaml
```
