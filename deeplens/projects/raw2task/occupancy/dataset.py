"""Dataset loader for calibrated camera-to-voxel occupancy experiments."""

from __future__ import annotations

import os
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from deeplens.projects.raw2task.data.kitti360_seg import _resize_pair, _to_tensor_pil
from deeplens.projects.raw2task.occupancy.schema import OccupancyRecord, load_manifest, missing_files


DEFAULT_OCC_KEYS = ("semantics", "semantic", "labels", "label", "occupancy", "occ")
DEFAULT_MASK_KEYS = ("mask_camera", "camera_mask", "mask_lidar", "lidar_mask", "valid_mask", "valid")


def _load_npz_first(path: str, keys: Tuple[str, ...]) -> np.ndarray:
    with np.load(path) as data:
        for key in keys:
            if key in data:
                return np.asarray(data[key])
        available = ", ".join(data.files)
    raise KeyError(f"{path} does not contain any of keys {keys}. Available keys: {available}")


def _load_array(path: str, keys: Tuple[str, ...]) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npz":
        return _load_npz_first(path, keys)
    if ext == ".npy":
        return np.load(path)
    raise ValueError(f"Unsupported occupancy array file extension for {path!r}; expected .npz or .npy")


def _resize_volume_nearest(volume: torch.Tensor, target_shape: Tuple[int, int, int]) -> torch.Tensor:
    if tuple(volume.shape[-3:]) == tuple(target_shape):
        return volume
    x = volume[None, None].float()
    x = F.interpolate(x, size=target_shape, mode="nearest")
    return x[0, 0].long()


def _resize_mask_nearest(mask: torch.Tensor, target_shape: Tuple[int, int, int]) -> torch.Tensor:
    if tuple(mask.shape[-3:]) == tuple(target_shape):
        return mask.bool()
    x = mask[None, None].float()
    x = F.interpolate(x, size=target_shape, mode="nearest")
    return x[0, 0] > 0.5


class OccupancyManifestDataset(Dataset):
    """Read multi-camera images and dense voxel labels from a JSONL manifest.

    Each manifest row must contain a real voxel label array. The dataset refuses
    to synthesize occupancy from 2D masks, because that would not address the
    reviewers' 3D/occupancy concern.
    """

    def __init__(
        self,
        root: str,
        manifest: str,
        image_size: Tuple[int, int],
        grid_shape: Tuple[int, int, int],
        num_cameras: int = 6,
        ignore_index: int = 255,
        occupancy_keys: Tuple[str, ...] = DEFAULT_OCC_KEYS,
        mask_keys: Tuple[str, ...] = DEFAULT_MASK_KEYS,
        require_calibration: bool = True,
        require_camera_mask: bool = True,
        max_samples: int | None = None,
        stride: int = 1,
    ) -> None:
        self.root = os.path.expanduser(root)
        self.manifest = os.path.expanduser(manifest)
        self.image_size = tuple(int(x) for x in image_size)
        self.grid_shape = tuple(int(x) for x in grid_shape)
        self.num_cameras = int(num_cameras)
        self.ignore_index = int(ignore_index)
        self.occupancy_keys = tuple(occupancy_keys)
        self.mask_keys = tuple(mask_keys)
        self.require_calibration = bool(require_calibration)
        self.require_camera_mask = bool(require_camera_mask)

        if stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}")
        self.records = load_manifest(self.manifest, root=self.root)[::stride]
        if max_samples is not None and int(max_samples) > 0:
            self.records = self.records[: int(max_samples)]
        missing = missing_files(self.records)
        if missing:
            preview = "\n".join(missing[:20])
            raise RuntimeError(f"Occupancy manifest references missing files:\n{preview}")

        if self.require_calibration:
            bad = [
                rec.sample_id
                for rec in self.records[:100]
                if any(cam.intrinsics is None or cam.extrinsics is None for cam in rec.cameras)
            ]
            if bad:
                raise RuntimeError(
                    "Calibration is required but missing for samples: "
                    + ", ".join(bad[:10])
                )

    def __len__(self) -> int:
        return len(self.records)

    def _load_images(self, rec: OccupancyRecord) -> torch.Tensor:
        cams = rec.cameras[: self.num_cameras]
        if len(cams) < self.num_cameras:
            raise RuntimeError(
                f"sample {rec.sample_id} has {len(cams)} cameras, expected {self.num_cameras}"
            )
        imgs = []
        for cam in cams:
            img = Image.open(cam.image).convert("RGB")
            img, _ = _resize_pair(img, Image.new("L", img.size), self.image_size)
            imgs.append(_to_tensor_pil(img))
        return torch.stack(imgs, dim=0)

    def _load_calibration(self, rec: OccupancyRecord) -> Dict[str, torch.Tensor]:
        cams = rec.cameras[: self.num_cameras]
        intr = []
        extr = []
        for cam in cams:
            intr.append(torch.tensor(cam.intrinsics if cam.intrinsics is not None else np.eye(3), dtype=torch.float32))
            extr.append(torch.tensor(cam.extrinsics if cam.extrinsics is not None else np.eye(4), dtype=torch.float32))
        return {"intrinsics": torch.stack(intr, dim=0), "extrinsics": torch.stack(extr, dim=0)}

    def _load_occupancy(self, rec: OccupancyRecord) -> tuple[torch.Tensor, torch.Tensor]:
        occ_np = _load_array(rec.occupancy, self.occupancy_keys)
        occ = torch.from_numpy(occ_np.astype(np.int64, copy=False))
        if occ.ndim != 3:
            raise RuntimeError(f"sample {rec.sample_id}: occupancy must be 3D, got shape {tuple(occ.shape)}")
        occ = _resize_volume_nearest(occ, self.grid_shape)

        mask_path = rec.camera_mask or rec.lidar_mask
        if mask_path:
            mask_np = _load_array(mask_path, self.mask_keys)
            valid = torch.from_numpy(mask_np.astype(np.bool_, copy=False))
            valid = _resize_mask_nearest(valid, self.grid_shape)
        elif self.require_camera_mask:
            raise RuntimeError(f"sample {rec.sample_id}: camera/lidar visibility mask is required")
        else:
            valid = occ != self.ignore_index

        valid = valid & (occ != self.ignore_index)
        return occ.long(), valid.bool()

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]
        occ, valid = self._load_occupancy(rec)
        sample: Dict[str, Any] = {
            "sample_id": rec.sample_id,
            "images": self._load_images(rec),
            "occupancy": occ,
            "valid_mask": valid,
            "meta": rec.meta or {},
        }
        sample.update(self._load_calibration(rec))
        return sample


def occupancy_collate(batch: list[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "sample_id": [b["sample_id"] for b in batch],
        "images": torch.stack([b["images"] for b in batch], dim=0),
        "occupancy": torch.stack([b["occupancy"] for b in batch], dim=0),
        "valid_mask": torch.stack([b["valid_mask"] for b in batch], dim=0),
        "intrinsics": torch.stack([b["intrinsics"] for b in batch], dim=0),
        "extrinsics": torch.stack([b["extrinsics"] for b in batch], dim=0),
        "meta": [b.get("meta", {}) for b in batch],
    }
    return out

