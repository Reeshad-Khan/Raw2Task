"""KITTI-360 perspective-camera semantic segmentation dataset."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset

_ORIGINAL_TO_TRAINID: Dict[int, int] = {
    0: 255, 1: 255, 2: 255, 3: 255, 4: 255, 5: 255, 6: 255,
    7: 0,   8: 1,   9: 255, 10: 255, 11: 2,  12: 3,  13: 4,
    14: 255, 15: 255, 16: 255, 17: 5, 18: 255, 19: 6, 20: 7,
    21: 8,  22: 9,  23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
    28: 15, 29: 255, 30: 255, 31: 16, 32: 17, 33: 18,
}

_LOOKUP = np.full(256, 255, dtype=np.uint8)
for _k, _v in _ORIGINAL_TO_TRAINID.items():
    if 0 <= _k < 256:
        _LOOKUP[_k] = _v

TRAIN_ID_CLASS_NAMES: List[str] = [
    "road", "sidewalk", "building", "wall", "fence",
    "pole", "traffic light", "traffic sign", "vegetation", "terrain",
    "sky", "person", "rider", "car", "truck",
    "bus", "train", "motorcycle", "bicycle",
]

NUM_CLASSES = 19
IGNORE_INDEX = 255
_DEFAULT_VAL_SEQS = {"2013_05_28_drive_0010_sync"}


def _remap_original_id_to_trainid(label: np.ndarray) -> np.ndarray:
    return _LOOKUP[label.astype(np.uint8)]


def _maybe_remap_to_trainid(label: np.ndarray, label_encoding: str = "original_id") -> np.ndarray:
    if label_encoding == "original_id":
        return _remap_original_id_to_trainid(label)
    return label.astype(np.uint8)


def _sanitize_trainid(label: np.ndarray) -> np.ndarray:
    out = label.astype(np.int32)
    out[(out < 0) | ((out >= NUM_CLASSES) & (out != IGNORE_INDEX))] = IGNORE_INDEX
    return out.astype(np.uint8)


class KITTI360Seg(Dataset):
    """
    Layout::
        root/data_2d_raw/<seq>/<camera>/data_rect/<frame>.png
        root/data_2d_semantics/train/<seq>/<camera>/semantic/<frame>.png
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        img_size: Tuple[int, int] = (384, 1280),
        camera: str = "image_00",
        sequences: Optional[Sequence[str]] = None,
        split_file: Optional[str] = None,
        label_encoding: str = "original_id",
        max_samples: Optional[int] = None,
        stride: int = 1,
        photometric_jitter: float = 0.0,
        viewpoint_jitter: Optional[Dict] = None,
    ):
        self.root = Path(root)
        self.split = split
        self.img_size = tuple(img_size)
        self.camera = camera
        self.label_encoding = label_encoding
        self.stride = max(1, int(stride))
        self.photometric_jitter = float(photometric_jitter)
        self.is_train = split == "train"

        if split_file and Path(split_file).exists():
            self.samples = self._load_split_file(split_file)
        else:
            self.samples = self._collect(sequences)

        if max_samples and len(self.samples) > max_samples:
            step = max(1, len(self.samples) // max_samples)
            self.samples = self.samples[::step][:max_samples]

    def _load_split_file(self, path: str) -> List[Tuple[Path, Path]]:
        samples = []
        with open(path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    img, lbl = Path(parts[0]), Path(parts[1])
                    if img.exists() and lbl.exists():
                        samples.append((img, lbl))
        return samples

    def _collect(self, sequences: Optional[Sequence[str]]) -> List[Tuple[Path, Path]]:
        sem_root = self.root / "data_2d_semantics" / "train"
        img_root = self.root / "data_2d_raw"

        if not sem_root.exists():
            raise FileNotFoundError(f"Semantics dir not found: {sem_root}")

        all_seqs = {d.name for d in sem_root.iterdir() if d.is_dir()}
        if sequences is not None:
            seq_set = set(sequences)
        elif self.split == "val":
            seq_set = all_seqs & _DEFAULT_VAL_SEQS
            if not seq_set:
                seq_set = {sorted(all_seqs)[-1]}
        else:
            seq_set = all_seqs - _DEFAULT_VAL_SEQS
            if not seq_set:
                seq_set = all_seqs

        samples: List[Tuple[Path, Path]] = []
        for seq in sorted(seq_set):
            label_dir = sem_root / seq / self.camera / "semantic"
            image_dir = img_root / seq / self.camera / "data_rect"
            if not label_dir.exists() or not image_dir.exists():
                continue
            for i, lf in enumerate(sorted(label_dir.glob("*.png"))):
                if i % self.stride != 0:
                    continue
                img_file = image_dir / lf.name
                if img_file.exists():
                    samples.append((img_file, lf))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, lbl_path = self.samples[idx]
        H, W = self.img_size

        image = Image.open(img_path).convert("RGB").resize((W, H), Image.BILINEAR)
        label = Image.open(lbl_path).resize((W, H), Image.NEAREST)

        if self.is_train and self.photometric_jitter > 0:
            j = self.photometric_jitter
            image = TF.adjust_brightness(image, 1.0 + random.uniform(-j, j))
            image = TF.adjust_contrast(image, 1.0 + random.uniform(-j, j))
            image = TF.adjust_saturation(image, 1.0 + random.uniform(-j, j))

        img_tensor = TF.to_tensor(image)
        lbl_np = np.array(label, dtype=np.uint8)
        lbl_np = _maybe_remap_to_trainid(lbl_np, self.label_encoding)
        lbl_np = _sanitize_trainid(lbl_np)
        return img_tensor, torch.from_numpy(lbl_np.copy()).long()
