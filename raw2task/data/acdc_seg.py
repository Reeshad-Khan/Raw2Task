"""ACDC (Adverse Conditions Dataset) semantic segmentation dataset.

Layout::
    root/rgb_anon/<condition>/<split>/<sequence>/<frame>_leftImg8bit.png
    root/gt/<condition>/<split>/<sequence>/<frame>_gtFine_labelTrainIds.png

Labels are already in Cityscapes trainId format (0-18, 255=ignore).
Conditions: fog, night, rain, snow.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset

TRAIN_ID_CLASS_NAMES: List[str] = [
    "road", "sidewalk", "building", "wall", "fence",
    "pole", "traffic light", "traffic sign", "vegetation", "terrain",
    "sky", "person", "rider", "car", "truck",
    "bus", "train", "motorcycle", "bicycle",
]

NUM_CLASSES  = 19
IGNORE_INDEX = 255
ALL_CONDITIONS = ("fog", "night", "rain", "snow")


def _sanitize_trainid(label: np.ndarray) -> np.ndarray:
    out = label.astype(np.int32)
    out[(out < 0) | ((out >= NUM_CLASSES) & (out != IGNORE_INDEX))] = IGNORE_INDEX
    return out.astype(np.uint8)


class ACDCSeg(Dataset):
    """ACDC semantic segmentation dataset.

    Args:
        root:               Path to the ACDC root directory.
        split:              "train", "val", or "test".
        img_size:           (H, W) to resize images and labels to.
        conditions:         Subset of ("fog", "night", "rain", "snow").
                            None means all four conditions.
        stride:             Keep every nth sample (1 = keep all).
        photometric_jitter: Max brightness/contrast/saturation jitter
                            fraction applied during training.
        use_ref:            If True, load the paired normal-condition
                            reference image instead of the adverse image.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        img_size: Tuple[int, int] = (512, 1024),
        conditions: Optional[Sequence[str]] = None,
        stride: int = 1,
        photometric_jitter: float = 0.0,
        use_ref: bool = False,
        max_samples: Optional[int] = None,
    ):
        self.root       = Path(root)
        self.split      = split
        self.img_size   = tuple(img_size)
        self.stride     = max(1, int(stride))
        self.photometric_jitter = float(photometric_jitter)
        self.is_train   = (split == "train")
        self.use_ref    = use_ref
        self.conditions = list(conditions) if conditions else list(ALL_CONDITIONS)

        self.samples: List[Tuple[Path, Path, str]] = self._collect()

        if max_samples and len(self.samples) > max_samples:
            step = max(1, len(self.samples) // max_samples)
            self.samples = self.samples[::step][:max_samples]

    def _collect(self) -> List[Tuple[Path, Path, str]]:
        samples: List[Tuple[Path, Path, str]] = []

        for cond in self.conditions:
            if self.use_ref:
                img_base = self.root / "rgb_anon" / "ref" / cond / self.split
                gt_base  = self.root / "gt"      / "ref" / cond / self.split
            else:
                img_base = self.root / "rgb_anon" / cond / self.split
                gt_base  = self.root / "gt"       / cond / self.split

            if not img_base.exists():
                continue

            for i, lf in enumerate(sorted(gt_base.rglob("*_labelTrainIds.png"))):
                if i % self.stride != 0:
                    continue
                # derive image path: replace gt → rgb_anon, remove _gtFine_labelTrainIds suffix
                rel      = lf.relative_to(gt_base)
                img_name = rel.name.replace("_gtFine_labelTrainIds.png", "_leftImg8bit.png")
                img_path = img_base / rel.parent / img_name
                if img_path.exists():
                    samples.append((img_path, lf, cond))

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, lbl_path, _ = self.samples[idx]
        H, W = self.img_size

        image = Image.open(img_path).convert("RGB").resize((W, H), Image.BILINEAR)
        label = Image.open(lbl_path).resize((W, H), Image.NEAREST)

        if self.is_train:
            # random horizontal flip
            if random.random() < 0.5:
                image = TF.hflip(image)
                label = TF.hflip(label)

            if self.photometric_jitter > 0:
                j = self.photometric_jitter
                image = TF.adjust_brightness(image, 1.0 + random.uniform(-j, j))
                image = TF.adjust_contrast(image,   1.0 + random.uniform(-j, j))
                image = TF.adjust_saturation(image, 1.0 + random.uniform(-j, j))

        img_tensor = TF.to_tensor(image)
        lbl_np     = _sanitize_trainid(np.array(label, dtype=np.uint8))
        return img_tensor, torch.from_numpy(lbl_np.copy()).long()

    def condition_of(self, idx: int) -> str:
        return self.samples[idx][2]
