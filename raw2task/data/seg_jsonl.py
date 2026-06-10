"""Segmentation dataset loaded from a JSONL manifest."""
from __future__ import annotations

import json
import random
from typing import List, Optional, Tuple

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset

from raw2task.data.kitti360_seg import _maybe_remap_to_trainid, _sanitize_trainid


class SegmentationJSONLDataset(Dataset):
    """Each JSONL line: {"image": "...", "label": "..."}"""

    def __init__(
        self,
        jsonl_path: str,
        split: str = "train",
        img_size: Tuple[int, int] = (512, 1024),
        label_encoding: str = "auto",
        max_samples: Optional[int] = None,
        stride: int = 1,
        photometric_jitter: float = 0.0,
        **kwargs,
    ):
        self.img_size = tuple(img_size)
        self.is_train = split == "train"
        self.label_encoding = label_encoding
        self.photometric_jitter = float(photometric_jitter)

        samples: List[Tuple[str, str]] = []
        with open(jsonl_path) as f:
            for i, line in enumerate(f):
                if i % max(1, stride) != 0:
                    continue
                rec = json.loads(line.strip())
                img = rec.get("image", rec.get("img", ""))
                lbl = rec.get("label", rec.get("seg", rec.get("mask", "")))
                samples.append((img, lbl))

        if max_samples and len(samples) > max_samples:
            samples = samples[:max_samples]
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        H, W = self.img_size
        img_path, lbl_path = self.samples[idx]
        image = Image.open(img_path).convert("RGB").resize((W, H), Image.BILINEAR)
        label = Image.open(lbl_path).resize((W, H), Image.NEAREST)

        if self.is_train and self.photometric_jitter > 0:
            j = self.photometric_jitter
            image = TF.adjust_brightness(image, 1.0 + random.uniform(-j, j))
            image = TF.adjust_contrast(image, 1.0 + random.uniform(-j, j))
            image = TF.adjust_saturation(image, 1.0 + random.uniform(-j, j))

        lbl_np = np.array(label, dtype=np.uint8)
        enc = self.label_encoding
        if enc == "auto":
            enc = "trainid" if lbl_np.max() < 20 else "original_id"
        lbl_np = _maybe_remap_to_trainid(lbl_np, enc)
        lbl_np = _sanitize_trainid(lbl_np)
        return TF.to_tensor(image), torch.from_numpy(lbl_np.copy()).long()
