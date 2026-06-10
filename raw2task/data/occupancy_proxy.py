"""Occupancy proxy task: wraps a segmentation dataset, maps 19→3 classes."""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

OCCUPANCY_PROXY_CLASS_NAMES: List[str] = ["free", "static", "dynamic"]

OCCUPANCY_PROXY_PALETTE: Dict[int, Tuple[int, int, int]] = {
    0: (128, 64, 128),
    1: (70,  70,  70),
    2: (0,   0,  142),
}

_TRAINID_TO_OCC = np.full(256, 255, dtype=np.uint8)
for _i in [0, 1, 9]:
    _TRAINID_TO_OCC[_i] = 0  # free: road, sidewalk, terrain
for _i in [2, 3, 4, 5, 6, 7, 8, 10]:
    _TRAINID_TO_OCC[_i] = 1  # static: building, wall, fence, pole, lights, signs, veg, sky
for _i in [11, 12, 13, 14, 15, 16, 17, 18]:
    _TRAINID_TO_OCC[_i] = 2  # dynamic: person, rider, car, truck, bus, train, moto, bike


def trainid_to_occupancy_proxy_np(label: np.ndarray) -> np.ndarray:
    return _TRAINID_TO_OCC[label.astype(np.uint8)]


def occupancy_proxy_metadata() -> Dict:
    return {
        "class_names": OCCUPANCY_PROXY_CLASS_NAMES,
        "palette": OCCUPANCY_PROXY_PALETTE,
        "num_classes": 3,
    }


class OccupancyProxyDataset(Dataset):
    def __init__(self, base_dataset: Dataset, ignore_index: int = 255):
        self.base = base_dataset
        self.ignore_index = ignore_index

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img, lbl = self.base[idx]
        occ = trainid_to_occupancy_proxy_np(lbl.numpy().astype(np.uint8))
        if self.ignore_index != 255:
            occ[occ == 255] = self.ignore_index
        return img, torch.from_numpy(occ.copy()).long()
