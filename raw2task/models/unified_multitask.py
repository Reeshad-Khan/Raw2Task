"""Unified 2D/3D/occupancy model for optics-sensor co-design experiments."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from raw2task.models.segmentation import ConvBNAct, DepthwiseSeparableBlock


class UnifiedRaw2TaskModel(nn.Module):
    """One shared perception trunk with 2D, 3D semantic, and occupancy heads."""

    def __init__(
        self,
        in_channels: int,
        num_2d_classes: int,
        num_3d_classes: int,
        grid_shape: Tuple[int, int, int],
        width: int = 64,
    ) -> None:
        super().__init__()
        self.grid_shape = tuple(int(x) for x in grid_shape)
        self.num_2d_classes = int(num_2d_classes)
        self.num_3d_classes = int(num_3d_classes)
        w = int(width)
        self.encoder = nn.Sequential(
            ConvBNAct(in_channels, w, 3, 2),
            DepthwiseSeparableBlock(w, 2 * w, 2),
            DepthwiseSeparableBlock(2 * w, 4 * w, 2),
            DepthwiseSeparableBlock(4 * w, 4 * w, 1),
        )
        self.context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(4 * w, 4 * w, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(4 * w, 4 * w, 1),
            nn.Sigmoid(),
        )
        self.seg2d_head = nn.Sequential(
            DepthwiseSeparableBlock(4 * w, 2 * w, 1),
            nn.Conv2d(2 * w, self.num_2d_classes, 1),
        )

        d, gh, gw = self.grid_shape
        self.seed_shape = (4 * w, max(1, d // 4), max(1, gh // 16), max(1, gw // 16))
        self.seed = nn.Sequential(
            nn.Linear(4 * w, 8 * w),
            nn.SiLU(inplace=True),
            nn.Linear(8 * w, int(torch.tensor(self.seed_shape).prod().item())),
        )
        self.voxel_decoder = nn.Sequential(
            nn.Conv3d(4 * w, 2 * w, 3, padding=1, bias=False),
            nn.BatchNorm3d(2 * w),
            nn.SiLU(inplace=True),
            nn.Conv3d(2 * w, 2 * w, 3, padding=1, bias=False),
            nn.BatchNorm3d(2 * w),
            nn.SiLU(inplace=True),
        )
        self.seg3d_head = nn.Conv3d(2 * w, self.num_3d_classes, 1)
        self.occupancy_head = nn.Conv3d(2 * w, 2, 1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(x)
        return feat * self.context(feat)

    def forward_2d(self, x: torch.Tensor, out_size: Tuple[int, int]) -> torch.Tensor:
        feat = self.encode(x)
        logits = self.seg2d_head(feat)
        if tuple(logits.shape[-2:]) != tuple(out_size):
            logits = F.interpolate(logits, size=out_size, mode="bilinear", align_corners=False)
        return logits

    def forward_voxel(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim == 5:
            b, n, c, h, w = x.shape
            flat = x.reshape(b * n, c, h, w)
            feat = self.encode(flat).mean(dim=(-2, -1)).reshape(b, n, -1).mean(dim=1)
        else:
            feat = self.encode(x).mean(dim=(-2, -1))
        seed = self.seed(feat).reshape(feat.shape[0], *self.seed_shape)
        vox = self.voxel_decoder(seed)
        seg3d = self.seg3d_head(vox)
        occ = self.occupancy_head(vox)
        if tuple(seg3d.shape[-3:]) != self.grid_shape:
            seg3d = F.interpolate(seg3d, size=self.grid_shape, mode="trilinear", align_corners=False)
            occ = F.interpolate(occ, size=self.grid_shape, mode="trilinear", align_corners=False)
        return seg3d, occ

