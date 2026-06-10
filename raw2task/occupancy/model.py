"""Reference camera-to-voxel occupancy model.

This is a production integration baseline, not a claimed SOTA occupancy model.
It is intentionally small and replaceable: any stronger Occ3D/OpenOccupancy
model can consume the same sensor-processed multi-camera tensor contract.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from raw2task.models.segmentation import ConvBNAct, DepthwiseSeparableBlock


class MultiCameraSensorFrontend(nn.Module):
    """Apply the optics-sensor frontend independently to each camera image."""

    def __init__(self, sensor: nn.Module) -> None:
        super().__init__()
        self.sensor = sensor

    @property
    def output_channels(self) -> int:
        return int(getattr(self.sensor, "output_channels", 3))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # images: (B, N, 3, H, W)
        b, n, c, h, w = images.shape
        flat = images.reshape(b * n, c, h, w)
        raw = self.sensor(flat)
        _, cr, hr, wr = raw.shape
        return raw.reshape(b, n, cr, hr, wr)


class OccupancyReferenceNet(nn.Module):
    """Camera-only voxel occupancy baseline with multi-camera feature fusion."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        grid_shape: Tuple[int, int, int],
        width: int = 64,
    ) -> None:
        super().__init__()
        self.grid_shape = tuple(int(x) for x in grid_shape)
        self.num_classes = int(num_classes)
        w = int(width)
        self.encoder = nn.Sequential(
            ConvBNAct(in_channels, w, 3, 2),
            DepthwiseSeparableBlock(w, 2 * w, 2),
            DepthwiseSeparableBlock(2 * w, 4 * w, 2),
            DepthwiseSeparableBlock(4 * w, 4 * w, 1),
        )
        self.camera_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(4 * w, 4 * w, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(4 * w, 4 * w, 1),
            nn.Sigmoid(),
        )
        d, gh, gw = self.grid_shape
        self.seed = nn.Sequential(
            nn.Linear(4 * w, 8 * w),
            nn.SiLU(inplace=True),
            nn.Linear(8 * w, 4 * w * max(1, d // 4) * max(1, gh // 16) * max(1, gw // 16)),
        )
        self.seed_shape = (4 * w, max(1, d // 4), max(1, gh // 16), max(1, gw // 16))
        self.decoder = nn.Sequential(
            nn.Conv3d(4 * w, 2 * w, 3, padding=1, bias=False),
            nn.BatchNorm3d(2 * w),
            nn.SiLU(inplace=True),
            nn.Conv3d(2 * w, 2 * w, 3, padding=1, bias=False),
            nn.BatchNorm3d(2 * w),
            nn.SiLU(inplace=True),
            nn.Conv3d(2 * w, self.num_classes, 1),
        )

    def forward(self, images: torch.Tensor, intrinsics=None, extrinsics=None) -> torch.Tensor:
        # images: (B, N, C, H, W). Calibration is accepted for API compatibility
        # with geometry-aware models; this reference model does not consume it.
        b, n, c, h, w = images.shape
        flat = images.reshape(b * n, c, h, w)
        feat = self.encoder(flat)
        feat = feat * self.camera_gate(feat)
        feat = feat.mean(dim=(-2, -1)).reshape(b, n, -1).mean(dim=1)
        seed = self.seed(feat).reshape(b, *self.seed_shape)
        logits = self.decoder(seed)
        if tuple(logits.shape[-3:]) != self.grid_shape:
            logits = F.interpolate(logits, size=self.grid_shape, mode="trilinear", align_corners=False)
        return logits

