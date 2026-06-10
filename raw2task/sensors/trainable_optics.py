"""Constrained differentiable optics for RAW-to-task co-design.

This module is deliberately more conservative than an unconstrained image blur:
the learned optics are represented as a small set of field-dependent,
energy-normalized PSFs with bounded low-order aberration coefficients. The
coefficients are optimized during training, then exported as a fixed camera
design for inference.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


COEFF_NAMES = ("defocus", "astig_x", "astig_y", "coma_x", "coma_y", "spherical")


def _zone_centers(num_zones: int) -> torch.Tensor:
    if num_zones <= 1:
        return torch.tensor([[0.0, 0.0]], dtype=torch.float32)
    if num_zones == 5:
        return torch.tensor(
            [
                [0.0, 0.0],
                [-0.75, -0.75],
                [0.75, -0.75],
                [-0.75, 0.75],
                [0.75, 0.75],
            ],
            dtype=torch.float32,
        )
    if num_zones == 9:
        vals = [-0.8, 0.0, 0.8]
        return torch.tensor([[x, y] for y in vals for x in vals], dtype=torch.float32)

    # Fallback: one center zone and a uniformly spaced off-axis ring.
    centers = [[0.0, 0.0]]
    for i in range(num_zones - 1):
        theta = 2.0 * math.pi * i / max(num_zones - 1, 1)
        centers.append([0.8 * math.cos(theta), 0.8 * math.sin(theta)])
    return torch.tensor(centers, dtype=torch.float32)


def _atanh_safe(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(-0.95, 0.95)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


class ConstrainedFieldPSF(nn.Module):
    """Low-order trainable PSF bank with smooth field blending.

    Parameters are intentionally small and bounded. Each field zone has a
    per-channel coefficient vector that controls blur width, astigmatic
    anisotropy, coma-like centroid shift, and a weak spherical term. The
    generated kernels are nonnegative and sum to one.
    """

    def __init__(
        self,
        num_zones: int = 5,
        channels: int = 3,
        kernel_size: int = 15,
        base_sigma_px: float = 1.05,
        min_sigma_px: float = 0.35,
        max_sigma_px: float = 3.75,
        max_shift_px: float = 1.25,
        field_sigma: float = 0.65,
        coeff_bounds: List[float] | Tuple[float, ...] | None = None,
        init_field_aberration: bool = True,
        padding_mode: str = "reflect",
        regularization_weights: Dict[str, float] | None = None,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd so PSFs have a center pixel.")
        if channels <= 0:
            raise ValueError("channels must be positive.")

        self.num_zones = int(num_zones)
        self.channels = int(channels)
        self.kernel_size = int(kernel_size)
        self.radius = int(kernel_size // 2)
        self.base_sigma_px = float(base_sigma_px)
        self.min_sigma_px = float(min_sigma_px)
        self.max_sigma_px = float(max_sigma_px)
        self.max_shift_px = float(max_shift_px)
        self.field_sigma = float(field_sigma)
        self.padding_mode = str(padding_mode)
        self.regularization_weights = {
            "psf_second_moment": 1.0,
            "field_smoothness": 0.25,
            "chromatic_smoothness": 0.10,
            "coma_shift": 0.25,
            "kernel_tv": 0.02,
        }
        if regularization_weights:
            self.regularization_weights.update({k: float(v) for k, v in regularization_weights.items()})

        bounds = coeff_bounds or [1.0, 0.75, 0.75, max_shift_px, max_shift_px, 0.6]
        if len(bounds) != len(COEFF_NAMES):
            raise ValueError(f"coeff_bounds must have {len(COEFF_NAMES)} entries.")
        self.register_buffer("coeff_bounds", torch.tensor(bounds, dtype=torch.float32).view(1, 1, -1))
        self.register_buffer("zone_centers", _zone_centers(self.num_zones), persistent=False)

        self.coeff_raw = nn.Parameter(torch.zeros(self.num_zones, self.channels, len(COEFF_NAMES)))
        if init_field_aberration:
            self._init_field_aberration()

        self._field_weight_cache: Dict[Tuple[int, int, str, str], torch.Tensor] = {}

    @classmethod
    def from_config(cls, lens_cfg: Dict[str, Any]) -> "ConstrainedFieldPSF":
        psf_cfg = lens_cfg.get("trainable_psf", {}) or {}
        reg_cfg = psf_cfg.get("regularization", {}) or {}
        return cls(
            num_zones=int(psf_cfg.get("num_zones", lens_cfg.get("num_zones", 5))),
            channels=int(psf_cfg.get("channels", 3)),
            kernel_size=int(psf_cfg.get("kernel_size", 15)),
            base_sigma_px=float(psf_cfg.get("base_sigma_px", 1.05)),
            min_sigma_px=float(psf_cfg.get("min_sigma_px", 0.35)),
            max_sigma_px=float(psf_cfg.get("max_sigma_px", 3.75)),
            max_shift_px=float(psf_cfg.get("max_shift_px", 1.25)),
            field_sigma=float(psf_cfg.get("field_sigma", 0.65)),
            coeff_bounds=psf_cfg.get("coeff_bounds", None),
            init_field_aberration=bool(psf_cfg.get("init_field_aberration", True)),
            padding_mode=str(psf_cfg.get("padding_mode", "reflect")),
            regularization_weights=reg_cfg,
        )

    def _init_field_aberration(self) -> None:
        centers = self.zone_centers
        target = torch.zeros_like(self.coeff_raw.data)
        chromatic = torch.linspace(-0.06, 0.06, self.channels).view(1, self.channels)
        max_r = math.sqrt(2.0)
        for z, (cx, cy) in enumerate(centers.tolist()):
            r = math.sqrt(cx * cx + cy * cy) / max_r
            target[z, :, 0] = 0.20 * r + chromatic[0]  # defocus
            target[z, :, 1] = 0.12 * (cx * cx - cy * cy)  # astig x
            target[z, :, 2] = 0.20 * cx * cy  # astig y
            target[z, :, 3] = 0.16 * cx * r  # coma x
            target[z, :, 4] = 0.16 * cy * r  # coma y
            target[z, :, 5] = 0.08 * r * r  # spherical

        bounds = self.coeff_bounds.to(target.device)
        self.coeff_raw.data.copy_(_atanh_safe(target / bounds))

    def effective_coefficients(self) -> torch.Tensor:
        return torch.tanh(self.coeff_raw) * self.coeff_bounds.to(self.coeff_raw.device, self.coeff_raw.dtype)

    def psf_kernels(
        self,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        coeff_override: torch.Tensor | None = None,
    ) -> torch.Tensor:
        coeff = self.effective_coefficients() if coeff_override is None else coeff_override
        if device is None:
            device = coeff.device
        if dtype is None:
            dtype = coeff.dtype
        coeff = coeff.to(device=device, dtype=dtype)

        coords = torch.arange(self.kernel_size, device=device, dtype=dtype) - float(self.radius)
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        xx = xx.view(1, 1, self.kernel_size, self.kernel_size)
        yy = yy.view(1, 1, self.kernel_size, self.kernel_size)

        defocus = coeff[..., 0]
        astig_x = coeff[..., 1]
        astig_y = coeff[..., 2]
        coma_x = coeff[..., 3].clamp(-self.max_shift_px, self.max_shift_px)
        coma_y = coeff[..., 4].clamp(-self.max_shift_px, self.max_shift_px)
        spherical = coeff[..., 5]

        astig_mag = torch.sqrt(astig_x.square() + astig_y.square() + 1e-8)
        theta = 0.5 * torch.atan2(astig_y, astig_x + 1e-8)
        sigma_center = self.base_sigma_px + defocus + 0.35 * spherical
        sigma_x = (sigma_center + astig_mag).clamp(self.min_sigma_px, self.max_sigma_px)
        sigma_y = (sigma_center - astig_mag).clamp(self.min_sigma_px, self.max_sigma_px)

        x = xx - coma_x[..., None, None]
        y = yy - coma_y[..., None, None]
        cos_t = torch.cos(theta)[..., None, None]
        sin_t = torch.sin(theta)[..., None, None]
        xr = cos_t * x + sin_t * y
        yr = -sin_t * x + cos_t * y

        quad = xr.square() / sigma_x[..., None, None].square().clamp_min(1e-8)
        quad = quad + yr.square() / sigma_y[..., None, None].square().clamp_min(1e-8)
        kernels = torch.exp(-0.5 * quad)
        kernels = kernels / kernels.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
        return kernels

    def _field_weights(self, height: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (int(height), int(width), str(device), str(dtype))
        cached = self._field_weight_cache.get(key)
        if cached is not None:
            return cached

        xs = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
        ys = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        centers = self.zone_centers.to(device=device, dtype=dtype)
        dx = xx.unsqueeze(0) - centers[:, 0].view(-1, 1, 1)
        dy = yy.unsqueeze(0) - centers[:, 1].view(-1, 1, 1)
        dist2 = dx.square() + dy.square()
        weights = torch.exp(-dist2 / max(2.0 * self.field_sigma * self.field_sigma, 1e-6))
        weights = weights / weights.sum(dim=0, keepdim=True).clamp_min(1e-8)
        weights = weights.unsqueeze(0)

        if len(self._field_weight_cache) > 3:
            self._field_weight_cache.clear()
        self._field_weight_cache[key] = weights
        return weights

    def forward(self, rgb: torch.Tensor, coeff_perturb_std: float = 0.0) -> torch.Tensor:
        if rgb.ndim != 4:
            raise ValueError("Expected BCHW tensor.")
        batch, channels, height, width = rgb.shape
        if channels != self.channels:
            raise ValueError(f"ConstrainedFieldPSF was built for {self.channels} channels, got {channels}.")

        coeff = None
        if coeff_perturb_std > 0.0 and self.training:
            coeff = self.effective_coefficients()
            noise = torch.randn_like(coeff) * float(coeff_perturb_std)
            bounds = self.coeff_bounds.to(coeff.device, coeff.dtype)
            coeff = (coeff + noise).clamp(-bounds, bounds)
        kernels = self.psf_kernels(device=rgb.device, dtype=rgb.dtype, coeff_override=coeff)
        field_w = self._field_weights(height, width, rgb.device, rgb.dtype)
        pad_mode = self.padding_mode
        if height <= self.radius or width <= self.radius:
            pad_mode = "replicate"
        padded = F.pad(rgb, (self.radius, self.radius, self.radius, self.radius), mode=pad_mode)

        out = rgb.new_zeros(batch, channels, height, width)
        for z in range(self.num_zones):
            kernel_z = kernels[z].reshape(channels, 1, self.kernel_size, self.kernel_size)
            blurred = F.conv2d(padded, kernel_z, groups=channels)
            out = out + blurred * field_w[:, z : z + 1]
        return out.clamp(0.0, 1.0)

    def regularization(self) -> Dict[str, torch.Tensor]:
        coeff = self.effective_coefficients()
        kernels = self.psf_kernels(device=coeff.device, dtype=coeff.dtype)

        coords = torch.arange(self.kernel_size, device=coeff.device, dtype=coeff.dtype) - float(self.radius)
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        rr2 = (xx.square() + yy.square()).view(1, 1, self.kernel_size, self.kernel_size)
        second_moment = (kernels * rr2).sum(dim=(-2, -1)).mean() / max(float(self.radius * self.radius), 1.0)

        if self.num_zones > 1:
            field_smoothness = (coeff - coeff.mean(dim=0, keepdim=True)).square().mean()
        else:
            field_smoothness = coeff.new_tensor(0.0)
        chromatic_smoothness = (coeff - coeff.mean(dim=1, keepdim=True)).square().mean()
        coma_shift = coeff[..., 3:5].square().mean() / max(self.max_shift_px * self.max_shift_px, 1e-6)
        kernel_tv = (
            (kernels[..., 1:, :] - kernels[..., :-1, :]).abs().mean()
            + (kernels[..., :, 1:] - kernels[..., :, :-1]).abs().mean()
        )

        terms = {
            "psf_second_moment": second_moment,
            "field_smoothness": field_smoothness,
            "chromatic_smoothness": chromatic_smoothness,
            "coma_shift": coma_shift,
            "kernel_tv": kernel_tv,
        }
        total = coeff.new_tensor(0.0)
        for name, value in terms.items():
            total = total + float(self.regularization_weights.get(name, 0.0)) * value
        terms["total"] = total
        return terms

    def export_design(self, include_kernels: bool = True) -> Dict[str, Any]:
        with torch.no_grad():
            coeff = self.effective_coefficients().detach().cpu()
            regs = {k: float(v.detach().cpu()) for k, v in self.regularization().items()}
            bounds = self.coeff_bounds.detach().cpu().view(-1)
            bound_use = (coeff.abs() / bounds.view(1, 1, -1).clamp_min(1e-8)).amax(dim=(0, 1))
            kernels = self.psf_kernels(device=torch.device("cpu"), dtype=torch.float32)
            kernel_sums = kernels.sum(dim=(-2, -1))
            kernel_min = kernels.amin()
            coeff_delta = coeff - coeff.mean(dim=0, keepdim=True)
            design: Dict[str, Any] = {
                "type": "constrained_field_psf",
                "num_zones": self.num_zones,
                "channels": self.channels,
                "kernel_size": self.kernel_size,
                "base_sigma_px": self.base_sigma_px,
                "min_sigma_px": self.min_sigma_px,
                "max_sigma_px": self.max_sigma_px,
                "max_shift_px": self.max_shift_px,
                "field_sigma": self.field_sigma,
                "coefficient_names": list(COEFF_NAMES),
                "zone_centers_xy": self.zone_centers.detach().cpu().tolist(),
                "coefficients": coeff.tolist(),
                "regularization": regs,
                "physical_constraints": {
                    "nonnegative_energy_normalized_psf": bool(kernel_min >= -1e-7 and torch.allclose(kernel_sums, torch.ones_like(kernel_sums), atol=1e-4)),
                    "odd_centered_kernel": bool(self.kernel_size % 2 == 1),
                    "bounded_low_order_coefficients": True,
                    "smooth_field_interpolation": True,
                    "per_channel_psf": True,
                    "padding_mode": self.padding_mode,
                },
                "constraint_margins": {
                    "min_psf_value": float(kernel_min),
                    "max_abs_kernel_sum_error": float((kernel_sums - 1.0).abs().max()),
                    "max_coeff_bound_fraction": float(bound_use.max()),
                    "mean_coeff_bound_fraction": float((coeff.abs() / bounds.view(1, 1, -1).clamp_min(1e-8)).mean()),
                    "rms_field_coefficient_delta": float(torch.sqrt(coeff_delta.square().mean())),
                    "coeff_bound_fraction_by_name": {
                        name: float(value) for name, value in zip(COEFF_NAMES, bound_use)
                    },
                },
            }
            if include_kernels:
                design["psf_kernels"] = kernels.tolist()
            return design
