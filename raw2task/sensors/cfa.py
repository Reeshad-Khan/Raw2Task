import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _pattern_weights(name: str, n_sites: int = 4) -> torch.Tensor:
    """Return (n_sites, 3) normalised initial RGB weights for the CFA tile."""
    n = (name or "").lower()
    R = torch.tensor([1.0, 0.0, 0.0])
    G = torch.tensor([0.0, 1.0, 0.0])
    B = torch.tensor([0.0, 0.0, 1.0])
    C = torch.tensor([1 / 3, 1 / 3, 1 / 3])

    if n in ("bayer_rggb", "rggb", "bayer"):
        base = [R, G, G, B]
    elif n in ("rccc",):
        base = [R, C, C, C]
    elif n in ("rccb",):
        base = [R, C, C, B]
    elif n in ("grbg", "bayer_grbg"):
        base = [G, R, B, G]
    elif n in ("gbrg", "bayer_gbrg"):
        base = [G, B, R, G]
    elif n in ("bggr", "bayer_bggr"):
        base = [B, G, G, R]
    else:
        base = [R, G, G, B]

    # For tiles larger than 2×2 we extend with spectral-diverse mixtures so that
    # gradient-based optimisation starts from a spread-out initialisation rather
    # than degenerate repetition of the same four filters.
    extras = [
        (R + G) / 2,
        (G + B) / 2,
        (R + B) / 2,
        C,
        (2 * R + G) / 3,
        (R + 2 * G) / 3,
        (2 * G + B) / 3,
        (G + 2 * B) / 3,
        (2 * R + B) / 3,
        (R + 2 * B) / 3,
        (R + G + 2 * B) / 4,
        (2 * R + G + B) / 4,
    ]
    pool = base + extras
    tile = [pool[i % len(pool)] for i in range(n_sites)]
    w = torch.stack(tile, dim=0)                                   # (n_sites, 3)
    return w / w.sum(dim=-1, keepdim=True).clamp_min(1e-6)


CFA_PATTERNS = {
    "bayer_rggb": torch.tensor([[0, 1], [2, 0]]),   # R=0, G=1, B=2
}


class LearnableCFA(nn.Module):
    """Learnable T×T CFA with non-negative, unit-sum spectral responses.

    tile_size=2 reproduces the original 2×2 Bayer-style learnable CFA.
    tile_size=3 gives a 3×3 tile with 9 independently learned spectral filters.
    tile_size=4 gives a 4×4 tile with 16 filters — maximum spectral diversity.

    Larger tiles trade per-filter spatial density for spectral diversity.
    For semantic segmentation (which benefits from class-discriminative spectral
    sampling), tile_size > 2 is expected to outperform Bayer.
    """

    def __init__(
        self,
        init: str = "bayer_rggb",
        init_floor: float = 1e-4,
        temperature: float = 1.0,
        min_transmittance: float = 0.0,
        tile_size: int = 2,
    ):
        super().__init__()
        self.tile_size = int(tile_size)
        if self.tile_size < 1:
            raise ValueError(f"tile_size must be >= 1, got {self.tile_size}")
        self.n_sites = self.tile_size * self.tile_size
        self.temperature = float(max(temperature, 1e-3))
        self.min_transmittance = float(max(0.0, min(min_transmittance, 0.30)))
        w0 = _pattern_weights(init, self.n_sites)                  # (n_sites, 3)
        floor = float(max(init_floor, 1e-6))
        if floor > 0:
            w0 = w0 + floor
        w0 = w0 / w0.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        self.weight = nn.Parameter(torch.log(w0.clamp_min(1e-6)) * self.temperature)

    def set_temperature(self, temperature: float) -> None:
        self.temperature = float(max(temperature, 1e-3))

    def effective_weights(self, perturb_std: float = 0.0) -> torch.Tensor:
        """Return (n_sites, 3) softmax-normalised spectral weights."""
        logits = self.weight
        if perturb_std > 0.0 and self.training:
            logits = logits + torch.randn_like(logits) * float(perturb_std)
        weights = F.softmax(logits / self.temperature, dim=-1)
        if self.min_transmittance > 0.0:
            floor = torch.as_tensor(
                self.min_transmittance, device=weights.device, dtype=weights.dtype
            )
            weights = floor + (1.0 - 3.0 * floor) * weights
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return weights

    def forward(self, rgb: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
        """Apply the T×T CFA mosaic.

        Args:
            rgb: (B, 3, H, W) linear-light scene image.
            weights: override effective_weights(); shape (n_sites, 3).
        Returns:
            raw: (B, 1, H, W) single-channel RAW mosaic.
        """
        B, C, H, W = rgb.shape
        T = self.tile_size
        w = self.effective_weights() if weights is None else weights  # (n_sites, 3)
        raw = torch.zeros((B, 1, H, W), dtype=rgb.dtype, device=rgb.device)
        idx = 0
        for r in range(T):
            for c in range(T):
                site = rgb[:, :, r::T, c::T]                          # (B, 3, H/T, W/T)
                site_w = w[idx].view(1, 3, 1, 1)
                raw[:, :, r::T, c::T] = (site * site_w).sum(dim=1, keepdim=True)
                idx += 1
        return raw


def superpixel_demux(mosi: torch.Tensor) -> torch.Tensor:
    """Unpack a 2×2 RAW mosaic into 4 packed channels. (2×2 tiles only.)"""
    B, C, H, W = mosi.shape
    x = mosi.view(B, 1, H // 2, 2, W // 2, 2)
    return x.permute(0, 1, 3, 5, 2, 4).contiguous().view(B, 4, H // 2, W // 2)


def soft_cfa_demosaic(
    raw: torch.Tensor,
    weights: torch.Tensor,
    kernel_size: int | None = None,
) -> torch.Tensor:
    """Differentiable CFA-aware demosaicking for any T×T tile pattern.

    Scatters each site's RAW value back into a 3-channel image weighted by that
    site's spectral response, then fills missing support with a normalised local
    average. Gradients propagate to CFA weights and all upstream modules.

    Args:
        raw:         (B, 1, H, W) single-channel RAW mosaic.
        weights:     (n_sites, 3) CFA spectral weights, n_sites = T*T.
        kernel_size: Size of the averaging kernel. If None, auto-set to
                     ``max(5, 2*T + 1)`` so the kernel spans at least one full
                     tile period in every direction.
    Returns:
        (B, 3, H, W) demosaicked RGB image, values in [0, 1].
    """
    if raw.ndim != 4 or raw.shape[1] != 1:
        raise ValueError(f"Expected RAW mosaic (B,1,H,W), got {tuple(raw.shape)}")
    n_sites = weights.shape[0]
    T = int(round(math.sqrt(n_sites)))
    if T * T != n_sites:
        raise ValueError(f"weights.shape[0]={n_sites} is not a perfect square")
    if weights.shape[1] != 3:
        raise ValueError(f"Expected weights shape (n_sites,3), got {tuple(weights.shape)}")

    if kernel_size is None:
        # Cover ≥ 2.5 tile periods so every pixel has dense support from each
        # filter type regardless of tile size.
        k_auto = max(5, 2 * T + 1)
        kernel_size = k_auto if k_auto % 2 == 1 else k_auto + 1

    k = int(kernel_size)
    if k < 1 or k % 2 == 0:
        raise ValueError(f"kernel_size must be a positive odd integer, got {k}")

    b, _, h, w_img = raw.shape
    coded   = raw.new_zeros((b, 3, h, w_img))
    support = raw.new_zeros((b, 3, h, w_img))
    w_soft  = weights.to(device=raw.device, dtype=raw.dtype).clamp_min(1e-6)

    idx = 0
    for r in range(T):
        for c in range(T):
            site_raw = raw[:, :, r::T, c::T]
            site_w   = w_soft[idx].view(1, 3, 1, 1)
            coded  [:, :, r::T, c::T] = site_raw * site_w
            support[:, :, r::T, c::T] = site_w
            idx += 1

    kernel = raw.new_ones((3, 1, k, k))
    pad    = k // 2
    num = F.conv2d(coded,   kernel, padding=pad, groups=3)
    den = F.conv2d(support, kernel, padding=pad, groups=3).clamp_min(1e-6)
    return torch.clamp(num / den, 0.0, 1.0)
