import torch
import torch.nn as nn
import torch.nn.functional as F

def _pattern_weights(name: str) -> torch.Tensor:
    """
    Return a (4,3) tensor of initial RGB weights for a 2x2 CFA tile.
    Order of 4 entries is row-major: (0,0),(0,1),(1,0),(1,1).
    Each row sums arbitrary; training will learn the best mixing.
    """
    n = (name or "").lower()
    # basic basis
    R = torch.tensor([1.0, 0.0, 0.0])
    G = torch.tensor([0.0, 1.0, 0.0])
    B = torch.tensor([0.0, 0.0, 1.0])
    C = torch.tensor([1/3, 1/3, 1/3])  # “clear” start: equal RGB

    if n in ("bayer_rggb", "rggb", "bayer"):
        # [[R, G],
        #  [G, B]]
        tile = [R, G, G, B]
    elif n in ("rccc",):
        # [[R, C],
        #  [C, C]]
        tile = [R, C, C, C]
    elif n in ("rccb",):
        # [[R, C],
        #  [C, B]]
        tile = [R, C, C, B]
    elif n in ("grbg", "bayer_grbg"):
        tile = [G, R, B, G]
    elif n in ("gbrg", "bayer_gbrg"):
        tile = [G, B, R, G]
    elif n in ("bggr", "bayer_bggr"):
        tile = [B, G, G, R]
    else:
        # default to RGGB
        tile = [R, G, G, B]
    return torch.stack(tile, dim=0)  # (4,3)

CFA_PATTERNS = {
    "bayer_rggb": torch.tensor([[0,1],[2,0]]),  # R=0, G=1, B=2
}

def _init_learned_cfa():
    return nn.Parameter(torch.zeros(4, 3))  # 4 sites × RGB weights

class LearnableCFA(nn.Module):
    """Learnable 2x2 CFA with nonnegative, unit-sum spectral responses."""
    def __init__(
        self,
        init: str = "bayer_rggb",
        init_floor: float = 1e-4,
        temperature: float = 1.0,
        min_transmittance: float = 0.0,
    ):
        super().__init__()
        self.temperature = float(max(temperature, 1e-3))
        self.min_transmittance = float(max(0.0, min(min_transmittance, 0.30)))
        w0 = _pattern_weights(init)  # torch.Tensor (4,3)
        floor = float(max(init_floor, 1e-6))
        if floor > 0:
            w0 = w0 + floor
        w0 = w0 / w0.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        self.weight = nn.Parameter(torch.log(w0.clamp_min(1e-6)) * self.temperature)

    def set_temperature(self, temperature: float) -> None:
        self.temperature = float(max(temperature, 1e-3))

    def effective_weights(self, perturb_std: float = 0.0) -> torch.Tensor:
        logits = self.weight
        if perturb_std > 0.0 and self.training:
            logits = logits + torch.randn_like(logits) * float(perturb_std)
        weights = F.softmax(logits / self.temperature, dim=-1)
        if self.min_transmittance > 0.0:
            floor = torch.as_tensor(self.min_transmittance, device=weights.device, dtype=weights.dtype)
            weights = floor + (1.0 - 3.0 * floor) * weights
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return weights

    def forward(self, rgb: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
        # rgb: (B,3,H,W), outputs RAW: (B,1,H,W)
        B, C, H, W = rgb.shape
        w = self.effective_weights() if weights is None else weights
        # split 2x2 positions
        r0 = rgb[:, :, 0::2, 0::2]   # (B,3,H/2,W/2)
        r1 = rgb[:, :, 0::2, 1::2]
        r2 = rgb[:, :, 1::2, 0::2]
        r3 = rgb[:, :, 1::2, 1::2]
        # apply per-position mixing
        y0 = (r0 * w[0][None, :, None, None]).sum(dim=1, keepdim=True)
        y1 = (r1 * w[1][None, :, None, None]).sum(dim=1, keepdim=True)
        y2 = (r2 * w[2][None, :, None, None]).sum(dim=1, keepdim=True)
        y3 = (r3 * w[3][None, :, None, None]).sum(dim=1, keepdim=True)
        # stitch back to full-res mosaic
        raw = torch.zeros((B, 1, H, W), dtype=rgb.dtype, device=rgb.device)
        raw[:, :, 0::2, 0::2] = y0
        raw[:, :, 0::2, 1::2] = y1
        raw[:, :, 1::2, 0::2] = y2
        raw[:, :, 1::2, 1::2] = y3
        return raw

def superpixel_demux(mosi):  # (B,1,H,W) -> (B,4,H/2,W/2)
    B, C, H, W = mosi.shape
    x = mosi.view(B, 1, H//2, 2, W//2, 2)
    return x.permute(0,1,3,5,2,4).contiguous().view(B,4,H//2,W//2)


def soft_cfa_demosaic(raw: torch.Tensor, weights: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
    """Differentiable CFA-aware RAW readout for RGB-pretrained task heads.

    This is not a learned ISP. It keeps the learned CFA responses explicit:
    each RAW sample is projected back into RGB proportionally to that pixel's
    CFA spectral response, then missing support is filled by normalized local
    interpolation. Gradients flow to the CFA weights and upstream optics/noise.
    """
    if raw.ndim != 4 or raw.shape[1] != 1:
        raise ValueError(f"Expected RAW mosaic with shape (B,1,H,W), got {tuple(raw.shape)}")
    if weights.shape != (4, 3):
        raise ValueError(f"Expected CFA weights with shape (4,3), got {tuple(weights.shape)}")

    b, _, h, w_img = raw.shape
    coded = raw.new_zeros((b, 3, h, w_img))
    support = raw.new_zeros((b, 3, h, w_img))

    sites = (
        (slice(0, None, 2), slice(0, None, 2)),
        (slice(0, None, 2), slice(1, None, 2)),
        (slice(1, None, 2), slice(0, None, 2)),
        (slice(1, None, 2), slice(1, None, 2)),
    )
    w_soft = weights.to(device=raw.device, dtype=raw.dtype).clamp_min(1e-6)
    for idx, (ys, xs) in enumerate(sites):
        site_raw = raw[:, :, ys, xs]
        site_w = w_soft[idx].view(1, 3, 1, 1)
        coded[:, :, ys, xs] = site_raw * site_w
        support[:, :, ys, xs] = site_w

    k = int(kernel_size)
    if k < 1 or k % 2 == 0:
        raise ValueError(f"kernel_size must be a positive odd integer, got {kernel_size}")
    kernel = raw.new_ones((3, 1, k, k))
    pad = k // 2
    num = F.conv2d(coded, kernel, padding=pad, groups=3)
    den = F.conv2d(support, kernel, padding=pad, groups=3).clamp_min(1e-6)
    return torch.clamp(num / den, 0.0, 1.0)
