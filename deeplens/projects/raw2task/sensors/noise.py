import torch
import torch.nn as nn

class PoissonGaussianNoiseQuant(nn.Module):
    """Shot/read noise and ADC quantization with optional learnable parameters.

    The bit depth is optimized through a bounded continuous proxy and rounded
    only when exported as a fixed hardware design. Forward quantization still
    uses a straight-through estimator.
    """
    def __init__(
        self,
        bit_depth=6,
        read_noise_std=0.005,
        shot_noise_scale=0.02,
        learn_noise: bool = False,
        learn_bit_depth: bool = False,
        min_read_noise_std: float = 1e-5,
        max_read_noise_std: float = 0.05,
        min_shot_noise_scale: float = 1e-5,
        max_shot_noise_scale: float = 0.12,
        min_bit_depth: float = 4.0,
        max_bit_depth: float = 12.0,
    ):
        super().__init__()
        self.min_read_noise_std = float(min_read_noise_std)
        self.max_read_noise_std = float(max_read_noise_std)
        self.min_shot_noise_scale = float(min_shot_noise_scale)
        self.max_shot_noise_scale = float(max_shot_noise_scale)
        self.min_bit_depth = float(min_bit_depth)
        self.max_bit_depth = float(max_bit_depth)

        self.read_noise_raw = nn.Parameter(
            self._inv_sigmoid_from_range(float(read_noise_std), self.min_read_noise_std, self.max_read_noise_std),
            requires_grad=bool(learn_noise),
        )
        self.shot_noise_raw = nn.Parameter(
            self._inv_sigmoid_from_range(float(shot_noise_scale), self.min_shot_noise_scale, self.max_shot_noise_scale),
            requires_grad=bool(learn_noise),
        )
        self.bit_depth_raw = nn.Parameter(
            self._inv_sigmoid_from_range(float(bit_depth), self.min_bit_depth, self.max_bit_depth),
            requires_grad=bool(learn_bit_depth),
        )

    @staticmethod
    def _inv_sigmoid_from_range(value: float, lo: float, hi: float) -> torch.Tensor:
        v = (float(value) - lo) / max(hi - lo, 1e-8)
        v = min(max(v, 1e-4), 1.0 - 1e-4)
        return torch.tensor(float(torch.logit(torch.tensor(v))))

    @staticmethod
    def _bounded(raw: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
        return lo + (hi - lo) * torch.sigmoid(raw)

    def read_noise_std_value(self) -> torch.Tensor:
        return self._bounded(self.read_noise_raw, self.min_read_noise_std, self.max_read_noise_std)

    def shot_noise_scale_value(self) -> torch.Tensor:
        return self._bounded(self.shot_noise_raw, self.min_shot_noise_scale, self.max_shot_noise_scale)

    def bit_depth_value(self) -> torch.Tensor:
        return self._bounded(self.bit_depth_raw, self.min_bit_depth, self.max_bit_depth)

    @property
    def read_noise_std(self) -> float:
        return float(self.read_noise_std_value().detach().cpu())

    @read_noise_std.setter
    def read_noise_std(self, value: float) -> None:
        self.read_noise_raw.data.copy_(
            self._inv_sigmoid_from_range(float(value), self.min_read_noise_std, self.max_read_noise_std).to(self.read_noise_raw.device)
        )

    @property
    def shot_noise_scale(self) -> float:
        return float(self.shot_noise_scale_value().detach().cpu())

    @shot_noise_scale.setter
    def shot_noise_scale(self, value: float) -> None:
        self.shot_noise_raw.data.copy_(
            self._inv_sigmoid_from_range(float(value), self.min_shot_noise_scale, self.max_shot_noise_scale).to(self.shot_noise_raw.device)
        )

    @property
    def bit_depth(self) -> int:
        return int(round(float(self.bit_depth_value().detach().cpu())))

    @bit_depth.setter
    def bit_depth(self, value: float) -> None:
        self.bit_depth_raw.data.copy_(
            self._inv_sigmoid_from_range(float(value), self.min_bit_depth, self.max_bit_depth).to(self.bit_depth_raw.device)
        )

    def forward(self, x, stochastic: bool | None = None):  # x in [0,1]
        x = torch.clamp(x, 0.0, 1.0)
        if stochastic is None:
            stochastic = bool(self.training)
        shot_scale = self.shot_noise_scale_value().to(device=x.device, dtype=x.dtype)
        read_std = self.read_noise_std_value().to(device=x.device, dtype=x.dtype)
        bit_depth = self.bit_depth_value().to(device=x.device, dtype=x.dtype)
        if stochastic:
            shot_std = torch.sqrt(torch.clamp(shot_scale * x, min=1e-6))
            noisy = x + torch.randn_like(x) * shot_std + torch.randn_like(x) * read_std
        else:
            noisy = x
        noisy = torch.clamp(noisy, 0.0, 1.0)
        levels = torch.pow(torch.tensor(2.0, device=x.device, dtype=x.dtype), bit_depth) - 1.0
        y = noisy * levels
        y_rounded = torch.round(y)
        y = y + (y_rounded - y).detach()   # straight-through estimator
        return y / levels

    def export_design(self):
        return {
            "bit_depth_continuous": float(self.bit_depth_value().detach().cpu()),
            "bit_depth_deploy": self.bit_depth,
            "read_noise_std": self.read_noise_std,
            "shot_noise_scale": self.shot_noise_scale,
            "learn_read_shot_noise": bool(self.read_noise_raw.requires_grad or self.shot_noise_raw.requires_grad),
            "learn_bit_depth": bool(self.bit_depth_raw.requires_grad),
            "bounds": {
                "read_noise_std": [self.min_read_noise_std, self.max_read_noise_std],
                "shot_noise_scale": [self.min_shot_noise_scale, self.max_shot_noise_scale],
                "bit_depth": [self.min_bit_depth, self.max_bit_depth],
            },
        }
