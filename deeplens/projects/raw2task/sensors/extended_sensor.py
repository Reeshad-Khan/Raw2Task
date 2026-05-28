import torch
import torch.nn as nn
from deeplens.geolens import Lens   # or GeoLens if you want multi-surface optics
from deeplens.projects.raw2task.sensors.cfa import LearnableCFA
from deeplens.projects.raw2task.sensors.noise import PoissonGaussianNoiseQuant

class ExtendedSensor(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.lens = Lens()  # Thin lens, DeepLens handles internals
        self.cfa = LearnableCFA(init="bayer_rggb")
        self.noise = PoissonGaussianNoiseQuant(
            bit_depth=cfg['sensor']['bit_depth'],
            read_noise_std=cfg['sensor']['read_noise_std'],
            shot_noise_scale=cfg['sensor']['shot_noise_scale'],
        )
        self.exposure_logit = nn.Parameter(torch.tensor(0.0))

    def forward(self, rgb):  # (B,3,H,W)
        # 1. Optics
        focus_distance = float(self.cfg['sensor'].get('focus_distance', 10.0))
        rgb = self.lens.render(rgb, depth=focus_distance)
        # 2. Exposure
        exposure = 0.25 + 3.75 * torch.sigmoid(self.exposure_logit)
        rgb = torch.clamp(rgb * exposure, 0, 1)
        # 3. CFA
        raw = self.cfa(rgb)
        # 4. Noise + Quantization
        raw = self.noise(raw)
        return raw
