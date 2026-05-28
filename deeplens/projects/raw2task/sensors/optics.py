import torch, math
import torch.nn as nn
import torch.nn.functional as F

def _gaussian_kernel(size: int, sigma: torch.Tensor, device):
    coords = torch.arange(size, device=device) - (size - 1)/2.0
    x, y = torch.meshgrid(coords, coords, indexing='ij')
    k = torch.exp(-(x**2 + y**2) / (2*sigma**2 + 1e-8))
    k = k / (k.sum() + 1e-8)
    return k

class DifferentiablePSF(nn.Module):
    """Simple Gaussian PSF with learnable log_sigma. Depthwise conv. 
    TODO: Replace with a thin wrapper that calls a DeepLens render for real optics."""
    def __init__(self, init_sigma=0.8, kernel_size=9):
        super().__init__()
        self.log_sigma = nn.Parameter(torch.tensor(math.log(init_sigma+1e-6), dtype=torch.float32))
        self.kernel_size = kernel_size

    def forward(self, x):  # (B,C,H,W)
        sigma = torch.exp(self.log_sigma) + 1e-6
        k = _gaussian_kernel(self.kernel_size, sigma, x.device)
        k = k.view(1,1,self.kernel_size,self.kernel_size).repeat(x.shape[1],1,1,1)
        return F.conv2d(x, k, padding=self.kernel_size//2, groups=x.shape[1])
