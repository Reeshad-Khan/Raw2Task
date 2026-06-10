# deeplens/projects/raw2task/utils/efficiency.py
# Copyright (c) 2025 DeepLens Authors.
# Utility: model efficiency measurement (params/FLOPs/MACs/acts/latency) and CSV/JSON logging.

from __future__ import annotations
import time
import json
import os
from dataclasses import dataclass, asdict
from typing import Tuple, Optional, Dict

import torch
import torch.nn as nn

# We try several FLOP backends in order of preference.
_BACKENDS = {"thop": None, "fvcore": None, "ptflops": None}
try:
    import thop  # type: ignore
    _BACKENDS["thop"] = thop
except Exception:
    pass
try:
    from fvcore.nn import FlopCountAnalysis, ActivationCountAnalysis  # type: ignore
    _BACKENDS["fvcore"] = True
except Exception:
    pass
try:
    from ptflops import get_model_complexity_info  # type: ignore
    _BACKENDS["ptflops"] = True
except Exception:
    pass


@dataclass
class EffResult:
    params_m: float
    flops_g: Optional[float]   # multiply-add FLOPs (or MACs) in G
    macs_g: Optional[float]    # if the backend reports MACs, we map here
    acts_m: Optional[float]    # activation count (elements) in millions
    latency_ms: float          # avg of several runs
    batch: int
    shape: Tuple[int, int, int]  # (C,H,W)
    notes: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def _count_params_m(model: nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6


@torch.no_grad()
def _measure_latency_ms(model: nn.Module, x: torch.Tensor, warmup: int = 10, iters: int = 30) -> float:
    model.eval()
    device = x.device
    # warmup
    for _ in range(warmup):
        _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0 / max(1, iters)


def _fvcore_counts(model: nn.Module, x: torch.Tensor) -> Tuple[Optional[float], Optional[float]]:
    try:
        f = FlopCountAnalysis(model, x)
        a = ActivationCountAnalysis(model, x)
        flops = float(f.total()) / 1e9  # G
        acts  = float(a.total()) / 1e6  # M
        return flops, acts
    except Exception:
        return None, None


def _thop_counts(model: nn.Module, x: torch.Tensor) -> Tuple[Optional[float], Optional[float]]:
    thop = _BACKENDS["thop"]
    if thop is None:
        return None, None
    try:
        macs, params = thop.profile(model, inputs=(x,), verbose=False)
        # thop returns MACs (not FLOPs) by default; also returns params (which we already compute)
        macs_g = float(macs) / 1e9
        return macs_g, None  # (macs_g, acts None)
    except Exception:
        return None, None


def _ptflops_counts(model: nn.Module, shape: Tuple[int, int, int], device: torch.device) -> Tuple[Optional[float], Optional[float]]:
    if not _BACKENDS["ptflops"]:
        return None, None
    try:
        # ptflops builds an input of (1,C,H,W) on CPU; it's fine to keep as an estimate
        flops_str, params_str = get_model_complexity_info(model.cpu(), shape, as_strings=True, print_per_layer_stat=False)
        # Strings like '1.23 GMac' or '567.89 KMac'
        def _to_g(s: str) -> float:
            s = s.upper()
            val = float(s.split()[0])
            if "GMAC" in s or "GFLOP" in s:
                return val
            if "MMAC" in s or "MFLOP" in s:
                return val / 1e3
            if "KMAC" in s or "KFLOP" in s:
                return val / 1e6
            return val / 1e9
        macs_g = _to_g(flops_str)
        model.to(device)
        return macs_g, None
    except Exception:
        model.to(device)
        return None, None


def _run_counts(model: nn.Module, x: torch.Tensor, input_shape: Tuple[int,int,int]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Returns (flops_g, macs_g, acts_m) — some may be None depending on backend availability.
    Priority:
      - fvcore for FLOPs & activations
      - thop for MACs
      - ptflops as a fallback for MACs
    """
    flops_g = acts_m = macs_g = None

    if _BACKENDS["fvcore"]:
        f, a = _fvcore_counts(model, x)
        flops_g = f
        acts_m  = a

    if macs_g is None and _BACKENDS["thop"]:
        m, _ = _thop_counts(model, x)
        macs_g = m

    if macs_g is None and _BACKENDS["ptflops"]:
        m, _ = _ptflops_counts(model, input_shape, x.device)
        macs_g = m

    return flops_g, macs_g, acts_m


class Chain(nn.Module):
    """Wraps (sensor -> core_model) for joint efficiency measurement."""
    def __init__(self, sensor: nn.Module, core: nn.Module):
        super().__init__()
        self.sensor = sensor
        self.core   = core

    def forward(self, x):
        raw = self.sensor(x)
        return self.core(raw)


@torch.no_grad()
def measure_model_efficiency(
    model: nn.Module,
    input_shape: Tuple[int, int, int],  # (C,H,W)
    device: torch.device,
    batch_size: int = 1,
    notes: str = "",
) -> EffResult:
    model.eval().to(device)
    C, H, W = input_shape
    x = torch.randn(batch_size, C, H, W, device=device)

    params_m  = _count_params_m(model)
    flops_g, macs_g, acts_m = _run_counts(model, x, input_shape)
    latency_ms = _measure_latency_ms(model, x)

    return EffResult(
        params_m=params_m,
        flops_g=flops_g,
        macs_g=macs_g,
        acts_m=acts_m,
        latency_ms=latency_ms,
        batch=batch_size,
        shape=input_shape,
        notes=notes,
    )


def save_efficiency(result: EffResult, out_dir: str, stem: str = "efficiency", also_csv: bool = True):
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, f"{stem}.json")
    with open(json_path, "w") as f:
        f.write(result.to_json())

    if also_csv:
        import csv
        csv_path = os.path.join(out_dir, f"{stem}.csv")
        write_header = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["params_m","flops_g","macs_g","acts_m","latency_ms","batch","C","H","W","notes"])
            w.writerow([
                result.params_m,
                result.flops_g if result.flops_g is not None else "",
                result.macs_g  if result.macs_g  is not None else "",
                result.acts_m  if result.acts_m  is not None else "",
                result.latency_ms,
                result.batch, result.shape[0], result.shape[1], result.shape[2],
                result.notes
            ])