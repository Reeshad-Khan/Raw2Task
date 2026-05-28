"""Segmentation model registry for raw-to-task experiments.

The registry keeps the paper experiments honest: the same training loop can run
our compact RAW/sensor model and modern RGB baselines on the same split and
metric. Optional backends are imported lazily so the core UNet smoke tests run
without installing every model zoo.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from deeplens.projects.raw2task.models.unet_tiny import UNetTiny


def _cached_hf_config_and_safetensors(model_id: str) -> tuple[str, str] | None:
    """Return local HF cache dirs for config and safetensors when available.

    Some older HF segmentation checkpoints keep `config.json` on the main
    snapshot and converted `model.safetensors` on an auto-conversion PR
    snapshot. Loading those directly avoids API calls during batch-size probes
    and avoids unsafe `pytorch_model.bin` fallback on torch<2.6 environments.
    """
    cache_root = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache" / "huggingface" / "hub"))
    repo_dir = cache_root / f"models--{model_id.replace('/', '--')}"
    snapshots = repo_dir / "snapshots"
    if not snapshots.is_dir():
        return None
    config_dirs = sorted(p for p in snapshots.iterdir() if (p / "config.json").exists())
    safe_dirs = sorted(p for p in snapshots.iterdir() if (p / "model.safetensors").exists())
    if not config_dirs or not safe_dirs:
        return None
    return str(config_dirs[-1]), str(safe_dirs[-1])


class ConvBNAct(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, groups: int = 1):
        p = k // 2
        super().__init__(
            nn.Conv2d(in_ch, out_ch, k, s, p, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )


class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNAct(in_ch, in_ch, k=3, s=stride, groups=in_ch),
            ConvBNAct(in_ch, out_ch, k=1, s=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class LiteSeg(nn.Module):
    """Small production-friendly baseline using depthwise separable blocks.

    It is deliberately stronger than the original tiny UNet while staying small,
    and gives reviewers a same-codepath compact baseline.
    """

    def __init__(self, in_channels: int, num_classes: int, width: int = 48):
        super().__init__()
        w = int(width)
        self.stem = nn.Sequential(ConvBNAct(in_channels, w, 3, 2), ConvBNAct(w, w, 3, 1))
        self.s2 = nn.Sequential(DepthwiseSeparableBlock(w, 2 * w, 2), DepthwiseSeparableBlock(2 * w, 2 * w))
        self.s4 = nn.Sequential(DepthwiseSeparableBlock(2 * w, 4 * w, 2), DepthwiseSeparableBlock(4 * w, 4 * w))
        self.context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(4 * w, 4 * w, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(4 * w, 4 * w, 1),
            nn.Sigmoid(),
        )
        self.fuse = nn.Sequential(ConvBNAct(6 * w, 2 * w, 1), DepthwiseSeparableBlock(2 * w, 2 * w))
        self.head = nn.Conv2d(2 * w, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_size = x.shape[-2:]
        x1 = self.stem(x)
        x2 = self.s2(x1)
        x4 = self.s4(x2)
        x4 = x4 * self.context(x4)
        x4_up = F.interpolate(x4, size=x2.shape[-2:], mode="bilinear", align_corners=False)
        y = self.fuse(torch.cat([x2, x4_up], dim=1))
        return F.interpolate(self.head(y), size=out_size, mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dilations=(1, 3, 6)):
        super().__init__()
        branches = []
        for dilation in dilations:
            if dilation == 1:
                branches.append(nn.Sequential(ConvBNAct(in_ch, out_ch, k=1, s=1)))
            else:
                branches.append(
                    nn.Sequential(
                        nn.Conv2d(in_ch, out_ch, 3, padding=dilation, dilation=dilation, bias=False),
                        nn.BatchNorm2d(out_ch),
                        nn.SiLU(inplace=True),
                    )
                )
        self.branches = nn.ModuleList(branches)
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.GroupNorm(1, out_ch),
            nn.SiLU(inplace=True),
        )
        self.project = nn.Sequential(
            ConvBNAct(out_ch * (len(self.branches) + 1), out_ch, k=1, s=1),
            nn.Dropout2d(p=0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = [branch(x) for branch in self.branches]
        pooled = self.image_pool(x)
        pooled = F.interpolate(pooled, size=x.shape[-2:], mode="bilinear", align_corners=False)
        outs.append(pooled)
        return self.project(torch.cat(outs, dim=1))


class LiteSegASPP(nn.Module):
    """Stronger LiteSeg decoder with an ASPP context block."""

    def __init__(self, in_channels: int, num_classes: int, width: int = 64):
        super().__init__()
        w = int(width)
        self.stem = nn.Sequential(ConvBNAct(in_channels, w, 3, 2), ConvBNAct(w, w, 3, 1))
        self.s2 = nn.Sequential(DepthwiseSeparableBlock(w, 2 * w, 2), DepthwiseSeparableBlock(2 * w, 2 * w))
        self.s4 = nn.Sequential(DepthwiseSeparableBlock(2 * w, 4 * w, 2), DepthwiseSeparableBlock(4 * w, 4 * w))
        self.low_proj = ConvBNAct(2 * w, 2 * w, k=1, s=1)
        self.aspp = ASPP(4 * w, 2 * w, dilations=(1, 3, 6))
        self.fuse = nn.Sequential(ConvBNAct(4 * w, 2 * w, 3, 1), DepthwiseSeparableBlock(2 * w, 2 * w))
        self.head = nn.Conv2d(2 * w, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_size = x.shape[-2:]
        x1 = self.stem(x)
        x2 = self.s2(x1)
        x4 = self.s4(x2)
        low = self.low_proj(x2)
        high = self.aspp(x4)
        high = F.interpolate(high, size=low.shape[-2:], mode="bilinear", align_corners=False)
        y = self.fuse(torch.cat([low, high], dim=1))
        return F.interpolate(self.head(y), size=out_size, mode="bilinear", align_corners=False)


class DDRLiteSeg(nn.Module):
    """DDRNet/PIDNet-inspired dual-resolution real-time segmenter.

    The high-resolution branch protects thin structures and boundaries while a
    lower-resolution context branch supplies semantics. It is intentionally
    implemented locally so the experiment remains reproducible without extra
    model-zoo dependencies.
    """

    def __init__(self, in_channels: int, num_classes: int, width: int = 64):
        super().__init__()
        w = int(max(32, width))
        self.stem = nn.Sequential(ConvBNAct(in_channels, w, 3, 2), ConvBNAct(w, w, 3, 1))
        self.hr = nn.Sequential(
            DepthwiseSeparableBlock(w, w, 1),
            DepthwiseSeparableBlock(w, w, 1),
            DepthwiseSeparableBlock(w, w, 1),
        )
        self.lr1 = nn.Sequential(DepthwiseSeparableBlock(w, 2 * w, 2), DepthwiseSeparableBlock(2 * w, 2 * w, 1))
        self.lr2 = nn.Sequential(DepthwiseSeparableBlock(2 * w, 4 * w, 2), DepthwiseSeparableBlock(4 * w, 4 * w, 1))
        self.lr_context = ASPP(4 * w, 2 * w, dilations=(1, 2, 4, 8))
        self.lr_to_hr = ConvBNAct(2 * w, w, 1, 1)
        self.hr_to_lr = ConvBNAct(w, 2 * w, 3, 2)
        self.boundary = nn.Sequential(ConvBNAct(w, w, 3, 1), nn.Conv2d(w, 1, 1), nn.Sigmoid())
        self.fuse = nn.Sequential(
            ConvBNAct(2 * w, 2 * w, 3, 1),
            DepthwiseSeparableBlock(2 * w, 2 * w, 1),
            nn.Dropout2d(0.05),
        )
        self.head = nn.Conv2d(2 * w, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_size = x.shape[-2:]
        stem = self.stem(x)
        hr = self.hr(stem)
        lr = self.lr1(stem)
        lr = lr + self.hr_to_lr(hr)
        lr = self.lr2(lr)
        lr = self.lr_context(lr)
        lr_up = F.interpolate(self.lr_to_hr(lr), size=hr.shape[-2:], mode="bilinear", align_corners=False)
        edge_gate = 0.75 + 0.5 * self.boundary(hr)
        fused = self.fuse(torch.cat([hr * edge_gate, lr_up], dim=1))
        return F.interpolate(self.head(fused), size=out_size, mode="bilinear", align_corners=False)


class BiSeLiteSeg(nn.Module):
    """BiSeNet-style spatial/context segmenter for fast driving baselines."""

    def __init__(self, in_channels: int, num_classes: int, width: int = 64):
        super().__init__()
        w = int(max(32, width))
        self.spatial = nn.Sequential(
            ConvBNAct(in_channels, w, 3, 2),
            ConvBNAct(w, w, 3, 1),
            DepthwiseSeparableBlock(w, w, 1),
        )
        self.context = nn.Sequential(
            ConvBNAct(in_channels, w, 3, 2),
            DepthwiseSeparableBlock(w, 2 * w, 2),
            DepthwiseSeparableBlock(2 * w, 4 * w, 2),
            ASPP(4 * w, 2 * w, dilations=(1, 3, 6)),
        )
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(3 * w, 3 * w, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(3 * w, 3 * w, 1),
            nn.Sigmoid(),
        )
        self.fuse = nn.Sequential(
            ConvBNAct(3 * w, 2 * w, 3, 1),
            DepthwiseSeparableBlock(2 * w, 2 * w, 1),
            nn.Dropout2d(0.05),
        )
        self.head = nn.Conv2d(2 * w, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_size = x.shape[-2:]
        sp = self.spatial(x)
        ctx = self.context(x)
        ctx = F.interpolate(ctx, size=sp.shape[-2:], mode="bilinear", align_corners=False)
        fused = torch.cat([sp, ctx], dim=1)
        fused = fused * self.attn(fused)
        fused = self.fuse(fused)
        return F.interpolate(self.head(fused), size=out_size, mode="bilinear", align_corners=False)


class InputAdapter(nn.Module):
    """Adapt arbitrary channel counts to RGB-pretrained model inputs."""

    def __init__(self, in_channels: int, out_channels: int = 3):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        with torch.no_grad():
            self.proj.weight.zero_()
            if in_channels == 1 and out_channels == 3:
                self.proj.weight[:, 0, 0, 0] = 1.0
            elif in_channels == 4 and out_channels == 3:
                # DeepLens-style packed RGGB starts as a physically meaningful
                # RGB proxy: R -> red, average both greens -> green, B -> blue.
                self.proj.weight[0, 0, 0, 0] = 1.0
                self.proj.weight[1, 1, 0, 0] = 0.5
                self.proj.weight[1, 2, 0, 0] = 0.5
                self.proj.weight[2, 3, 0, 0] = 1.0
            elif in_channels == 5 and out_channels == 3:
                # RGGB + ISO. The ISO channel is metadata for the adapter to
                # learn from, but the pretrained path begins from RGB-equivalent
                # behavior instead of averaging ISO into color.
                self.proj.weight[0, 0, 0, 0] = 1.0
                self.proj.weight[1, 1, 0, 0] = 0.5
                self.proj.weight[1, 2, 0, 0] = 0.5
                self.proj.weight[2, 3, 0, 0] = 1.0
            elif in_channels == out_channels:
                for c in range(in_channels):
                    self.proj.weight[c, c, 0, 0] = 1.0
            else:
                self.proj.weight.fill_(1.0 / max(1, in_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class TorchVisionSegWrapper(nn.Module):
    def __init__(self, model: nn.Module, in_channels: int):
        super().__init__()
        self.adapter = nn.Identity() if in_channels == 3 else InputAdapter(in_channels, 3)
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.model(self.adapter(x))
        if isinstance(y, dict):
            return y["out"]
        return y


class HFSegmentationWrapper(nn.Module):
    def __init__(self, model: nn.Module, in_channels: int, normalize_input: bool = False):
        super().__init__()
        self.adapter = nn.Identity() if in_channels == 3 else InputAdapter(in_channels, 3)
        self.model = model
        self.normalize_input = bool(normalize_input)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_size = x.shape[-2:]
        x = self.adapter(x)
        if self.normalize_input:
            x = (x - self.mean.to(x.device, x.dtype)) / self.std.to(x.device, x.dtype)
        y = self.model(pixel_values=x)
        logits = y.logits if hasattr(y, "logits") else y[0]
        if logits.shape[-2:] != out_size:
            logits = F.interpolate(logits, size=out_size, mode="bilinear", align_corners=False)
        return logits


def _torchvision_seg(name: str, num_classes: int, in_channels: int, pretrained_backbone: bool) -> nn.Module:
    import torchvision.models.segmentation as seg

    weights_backbone = "DEFAULT" if pretrained_backbone else None
    if name == "lraspp_mobilenet_v3_large":
        model = seg.lraspp_mobilenet_v3_large(
            weights=None, weights_backbone=weights_backbone, num_classes=num_classes
        )
    elif name == "deeplabv3_mobilenet_v3_large":
        model = seg.deeplabv3_mobilenet_v3_large(
            weights=None, weights_backbone=weights_backbone, num_classes=num_classes
        )
    elif name == "deeplabv3_resnet50":
        model = seg.deeplabv3_resnet50(
            weights=None, weights_backbone=weights_backbone, num_classes=num_classes
        )
    elif name == "fcn_resnet50":
        model = seg.fcn_resnet50(
            weights=None, weights_backbone=weights_backbone, num_classes=num_classes
        )
    else:
        raise KeyError(name)
    return TorchVisionSegWrapper(model, in_channels=in_channels)


def _hf_segformer(cfg: Dict[str, Any], num_classes: int, in_channels: int) -> nn.Module:
    try:
        from transformers import AutoConfig, AutoModelForSemanticSegmentation, SegformerConfig, SegformerForSemanticSegmentation
    except Exception as exc:
        raise ImportError("Install `transformers` to use Hugging Face segmentation backbones.") from exc

    model_id = cfg.get("hf_model_id", "nvidia/segformer-b0-finetuned-cityscapes-1024-1024")
    pretrained = bool(cfg.get("pretrained", False))
    if pretrained:
        local_pair = _cached_hf_config_and_safetensors(str(model_id))
        if bool(cfg.get("prefer_local_safetensors", True)) and local_pair is not None:
            config_dir, weights_dir = local_pair
            hf_cfg = AutoConfig.from_pretrained(config_dir, num_labels=num_classes, local_files_only=True)
            model = AutoModelForSemanticSegmentation.from_pretrained(
                weights_dir,
                config=hf_cfg,
                ignore_mismatched_sizes=True,
                use_safetensors=True,
                local_files_only=True,
            )
        else:
            try:
                model = AutoModelForSemanticSegmentation.from_pretrained(
                    model_id,
                    num_labels=num_classes,
                    ignore_mismatched_sizes=True,
                    use_safetensors=True,
                    local_files_only=bool(cfg.get("local_files_only", False)),
                )
            except Exception:
                model = AutoModelForSemanticSegmentation.from_pretrained(
                    model_id,
                    num_labels=num_classes,
                    ignore_mismatched_sizes=True,
                    local_files_only=bool(cfg.get("local_files_only", False)),
                )
    else:
        try:
            hf_cfg = AutoConfig.from_pretrained(model_id, num_labels=num_classes, local_files_only=bool(cfg.get("local_files_only", True)))
            model = AutoModelForSemanticSegmentation.from_config(hf_cfg)
        except Exception:
            hf_cfg = SegformerConfig(
                num_labels=num_classes,
                depths=[2, 2, 2, 2],
                hidden_sizes=[32, 64, 160, 256],
                decoder_hidden_size=256,
            )
            model = SegformerForSemanticSegmentation(hf_cfg)
    return HFSegmentationWrapper(
        model,
        in_channels=in_channels,
        normalize_input=bool(cfg.get("normalize_input", pretrained)),
    )


def _hf_auto_segmentation(cfg: Dict[str, Any], num_classes: int, in_channels: int) -> nn.Module:
    try:
        from transformers import AutoConfig, AutoModelForSemanticSegmentation
    except Exception as exc:
        raise ImportError("Install `transformers` to use Hugging Face segmentation backbones.") from exc

    model_id = cfg.get("hf_model_id", "facebook/mask2former-swin-tiny-cityscapes-semantic")
    pretrained = bool(cfg.get("pretrained", True))
    if pretrained:
        local_pair = _cached_hf_config_and_safetensors(str(model_id))
        if bool(cfg.get("prefer_local_safetensors", True)) and local_pair is not None:
            config_dir, weights_dir = local_pair
            hf_cfg = AutoConfig.from_pretrained(config_dir, num_labels=num_classes, local_files_only=True)
            model = AutoModelForSemanticSegmentation.from_pretrained(
                weights_dir,
                config=hf_cfg,
                ignore_mismatched_sizes=True,
                use_safetensors=True,
                local_files_only=True,
            )
        else:
            try:
                model = AutoModelForSemanticSegmentation.from_pretrained(
                    model_id,
                    num_labels=num_classes,
                    ignore_mismatched_sizes=True,
                    use_safetensors=True,
                    local_files_only=bool(cfg.get("local_files_only", False)),
                )
            except Exception:
                model = AutoModelForSemanticSegmentation.from_pretrained(
                    model_id,
                    num_labels=num_classes,
                    ignore_mismatched_sizes=True,
                    local_files_only=bool(cfg.get("local_files_only", False)),
                )
    else:
        hf_cfg = AutoConfig.from_pretrained(
            model_id,
            num_labels=num_classes,
            local_files_only=bool(cfg.get("local_files_only", True)),
        )
        model = AutoModelForSemanticSegmentation.from_config(hf_cfg)
    return HFSegmentationWrapper(
        model,
        in_channels=in_channels,
        normalize_input=bool(cfg.get("normalize_input", pretrained)),
    )


def _lraspp_efficientnet(cfg: Dict[str, Any], num_classes: int, in_channels: int) -> nn.Module:
    try:
        import timm
    except Exception as exc:
        raise ImportError("Install `timm` to use EfficientNet backbones for LR-ASPP.") from exc

    model_name = cfg.get("backbone", cfg.get("timm_model", "efficientnet_b0"))
    pretrained_backbone = bool(cfg.get("pretrained_backbone", False))

    # Create a feature-extracting backbone (returns a list of feature maps)
    backbone = timm.create_model(model_name, pretrained=pretrained_backbone, features_only=True)
    feat_channels = list(backbone.feature_info.channels())

    class LRASPPHead(nn.Module):
        def __init__(self, low_ch: int, high_ch: int, num_classes: int):
            super().__init__()
            self.low_proj = nn.Sequential(nn.Conv2d(low_ch, 48, 1, bias=False), nn.BatchNorm2d(48), nn.SiLU(inplace=True))
            self.high_proj = nn.Sequential(nn.Conv2d(high_ch, 256, 1, bias=False), nn.BatchNorm2d(256), nn.SiLU(inplace=True))
            self.context = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(high_ch, 256, 1), nn.SiLU(inplace=True), nn.Conv2d(256, 256, 1), nn.Sigmoid())
            self.fuse = nn.Sequential(ConvBNAct(48 + 256, 256, 3, 1), nn.Conv2d(256, num_classes, 1))

        def forward(self, low: torch.Tensor, high: torch.Tensor, out_size: tuple) -> torch.Tensor:
            low_p = self.low_proj(low)
            high_c = high * self.context(high)
            high_p = self.high_proj(high_c)
            high_up = F.interpolate(high_p, size=low_p.shape[-2:], mode="bilinear", align_corners=False)
            y = torch.cat([low_p, high_up], dim=1)
            y = self.fuse(y)
            return F.interpolate(y, size=out_size, mode="bilinear", align_corners=False)

    class TimmLRASPP(nn.Module):
        def __init__(self, backbone, feat_channels, in_channels, num_classes):
            super().__init__()
            self.adapter = nn.Identity() if in_channels == 3 else InputAdapter(in_channels, 3)
            self.backbone = backbone
            self.feat_channels = feat_channels
            low_ch = feat_channels[0]
            high_ch = feat_channels[-1]
            self.head = LRASPPHead(low_ch, high_ch, num_classes)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out_size = x.shape[-2:]
            feats = self.backbone(self.adapter(x))
            if not isinstance(feats, (list, tuple)) or len(feats) < 2:
                # fallback: treat single feature map as high-level only
                high = feats if isinstance(feats, torch.Tensor) else feats[-1]
                low = high
            else:
                low = feats[0]
                high = feats[-1]
            return self.head(low, high, out_size)

    return TimmLRASPP(backbone, feat_channels, in_channels=in_channels, num_classes=num_classes)


def build_segmentation_model(cfg: Dict[str, Any], in_channels: int, num_classes: int) -> nn.Module:
    mcfg = cfg.get("model", {})
    name = str(mcfg.get("name", "unet_tiny")).lower()
    width = int(mcfg.get("width", 32))

    if name in ("unet", "unet_tiny", "tiny_unet"):
        return UNetTiny(in_channels=in_channels, num_classes=num_classes, base=width)
    if name in ("liteseg", "lite_seg", "raw_liteseg"):
        return LiteSeg(in_channels=in_channels, num_classes=num_classes, width=width)
    if name in ("liteseg_aspp", "lite_seg_aspp", "raw_liteseg_aspp"):
        return LiteSegASPP(in_channels=in_channels, num_classes=num_classes, width=max(width, 64))
    if name in ("ddr_lite", "ddrlite", "ddrnet_lite", "pidnet_lite", "susc_snet_lite"):
        return DDRLiteSeg(in_channels=in_channels, num_classes=num_classes, width=max(width, 64))
    if name in ("bise_lite", "biselite", "bisenet_lite", "bisenetv2_lite"):
        return BiSeLiteSeg(in_channels=in_channels, num_classes=num_classes, width=max(width, 64))
    if name in {
        "lraspp_mobilenet_v3_large",
        "deeplabv3_mobilenet_v3_large",
        "deeplabv3_resnet50",
        "fcn_resnet50",
    }:
        return _torchvision_seg(
            name,
            num_classes=num_classes,
            in_channels=in_channels,
            pretrained_backbone=bool(mcfg.get("pretrained_backbone", False)),
        )
    if name in ("segformer", "segformer_b0", "hf_segformer"):
        return _hf_segformer(mcfg, num_classes=num_classes, in_channels=in_channels)
    if name in ("segformer_b1",):
        # default HF model id for SegFormer-B1 finetuned on Cityscapes
        mcfg.setdefault("hf_model_id", "nvidia/segformer-b1-finetuned-cityscapes-1024-1024")
        return _hf_segformer(mcfg, num_classes=num_classes, in_channels=in_channels)
    if name in ("segformer_b2",):
        # default HF model id for SegFormer-B2 finetuned on Cityscapes
        mcfg.setdefault("hf_model_id", "nvidia/segformer-b2-finetuned-cityscapes-1024-1024")
        return _hf_segformer(mcfg, num_classes=num_classes, in_channels=in_channels)
    if name.startswith("lraspp_efficientnet"):
        # allow names like lraspp_efficientnet_b0 or lraspp_efficientnet_b1
        # map to timm model name: efficientnet_b0, efficientnet_b1, etc.
        parts = name.split("_")
        # default to efficientnet_b0
        timm_model = "efficientnet_b0"
        for p in parts:
            if p.startswith("b") and p[1:].isdigit():
                timm_model = f"efficientnet_{p}"
        mcfg.setdefault("timm_model", timm_model)
        return _lraspp_efficientnet(mcfg, num_classes=num_classes, in_channels=in_channels)
    if name in ("hf_auto", "mask2former", "oneformer", "auto_semantic"):
        return _hf_auto_segmentation(mcfg, num_classes=num_classes, in_channels=in_channels)

    raise ValueError(
        f"Unknown segmentation model {name!r}. Available: unet_tiny, liteseg, "
        "liteseg_aspp, ddrnet_lite, bisenet_lite, lraspp_mobilenet_v3_large, deeplabv3_mobilenet_v3_large, "
        "deeplabv3_resnet50, fcn_resnet50, segformer, hf_auto."
    )
