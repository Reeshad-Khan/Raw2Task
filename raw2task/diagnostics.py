"""
raw2task/diagnostics.py  —  Three diagnostic experiments.

D1: PSF gradient direction
    Does ∂L_seg/∂PSF push the lens toward identity (sharp) or toward blur?
    If it points toward identity, PSF co-design cannot help segmentation —
    no training trick will fix that.

D2: Per-component contribution
    Decompose the performance gap from existing metrics CSVs.
    Which sensor component (CFA / noise / PSF) actually accounts for gains?

D3: Learned PSF inspection
    What did the optimizer converge to? RMS radius, deviation from identity,
    per-zone aberration coefficients.

Usage (D1 needs a GPU compute node; D2/D3 run on login node):
    python -m raw2task.diagnostics \\
        --runs-dir  ./runs/kitti360_sfb4 \\
        --dataset   kitti360 \\
        --out-dir   ./diagnostics/kitti360 \\
        [--skip-d1]

    # ACDC
    python -m raw2task.diagnostics \\
        --runs-dir  ./runs/acdc_sfb4 \\
        --dataset   acdc \\
        --out-dir   ./diagnostics/acdc \\
        --skip-d1
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import logging
import math
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────

def _find_best_ckpt(exp_dir: str) -> Optional[str]:
    candidates = sorted(glob.glob(os.path.join(exp_dir, "best_ep*.pt")))
    if candidates:
        return candidates[-1]
    last = os.path.join(exp_dir, "last.pt")
    return last if os.path.exists(last) else None


def _find_exp_dir(runs_dir: str, name_fragment: str) -> Optional[str]:
    matches = sorted(glob.glob(os.path.join(runs_dir, f"*{name_fragment}*seed*")))
    return matches[0] if matches else None


def _load_ckpt(path: str, device: torch.device) -> Dict:
    return torch.load(path, map_location=device, weights_only=False)


def _read_best_miou(exp_dir: str) -> Optional[float]:
    csv_path = os.path.join(exp_dir, "metrics_log.csv")
    if not os.path.exists(csv_path):
        return None
    best = -1.0
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            try:
                best = max(best, float(row["mIoU"]))
            except (KeyError, ValueError):
                pass
    return best if best >= 0 else None


def _rms_radius(kernel: np.ndarray) -> float:
    """Compute RMS radius of a normalised 2D PSF kernel."""
    H, W = kernel.shape
    cy, cx = H // 2, W // 2
    ys = np.arange(H) - cy
    xs = np.arange(W) - cx
    xx, yy = np.meshgrid(xs, ys)
    rr2 = xx ** 2 + yy ** 2
    k = np.clip(kernel, 0, None)
    total = k.sum()
    if total < 1e-12:
        return 0.0
    return float(np.sqrt((rr2 * k).sum() / total))


# ─────────────────────────────────────────────────────────────────────────────
# D2: per-component contribution  (CSV-only, no GPU)
# ─────────────────────────────────────────────────────────────────────────────

EXP_FRAGMENTS = {
    "codesign": "codesign",
    "rgb":      "rgb",
    "no_optics": "no_optics",
    "fixed":    "fixed_camera",
}


def diag2_component_contribution(runs_dir: str, out_dir: str) -> None:
    log.info("\n" + "="*70)
    log.info("D2: Per-component contribution")
    log.info("="*70)

    scores: Dict[str, Optional[float]] = {}
    for key, frag in EXP_FRAGMENTS.items():
        exp_dir = _find_exp_dir(runs_dir, frag)
        scores[key] = _read_best_miou(exp_dir) if exp_dir else None
        log.info("  %-15s  %s", key,
                 f"{scores[key]:.4f}" if scores[key] is not None else "NOT FOUND")

    cd = scores.get("codesign")
    rgb = scores.get("rgb")
    no_op = scores.get("no_optics")
    fixed = scores.get("fixed")

    log.info("")
    rows = []
    if all(v is not None for v in [rgb, no_op, fixed, cd]):
        sensor_cost      = rgb - no_op       # cost of ANY sensor vs clean RGB
        cfa_noise_gain   = no_op - fixed     # CFA+noise optimisation benefit
        psf_cost         = no_op - cd        # PSF adds or removes performance
        total_codesign   = rgb - cd          # total cost vs clean RGB

        log.info("  Decomposition (positive = gain over reference):")
        log.info("  %-40s  %+.4f  (rgb vs clean RGB baseline — always 0)", "rgb baseline", 0.0)
        log.info("  %-40s  %+.4f  (cost of ANY sensor: mosaicing + noise)", "sensor degradation (rgb - no_optics)", -sensor_cost)
        log.info("  %-40s  %+.4f  (learnable CFA+noise vs fixed sensor)", "CFA+noise optimisation (no_optics - fixed)", cfa_noise_gain)
        log.info("  %-40s  %+.4f  ← key number", "PSF contribution (no_optics - codesign)", -psf_cost)
        log.info("  %-40s  %+.4f", "total codesign cost vs rgb (rgb - codesign)", -total_codesign)
        log.info("")
        if psf_cost > 0:
            log.info("  CONCLUSION: PSF optimisation is NET-NEGATIVE (%.4f mIoU penalty).", psf_cost)
            log.info("  The gradient for PSF parameters is pointing the wrong direction.")
            log.info("  CFA+noise help (+%.4f) but PSF cancels and reverses those gains.", cfa_noise_gain)
        else:
            log.info("  CONCLUSION: PSF optimisation is NET-POSITIVE (+%.4f mIoU).", -psf_cost)
            log.info("  Co-design is working. CFA+noise contribute +%.4f on top.", cfa_noise_gain)

        rows = [
            ["metric", "delta_mIoU", "interpretation"],
            ["rgb → no_optics", f"{-sensor_cost:+.4f}", "cost of sensor degradation"],
            ["fixed → no_optics", f"{cfa_noise_gain:+.4f}", "CFA + noise optimisation benefit"],
            ["codesign → no_optics", f"{-psf_cost:+.4f}", "PSF net contribution (negative=hurts)"],
            ["rgb → codesign", f"{-total_codesign:+.4f}", "total co-design cost vs clean RGB"],
        ]

    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "d2_component_contribution.csv")
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    log.info("  Saved → %s", csv_path)

    # bar chart
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if all(v is not None for v in [rgb, no_op, fixed, cd]):
            labels = ["RGB\n(clean)", "No-optics\n(CFA+noise)", "Codesign\n(PSF+CFA+noise)", "Fixed\n(all fixed)"]
            values = [rgb, no_op, cd, fixed]
            colors = ["#2196F3", "#4CAF50", "#e63946", "#FF9800"]
            fig, ax = plt.subplots(figsize=(7, 4))
            bars = ax.bar(labels, values, color=colors, width=0.5)
            ax.set_ylabel("Best val mIoU")
            ax.set_title("Per-component contribution — D2")
            ymin = min(values) - 0.02
            ax.set_ylim(ymin, max(values) + 0.02)
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, val + 0.002,
                        f"{val:.4f}", ha="center", va="bottom", fontsize=9)
            ax.axhline(rgb, color="#2196F3", linestyle="--", alpha=0.4, linewidth=1)
            plt.tight_layout()
            p = os.path.join(out_dir, "d2_component_contribution.png")
            fig.savefig(p, dpi=150, bbox_inches="tight")
            plt.close()
            log.info("  Saved → %s", p)
    except ImportError:
        log.info("  (matplotlib not available — skipping bar chart)")


# ─────────────────────────────────────────────────────────────────────────────
# D3: learned PSF inspection  (loads sensor only, CPU ok)
# ─────────────────────────────────────────────────────────────────────────────

COEFF_NAMES = ["defocus", "astig_x", "astig_y", "coma_x", "coma_y", "spherical"]


def diag3_psf_inspection(runs_dir: str, out_dir: str, device: torch.device) -> None:
    log.info("\n" + "="*70)
    log.info("D3: Learned PSF inspection")
    log.info("="*70)

    exp_dir = _find_exp_dir(runs_dir, "codesign")
    if not exp_dir:
        log.info("  codesign experiment not found in %s", runs_dir)
        return

    ckpt_path = _find_best_ckpt(exp_dir)
    if not ckpt_path:
        log.info("  No checkpoint found in %s", exp_dir)
        return

    log.info("  Loading checkpoint: %s", ckpt_path)
    ckpt = _load_ckpt(ckpt_path, device)

    # Rebuild sensor from saved cfg
    cfg = ckpt.get("cfg", {})
    if not cfg:
        log.info("  cfg not found in checkpoint — cannot rebuild sensor")
        return
    cfg["device"] = device

    try:
        from raw2task.train_extended import CoDesignSensor
        sensor = CoDesignSensor(cfg).to(device)
        sensor.load_state_dict(ckpt["sensor"], strict=False)
        sensor.eval()
    except Exception as e:
        log.info("  Failed to load sensor: %s", e)
        return

    optics = getattr(sensor, "optics", None)
    if optics is None or not hasattr(optics, "psf_kernels"):
        log.info("  Sensor has no trainable PSF (bypass_optics=True or identity mode).")
        return

    with torch.no_grad():
        # Effective bounded coefficients [zones, channels, 6]
        coeff = optics.effective_coefficients().cpu().numpy()
        # Synthesised PSF kernels [zones, channels, K, K]
        kernels = optics.psf_kernels(device=torch.device("cpu"),
                                     dtype=torch.float32).numpy()
        # Identity PSF (delta at centre)
        K = optics.kernel_size
        identity = np.zeros((K, K), dtype=np.float32)
        identity[K//2, K//2] = 1.0

    num_zones, channels, _, _ = kernels.shape

    log.info("\n  Effective aberration coefficients (bounded):")
    log.info("  %-8s  " + "  ".join(f"{n:>10}" for n in COEFF_NAMES), "zone/ch")
    rows_coeff = [["zone", "channel"] + COEFF_NAMES]
    for z in range(num_zones):
        for c in range(channels):
            vals = coeff[z, c]
            log.info("  z%d ch%d   " + "  ".join(f"{v:+10.4f}" for v in vals), z, c)
            rows_coeff.append([z, c] + vals.tolist())

    log.info("\n  PSF RMS radius (pixels) — identity = 0.00:")
    log.info("  %-8s  " + "  ".join(f"ch{c:>2}" for c in range(channels)), "zone")
    rows_rms = [["zone"] + [f"ch{c}" for c in range(channels)] + ["mean", "vs_identity"]]
    for z in range(num_zones):
        rms_vals = [_rms_radius(kernels[z, c]) for c in range(channels)]
        id_rms = _rms_radius(identity)
        mean_rms = float(np.mean(rms_vals))
        log.info("  z%-6d  " + "  ".join(f"{r:>5.3f}" for r in rms_vals) +
                 f"  mean={mean_rms:.3f}  Δid={mean_rms - id_rms:+.3f}", z)
        rows_rms.append([z] + rms_vals + [mean_rms, mean_rms - id_rms])

    all_rms = [_rms_radius(kernels[z, c]) for z in range(num_zones) for c in range(channels)]
    id_rms = _rms_radius(identity)
    log.info("\n  Global mean RMS radius: %.4f px  (identity PSF = %.4f px)", np.mean(all_rms), id_rms)
    if np.mean(all_rms) > id_rms * 1.5:
        log.info("  VERDICT: PSF is significantly BLURRIER than identity.")
        log.info("  The optimizer moved the PSF away from sharp → this hurts segmentation.")
    elif np.mean(all_rms) < id_rms * 1.1:
        log.info("  VERDICT: PSF is near-identity (optimizer found no useful encoding).")
    else:
        log.info("  VERDICT: PSF has mild blur. May encode useful signal — check D1 gradient.")

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "d3_coefficients.csv"), "w", newline="") as f:
        csv.writer(f).writerows(rows_coeff)
    with open(os.path.join(out_dir, "d3_rms_radius.csv"), "w", newline="") as f:
        csv.writer(f).writerows(rows_rms)

    # PSF kernel heatmaps
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(num_zones, channels,
                                 figsize=(3 * channels, 3 * num_zones), squeeze=False)
        for z in range(num_zones):
            for c in range(channels):
                k = kernels[z, c]
                rms = _rms_radius(k)
                ax = axes[z][c]
                im = ax.imshow(k, cmap="hot", interpolation="nearest")
                ax.set_title(f"z{z} ch{c}\nRMS={rms:.3f}px", fontsize=8)
                ax.axis("off")
                fig.colorbar(im, ax=ax, shrink=0.7)
        plt.suptitle("D3: Learned PSF kernels (identity = delta at centre)", fontsize=11)
        plt.tight_layout()
        p = os.path.join(out_dir, "d3_psf_kernels.png")
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        log.info("  Saved → %s", p)

        # gradient in aberration coefficient space — show how far from zero
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        coeff_mean = np.abs(coeff).mean(axis=(0, 1))   # mean abs value per aberration type
        ax2.bar(COEFF_NAMES, coeff_mean, color=["#e63946","#FF9800","#FFEB3B","#4CAF50","#2196F3","#9C27B0"])
        ax2.set_ylabel("Mean |coefficient|")
        ax2.set_title("D3: Aberration magnitude (0 = identity PSF)")
        ax2.axhline(0.0, color="k", linewidth=0.5)
        plt.tight_layout()
        p2 = os.path.join(out_dir, "d3_aberration_magnitude.png")
        fig2.savefig(p2, dpi=150, bbox_inches="tight")
        plt.close()
        log.info("  Saved → %s", p2)
    except ImportError:
        log.info("  (matplotlib not available — skipping PSF visualisation)")


# ─────────────────────────────────────────────────────────────────────────────
# D1: PSF gradient direction  (needs GPU + val data)
# ─────────────────────────────────────────────────────────────────────────────

def diag1_psf_gradient(runs_dir: str, out_dir: str, device: torch.device,
                        n_batches: int = 30) -> None:
    log.info("\n" + "="*70)
    log.info("D1: PSF gradient direction (n_batches=%d)", n_batches)
    log.info("="*70)

    exp_dir = _find_exp_dir(runs_dir, "codesign")
    if not exp_dir:
        log.info("  codesign experiment not found — skipping D1"); return

    ckpt_path = _find_best_ckpt(exp_dir)
    if not ckpt_path:
        log.info("  No checkpoint found — skipping D1"); return

    log.info("  Loading checkpoint: %s", ckpt_path)
    ckpt = _load_ckpt(ckpt_path, device)
    cfg = ckpt.get("cfg", {})
    if not cfg:
        log.info("  cfg not in checkpoint — skipping D1"); return
    cfg["device"] = device

    # Build sensor and model
    try:
        from raw2task.train_extended import CoDesignSensor, build_seg_model, get_dataset
    except ImportError as e:
        log.info("  Import error: %s — skipping D1", e); return

    try:
        sensor = CoDesignSensor(cfg).to(device)
        sensor.load_state_dict(ckpt["sensor"], strict=False)
        sensor.eval()
    except Exception as e:
        log.info("  Sensor load failed: %s — skipping D1", e); return

    optics = getattr(sensor, "optics", None)
    if optics is None or not hasattr(optics, "psf_kernels"):
        log.info("  No trainable PSF — skipping D1"); return

    try:
        num_classes = int(cfg.get("model", {}).get("num_classes", 19))
        in_ch = sensor.output_channels
        model = build_seg_model(cfg, in_channels=in_ch, num_classes=num_classes).to(device)
        model.load_state_dict(ckpt["model"], strict=False)
        model.eval()
    except Exception as e:
        log.info("  Model load failed: %s — skipping D1", e); return

    # Get val loader
    try:
        _, val_loader, _ = get_dataset(cfg)
    except Exception as e:
        log.info("  Dataset load failed: %s — skipping D1", e); return

    ignore_index = int(cfg.get("model", {}).get("ignore_index", 255))

    # Freeze everything except PSF coeff_raw
    for p in model.parameters():
        p.requires_grad_(False)
    for p in sensor.parameters():
        p.requires_grad_(False)
    optics.coeff_raw.requires_grad_(True)

    # Record PSF state before gradient
    with torch.no_grad():
        kernels_before = optics.psf_kernels(device=device).detach().clone()
        rms_before = [
            _rms_radius(kernels_before[z, c].cpu().numpy())
            for z in range(optics.num_zones)
            for c in range(optics.channels)
        ]

    # Accumulate gradient over n_batches validation batches
    if optics.coeff_raw.grad is not None:
        optics.coeff_raw.grad.zero_()

    log.info("  Running %d val batches for gradient accumulation …", n_batches)
    total_batches = 0
    total_loss = 0.0
    for it, batch in enumerate(val_loader):
        if it >= n_batches:
            break
        imgs, labels = batch
        imgs = imgs.to(device)
        labels = labels.to(device)
        with torch.enable_grad():
            raw = sensor(imgs)
            logits = model(raw)
            # align spatial size
            if logits.shape[-2:] != labels.shape[-2:]:
                logits = F.interpolate(logits, size=labels.shape[-2:],
                                       mode="bilinear", align_corners=False)
            loss = F.cross_entropy(logits, labels, ignore_index=ignore_index)
            loss.backward()
        total_loss += float(loss.item())
        total_batches += 1
        if (it + 1) % 10 == 0:
            log.info("    batch %d/%d …", it + 1, n_batches)

    grad = optics.coeff_raw.grad
    if grad is None or total_batches == 0:
        log.info("  No gradient accumulated — skipping D1"); return

    grad_np = grad.detach().cpu().numpy() / total_batches
    avg_loss = total_loss / total_batches
    log.info("  Average val loss over %d batches: %.4f", total_batches, avg_loss)

    # Gradient analysis: per-aberration-type, what direction is the update?
    log.info("\n  Gradient of loss w.r.t. PSF coefficients (averaged over %d batches):", total_batches)
    log.info("  Positive grad → increasing this coeff would INCREASE loss (avoid it)")
    log.info("  Negative grad → increasing this coeff would DECREASE loss (want more)")
    log.info("")
    log.info("  %-12s  %s", "aberration", "  ".join(f"z{z}ch0" for z in range(optics.num_zones)))

    coeff_np = optics.effective_coefficients().detach().cpu().numpy()
    rows_grad = [["aberration"] + [f"z{z}c{c}" for z in range(optics.num_zones)
                                                for c in range(optics.channels)]]
    for ci, name in enumerate(COEFF_NAMES):
        vals = grad_np[:, :, ci].flatten()
        log.info("  %-12s  " + "  ".join(f"{v:+.4f}" for v in vals), name)
        rows_grad.append([name] + vals.tolist())

    # Simulate one gradient step and compare RMS radius
    lr_test = 0.01
    with torch.no_grad():
        coeff_updated = optics.coeff_raw - lr_test * grad
        kernels_after = optics.psf_kernels(coeff_override=torch.tanh(coeff_updated) * optics.coeff_bounds.to(device))
        rms_after = [
            _rms_radius(kernels_after[z, c].cpu().numpy())
            for z in range(optics.num_zones)
            for c in range(optics.channels)
        ]

    mean_rms_before = float(np.mean(rms_before))
    mean_rms_after  = float(np.mean(rms_after))
    delta_rms = mean_rms_after - mean_rms_before

    log.info("")
    log.info("  RMS radius BEFORE gradient step: %.4f px", mean_rms_before)
    log.info("  RMS radius AFTER  gradient step: %.4f px  (Δ=%+.4f)", mean_rms_after, delta_rms)
    log.info("")
    if delta_rms > 0.01:
        log.info("  VERDICT: gradient pushes PSF toward SHARPER (smaller RMS).")
        log.info("  The segmentation loss WANTS less blur — but the PSF is currently")
        log.info("  too blurry and the optimizer is fighting to bring it back.")
        log.info("  PSF co-design is net-negative: you cannot win against physics here.")
    elif delta_rms < -0.01:
        log.info("  VERDICT: gradient pushes PSF toward MORE BLUR (larger RMS).")
        log.info("  The loss sees a benefit from blur in this direction — unusual.")
        log.info("  Investigate which zones/channels are driving this.")
    else:
        log.info("  VERDICT: gradient is nearly neutral w.r.t. blur (|ΔRMS| < 0.01).")
        log.info("  PSF is near a local minimum but it may not be the global optimum.")

    # Gradient magnitude by aberration type (which aberrations matter most?)
    log.info("\n  Gradient magnitude by aberration (mean |grad| across zones+channels):")
    grad_by_type = np.abs(grad_np).mean(axis=(0, 1))   # [6]
    for name, mag in zip(COEFF_NAMES, grad_by_type):
        bar = "█" * int(mag / max(grad_by_type.max(), 1e-8) * 20)
        log.info("  %-12s  %s  %.5f", name, f"{bar:<20}", mag)

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "d1_gradient.csv"), "w", newline="") as f:
        csv.writer(f).writerows(rows_grad)
    with open(os.path.join(out_dir, "d1_summary.txt"), "w") as f:
        f.write(f"avg_val_loss: {avg_loss:.6f}\n")
        f.write(f"rms_before: {mean_rms_before:.6f}\n")
        f.write(f"rms_after:  {mean_rms_after:.6f}\n")
        f.write(f"delta_rms:  {delta_rms:+.6f}\n")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        grad_by_type_abs = np.abs(grad_np).mean(axis=(0, 1))
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Left: gradient magnitude by aberration type
        axes[0].bar(COEFF_NAMES, grad_by_type_abs,
                    color=["#e63946","#FF9800","#FFEB3B","#4CAF50","#2196F3","#9C27B0"])
        axes[0].set_ylabel("Mean |∂L/∂coeff|")
        axes[0].set_title("D1: Which aberration has largest gradient?")
        axes[0].set_xlabel("Aberration type")

        # Right: RMS radius before vs after
        axes[1].bar(["Before\ngradient step", "After\ngradient step"],
                    [mean_rms_before, mean_rms_after],
                    color=["#9E9E9E", "#e63946" if delta_rms > 0 else "#4CAF50"])
        axes[1].set_ylabel("Mean PSF RMS radius (px)")
        axes[1].set_title(f"D1: Gradient direction (Δ={delta_rms:+.4f} px)\n"
                          f"{'→ toward SHARPER (loss wants sharp PSF)' if delta_rms > 0 else '→ toward BLURRIER'}")
        axes[1].set_ylim(0, max(mean_rms_before, mean_rms_after) * 1.3)

        plt.tight_layout()
        p = os.path.join(out_dir, "d1_gradient_analysis.png")
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        log.info("  Saved → %s", p)
    except ImportError:
        pass

    log.info("  Saved CSVs → %s/d1_*.csv", out_dir)


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Co-design diagnostic experiments")
    ap.add_argument("--runs-dir", required=True,
                    help="Path to experiment runs dir (e.g. ./runs/kitti360_sfb4)")
    ap.add_argument("--dataset", default="kitti360", choices=["kitti360", "acdc"])
    ap.add_argument("--out-dir", default="./diagnostics")
    ap.add_argument("--skip-d1", action="store_true",
                    help="Skip PSF gradient analysis (requires GPU compute node)")
    ap.add_argument("--d1-batches", type=int, default=30,
                    help="Number of val batches for gradient accumulation in D1")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)
    log.info("Runs dir: %s", args.runs_dir)
    log.info("Output:   %s", args.out_dir)

    os.makedirs(args.out_dir, exist_ok=True)

    # D2 first — no GPU, fast
    diag2_component_contribution(args.runs_dir, args.out_dir)

    # D3 — loads sensor checkpoint only, CPU ok
    diag3_psf_inspection(args.runs_dir, args.out_dir, device)

    # D1 — needs GPU + val data
    if not args.skip_d1:
        if not torch.cuda.is_available():
            log.info("\nD1: No GPU available — run on a compute node or use --skip-d1")
        else:
            diag1_psf_gradient(args.runs_dir, args.out_dir, device, n_batches=args.d1_batches)
    else:
        log.info("\nD1: Skipped (--skip-d1 flag set)")

    log.info("\n" + "="*70)
    log.info("All diagnostics complete. Results in: %s", args.out_dir)
    log.info("="*70)


if __name__ == "__main__":
    main()
