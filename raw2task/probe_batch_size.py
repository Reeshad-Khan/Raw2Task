"""Preflight GPU batch-size probe for raw2task experiment matrices.

This script runs synthetic forward/backward steps for each selected matrix
experiment, finds the largest local batch that fits, and writes a fixed
batch-size plan. The training launcher can then consume that plan with
``--batch-plan`` so the real experiments are reproducible and do not mutate
batch size dynamically.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import subprocess
from typing import Any, Dict, Iterable, List

import torch
import torch.nn.functional as F
import yaml

from raw2task.run_end2end_pipeline import _parse_csv_list
from raw2task.run_paper_experiments import _expand_jobs
from raw2task.train_extended import (
    CoDesignSensor,
    PassThroughSensor,
    align_logits_to_labels,
    build_seg_model,
    set_seed,
)


def _gpu_info(gpu: str) -> Dict[str, Any]:
    if str(gpu).lower() == "cpu":
        return {"gpu": "cpu"}
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--id",
                str(gpu),
                "--query-gpu=index,name,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        idx, name, total, free = [x.strip() for x in out.splitlines()[0].split(",")[:4]]
        return {"gpu": idx, "name": name, "memory_total_mb": int(total), "memory_free_mb": int(free)}
    except Exception as exc:
        return {"gpu": str(gpu), "error": str(exc)}


def _normalize_probe_cfg(cfg_in: Dict[str, Any], batch_size: int) -> Dict[str, Any]:
    cfg = copy.deepcopy(cfg_in)
    cfg.setdefault("task", "seg")
    cfg.setdefault("seed", 0)
    cfg.setdefault("pipeline", {}).setdefault("input", "sensor")
    cfg.setdefault("data", {})
    cfg["data"]["batch_size"] = int(batch_size)
    cfg["data"].setdefault("img_size", [192, 640])
    cfg.setdefault("model", {})
    cfg["model"].setdefault("num_classes", 19)
    cfg["model"].setdefault("ignore_index", 255)
    cfg["model"].setdefault("width", 48)
    cfg.setdefault("train", {})
    cfg["train"].setdefault("amp", True)
    cfg["train"].setdefault("channels_last", False)
    cfg.setdefault("sensor", {})
    cfg["sensor"].setdefault("cfa_init", "bayer_rggb")
    cfg["sensor"].setdefault("exposure_init", -1.38629436)
    cfg["sensor"].setdefault("bit_depth", 8)
    cfg["sensor"].setdefault("read_noise_std", 0.003)
    cfg["sensor"].setdefault("shot_noise_scale", 0.02)
    cfg["sensor"].setdefault("learn_cfa", True)
    cfg["sensor"].setdefault("learn_exposure", True)
    cfg["sensor"].setdefault("learn_noise", True)
    cfg["sensor"].setdefault("learn_bit_depth", True)
    cfg["sensor"].setdefault("sensor_pitch", 1.6e-6)
    cfg.setdefault("lens", {})
    cfg["lens"].setdefault("mode", "trainable_psf")
    cfg["lens"].setdefault("learn_optics", True)
    cfg["lens"].setdefault("normalize_output", True)
    cfg["lens"].setdefault("target_mean", 0.5)
    cfg["lens"].setdefault("trainable_psf", {})
    cfg["lens"]["trainable_psf"].setdefault("num_zones", 5)
    cfg["lens"]["trainable_psf"].setdefault("kernel_size", 15)
    cfg["lens"]["trainable_psf"].setdefault("base_sigma_px", 1.05)
    cfg["lens"]["trainable_psf"].setdefault("min_sigma_px", 0.35)
    cfg["lens"]["trainable_psf"].setdefault("max_sigma_px", 3.75)
    cfg["lens"]["trainable_psf"].setdefault("max_shift_px", 1.25)
    cfg["lens"]["trainable_psf"].setdefault("field_sigma", 0.65)
    return cfg


def _build_chain(cfg: Dict[str, Any], device: torch.device):
    input_mode = str((cfg.get("pipeline") or {}).get("input", "sensor")).lower()
    if input_mode in ("rgb", "rgb_baseline", "processed_rgb"):
        sensor = PassThroughSensor(cfg).to(device)
    else:
        sensor = CoDesignSensor(cfg).to(device)
    model = build_seg_model(
        cfg,
        in_channels=int(sensor.output_channels),
        num_classes=int((cfg.get("model") or {}).get("num_classes", 19)),
    ).to(device)
    if bool((cfg.get("train") or {}).get("channels_last", False)):
        sensor = sensor.to(memory_format=torch.channels_last)
        model = model.to(memory_format=torch.channels_last)
    return sensor, model


def _fits(cfg_in: Dict[str, Any], batch_size: int, gpu: str, steps: int, margin_mb: int) -> tuple[bool, str, int]:
    if str(gpu).lower() == "cpu":
        return False, "CPU probing is not supported for GPU batch sizing.", 0
    torch.cuda.set_device(int(gpu))
    device = torch.device(f"cuda:{int(gpu)}")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    cfg = _normalize_probe_cfg(cfg_in, batch_size=batch_size)
    cfg["device"] = device
    set_seed(int(cfg.get("seed", 0)))
    try:
        sensor, model = _build_chain(cfg, device)
        sensor.train()
        model.train()
        h, w = [int(x) for x in (cfg.get("data") or {}).get("img_size", [192, 640])]
        x = torch.rand(batch_size, 3, h, w, device=device)
        if bool((cfg.get("train") or {}).get("channels_last", False)):
            x = x.contiguous(memory_format=torch.channels_last)
        y = torch.randint(
            low=0,
            high=int((cfg.get("model") or {}).get("num_classes", 19)),
            size=(batch_size, h, w),
            device=device,
            dtype=torch.long,
        )
        amp = bool((cfg.get("train") or {}).get("amp", True))
        for _ in range(max(1, int(steps))):
            sensor.zero_grad(set_to_none=True)
            model.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp):
                raw = sensor(x)
                logits = align_logits_to_labels(model(raw), y)
                loss = F.cross_entropy(
                    logits,
                    y,
                    ignore_index=int((cfg.get("model") or {}).get("ignore_index", 255)),
                )
            loss.backward()
        peak_mb = int(torch.cuda.max_memory_allocated(device) / (1024 * 1024))
        free_mb, _total_mb = torch.cuda.mem_get_info(device)
        free_mb = int(free_mb / (1024 * 1024))
        ok = free_mb >= int(margin_mb)
        reason = f"peak_allocated={peak_mb}MB free_after={free_mb}MB margin_required={margin_mb}MB"
        del sensor, model, x, y, raw, logits, loss
        torch.cuda.empty_cache()
        return ok, reason, peak_mb
    except RuntimeError as exc:
        msg = str(exc)
        torch.cuda.empty_cache()
        if "out of memory" in msg.lower() or "cuda" in msg.lower():
            return False, msg.split("\n")[0], int(torch.cuda.max_memory_allocated(device) / (1024 * 1024))
        raise


def _candidate_batches(max_batch: int) -> List[int]:
    vals = sorted(set([1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48, 64]))
    return [v for v in vals if v <= max_batch]


def probe_job(cfg: Dict[str, Any], gpu: str, max_batch: int, safety_factor: float, steps: int, margin_mb: int) -> Dict[str, Any]:
    tested: List[Dict[str, Any]] = []
    best = 0
    for bs in _candidate_batches(max_batch):
        ok, reason, peak = _fits(cfg, bs, gpu=gpu, steps=steps, margin_mb=margin_mb)
        tested.append({"batch_size": bs, "ok": ok, "reason": reason, "peak_allocated_mb": peak})
        print(f"[probe] batch_size={bs} ok={ok} {reason}", flush=True)
        if ok:
            best = bs
        else:
            break
    if best <= 0:
        best = 1
    safe = max(1, int(math.floor(best * float(safety_factor))))
    requested_bs = max(1, int((cfg.get("data") or {}).get("batch_size", 1)))
    requested_accum = max(1, int((cfg.get("train") or {}).get("accum_steps", 1)))
    requested_effective = requested_bs * requested_accum
    accum = max(1, int(math.ceil(requested_effective / float(safe))))
    return {
        "batch_size": safe,
        "accum_steps": accum,
        "max_fit_batch_size": best,
        "requested_batch_size": requested_bs,
        "requested_accum_steps": requested_accum,
        "requested_effective_batch": requested_effective,
        "safety_factor": float(safety_factor),
        "tested": tested,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", default="deeplens/projects/raw2task/configs/industry_paper_matrix.yaml")
    parser.add_argument("--only", default="")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--max-batch", type=int, default=16)
    parser.add_argument("--safety-factor", type=float, default=0.8)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--margin-mb", type=int, default=4096)
    parser.add_argument("--out", default="runs/batch_size_plan.yaml")
    args = parser.parse_args()

    random.seed(0)
    seeds = [int(x) for x in _parse_csv_list(args.seeds)] if args.seeds else [0]
    jobs = _expand_jobs(
        os.path.abspath(os.path.expanduser(args.matrix)),
        only=_parse_csv_list(args.only),
        seeds_override=seeds[:1],
    )
    if not jobs:
        raise SystemExit("No jobs selected for batch probing.")

    out: Dict[str, Any] = {
        "matrix": os.path.abspath(os.path.expanduser(args.matrix)),
        "gpu": _gpu_info(args.gpu),
        "experiments": {},
    }
    for job in jobs:
        print(f"\n[probe] {job.experiment} seed={job.seed} on gpu={args.gpu}", flush=True)
        out["experiments"][job.experiment] = probe_job(
            job.cfg,
            gpu=args.gpu,
            max_batch=args.max_batch,
            safety_factor=args.safety_factor,
            steps=args.steps,
            margin_mb=args.margin_mb,
        )

    out_path = os.path.abspath(os.path.expanduser(args.out))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        yaml.safe_dump(out, f, sort_keys=False)
    print(f"\n[probe] wrote fixed batch plan: {out_path}")
    print(json.dumps({k: {"batch_size": v["batch_size"], "accum_steps": v["accum_steps"]} for k, v in out["experiments"].items()}, indent=2))


if __name__ == "__main__":
    main()
