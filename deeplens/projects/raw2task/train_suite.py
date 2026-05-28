# deeplens/projects/raw2task/train_suite.py
# Copyright (c) 2025.
"""
Run a grid of ablations on top of a base config and collect scores + efficiency.
Usage:
  python -m deeplens.projects.raw2task.train_suite \
    --config deeplens/projects/raw2task/configs/cifar10_cls_baseline.yaml \
    --out_root ./ablations/out \
    --task seg
"""

from __future__ import annotations
import argparse, copy, csv, itertools, json, os, pathlib
import yaml
from typing import Any, Dict, List

import torch

# Import your training entrypoint
from deeplens.projects.raw2task.train_extended import train as train_once

def _deep_set(d: Dict[str, Any], path: str, value: Any):
    """
    Write d["a"]["b"]["c"] = value for path="a.b.c" creating missing dicts.
    """
    keys = path.split(".")
    cur = d
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value

def _read_json(path: str) -> dict | None:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=str)
    ap.add_argument("--out_root", required=True, type=str)
    ap.add_argument("--task", choices=["cls","seg"], default=None)
    # Quick presets for our ablations; you can add more paths/values here.
    # You can also pass a YAML file of your own grids later if needed.
    args = ap.parse_args()

    with open(os.path.expanduser(args.config), "r") as f:
        base_cfg = yaml.safe_load(f)

    if args.task is not None:
        base_cfg["task"] = args.task

    # ---------- Define your ablation grid here ----------
    # Format: list of (dot_path, list_of_values)
    grid: List[tuple[str, List[Any]]] = [
        # Loss choices
        ("train.use_lovasz", [True, False]),
        # OHEM %
        ("train.ohem_pct", [0.10, 0.25, 0.40]),
        # Smoothness lambda
        ("train.smooth_lambda", [0.00, 0.05, 0.10]),
        # Network width/base
        ("model.width", [16, 32, 48]),
        # Sensor bit depth
        ("sensor.bit_depth", [4, 6, 8]),
        # Debug bypasses (= ablations of optics/CFA/noise)
        ("debug.bypass_optics", [False, True]),
        ("debug.bypass_cfa", [False, True]),
        ("debug.bypass_noise", [False, True]),
    ]

    # Make out root
    out_root = os.path.abspath(args.out_root)
    os.makedirs(out_root, exist_ok=True)

    # Build all combinations
    keys = [k for k,_ in grid]
    vals = [v for _,v in grid]
    combos = list(itertools.product(*vals))

    # CSV summary
    summary_csv = os.path.join(out_root, "ablation_summary.csv")
    write_header = not os.path.exists(summary_csv)
    with open(summary_csv, "a", newline="") as fcsv:
        writer = csv.writer(fcsv)
        if write_header:
            writer.writerow(
                keys +
                ["ckpt_dir","best_val","metric_name","eff_sensor_params_m","eff_model_params_m",
                 "eff_chain_flops_g","eff_chain_macs_g","eff_chain_latency_ms"]
            )

        for idx, combo in enumerate(combos, 1):
            cfg = copy.deepcopy(base_cfg)

            # Apply overrides
            tag_bits = []
            for k, v in zip(keys, combo):
                _deep_set(cfg, k, v)
                tag_bits.append(f"{k.replace('.','_')}={v}")

            # Unique out dir for this run
            tag = "__".join(tag_bits)
            ckpt_dir = os.path.join(out_root, f"run_{idx:03d}__{tag}")
            _deep_set(cfg, "train.ckpt_dir", ckpt_dir)

            # Train
            print(f"\n=== [{idx}/{len(combos)}] {tag} ===")
            train_once(cfg)  # Calls your existing training loop

            # Read val metric and efficiency
            # For seg, metric is mIoU; for cls, it's acc. We saved per-epoch files.
            # We’ll try reading 'best' from last.pt and fall back to latest metrics CSV.
            best_val = None
            metric_name = "score"
            last_pt = os.path.join(ckpt_dir, "last.pt")
            if os.path.exists(last_pt):
                try:
                    ckpt = torch.load(last_pt, map_location="cpu")
                    best_val = float(ckpt.get("best", None))
                except Exception:
                    pass

            if best_val is None:
                # Fallback: try the metrics CSV used in train_extended
                csv_path = os.path.join(ckpt_dir, "metrics_log.csv")
                if os.path.exists(csv_path):
                    try:
                        import pandas as pd
                        df = pd.read_csv(csv_path)
                        if "mIoU" in df.columns:
                            best_val = float(df["mIoU"].max())
                            metric_name = "mIoU"
                        elif "pixel_acc" in df.columns:
                            best_val = float(df["pixel_acc"].max())
                            metric_name = "pixel_acc"
                    except Exception:
                        pass

            # Efficiency JSONs
            eff_s = _read_json(os.path.join(ckpt_dir, "efficiency_sensor.json")) or {}
            eff_m = _read_json(os.path.join(ckpt_dir, "efficiency_model.json")) or {}
            eff_c = _read_json(os.path.join(ckpt_dir, "efficiency_chain.json")) or {}

            writer.writerow(
                list(combo) +
                [ckpt_dir, best_val, metric_name,
                 eff_s.get("params_m",""), eff_m.get("params_m",""),
                 eff_c.get("flops_g",""),  eff_c.get("macs_g",""),
                 eff_c.get("latency_ms","")]
            )

    print(f"\nAblation complete. Summary at: {summary_csv}")
    print("Tip: open in pandas for quick ranking by best metric under a latency/params constraint.")
if __name__ == "__main__":
    main()
