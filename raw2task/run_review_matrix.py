"""Run named reviewer-facing experiments and summarize them.

Unlike the older full-factorial ``train_suite.py``, this script runs a compact,
interpretable matrix: RGB baselines, full co-design, and one-factor ablations.
Each run gets an explicit name and seed so the resulting CSV can be copied into
paper tables without guessing what a long tag means.
"""

from __future__ import annotations

import argparse
import copy
import csv
import glob
import json
import os
from typing import Any, Dict, Iterable, List

import torch
import yaml

from raw2task.train_extended import train as train_once


def _deep_set(d: Dict[str, Any], path: str, value: Any) -> None:
    cur = d
    keys = path.split(".")
    for key in keys[:-1]:
        cur = cur.setdefault(key, {})
    cur[keys[-1]] = value


def _deep_update(d: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            _deep_update(d[k], v)
        else:
            d[k] = v
    return d


def _latest_best_metrics(ckpt_dir: str) -> Dict[str, Any]:
    best = None
    last_pt = os.path.join(ckpt_dir, "last.pt")
    if os.path.isfile(last_pt):
        try:
            try:
                ckpt = torch.load(last_pt, map_location="cpu", weights_only=False)
            except TypeError:
                ckpt = torch.load(last_pt, map_location="cpu")
            best = float(ckpt.get("best", 0.0))
        except Exception:
            best = None

    rows = sorted(glob.glob(os.path.join(ckpt_dir, "metrics_epoch*.json")))
    best_metrics = {}
    for p in rows:
        try:
            with open(p, "r") as f:
                m = json.load(f)
            if not best_metrics or float(m.get("mIoU", -1)) > float(best_metrics.get("mIoU", -1)):
                best_metrics = m
        except Exception:
            continue
    if best is not None:
        best_metrics["best_val"] = best
    return best_metrics


def _avg_checkpoint_metrics(ckpt_dir: str) -> Dict[str, Any]:
    rows = sorted(glob.glob(os.path.join(ckpt_dir, "metrics_avg_*k*.json")))
    best_metrics: Dict[str, Any] = {}
    best_score = -1.0
    for p in rows:
        try:
            with open(p, "r") as f:
                metrics = json.load(f)
            score = float(metrics.get("mIoU", metrics.get("miou", -1.0)))
        except Exception:
            continue
        if score > best_score:
            best_score = score
            best_metrics = metrics
            best_metrics["metrics_path"] = p
    return best_metrics


def _best_ckpt_path(ckpt_dir: str) -> str:
    candidates = sorted(glob.glob(os.path.join(ckpt_dir, "best_ep*.pt")))
    if not candidates:
        return ""
    scored = []
    for path in candidates:
        score = -1.0
        base = os.path.basename(path)
        for marker in ("_miou", "_acc"):
            if marker in base:
                try:
                    score = float(base.split(marker, 1)[1].rsplit(".pt", 1)[0])
                except Exception:
                    score = -1.0
                break
        scored.append((score, os.path.getmtime(path), path))
    return sorted(scored)[-1][2]


def _best_avg_ckpt_path(ckpt_dir: str) -> str:
    candidates = sorted(glob.glob(os.path.join(ckpt_dir, "best_avg_*_miou*.pt")))
    if not candidates:
        return ""
    scored = []
    for path in candidates:
        score = -1.0
        base = os.path.basename(path)
        marker = "_miou"
        if marker in base:
            try:
                score = float(base.split(marker, 1)[1].rsplit(".pt", 1)[0])
            except Exception:
                score = -1.0
        scored.append((score, os.path.getmtime(path), path))
    return sorted(scored)[-1][2]


def _parse_csv_list(value: str) -> List[str]:
    if not value:
        return []
    return [x.strip() for x in value.split(",") if x.strip()]


def run_matrix(
    matrix_path: str,
    only: Iterable[str] | None = None,
    seeds_override: Iterable[int] | None = None,
    dry_run: bool = False,
    skip_existing: bool = False,
    fresh_summary: bool = False,
) -> str:
    with open(os.path.expanduser(matrix_path), "r") as f:
        matrix = yaml.safe_load(f)
    with open(os.path.expanduser(matrix["base_config"]), "r") as f:
        base_cfg = yaml.safe_load(f)

    out_root = os.path.abspath(os.path.expanduser(matrix.get("out_root", "./runs/raw2task_review_matrix")))
    os.makedirs(out_root, exist_ok=True)
    seeds = list(seeds_override) if seeds_override is not None else matrix.get("seeds", [base_cfg.get("seed", 0)])
    experiments: List[Dict[str, Any]] = matrix["experiments"]
    only_set = set(only or [])
    if only_set:
        experiments = [exp for exp in experiments if exp["name"] in only_set]

    summary_path = os.path.join(out_root, "summary.csv")
    if dry_run:
        for exp in experiments:
            for seed in seeds:
                ckpt_dir = os.path.join(out_root, f"{exp['name']}_seed{seed}")
                print(f"[dry-run] {exp['name']} seed={seed} -> {ckpt_dir}")
        print(f"Dry run only; summary would be saved to {summary_path}")
        return summary_path

    mode = "w" if fresh_summary else "a"
    write_header = fresh_summary or not os.path.exists(summary_path)
    with open(summary_path, mode, newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "experiment",
                "seed",
                "ckpt_dir",
                "best_ckpt",
                "avg_best_ckpt",
                "design_json",
                "best_val",
                "pixel_acc",
                "mIoU",
                "avg_bestk_mIoU",
                "avg_bestk_pixel_acc",
                "avg_bestk_metrics",
                "params_model_m",
                "latency_chain_ms",
            ])

        for exp in experiments:
            exp_name = exp["name"]
            overrides = exp.get("overrides", {})
            for seed in seeds:
                cfg = copy.deepcopy(base_cfg)
                cfg["seed"] = int(seed)
                _deep_update(cfg, overrides)
                ckpt_dir = os.path.join(out_root, f"{exp_name}_seed{seed}")
                _deep_set(cfg, "train.ckpt_dir", ckpt_dir)
                _deep_set(cfg, "train.resume", False)
                os.makedirs(ckpt_dir, exist_ok=True)

                print(f"\n=== Running {exp_name} seed={seed} -> {ckpt_dir} ===")
                if skip_existing and os.path.isfile(os.path.join(ckpt_dir, "last.pt")):
                    print("Existing run detected; summarizing without retraining.")
                else:
                    train_once(cfg)

                metrics = _latest_best_metrics(ckpt_dir)
                avg_metrics = _avg_checkpoint_metrics(ckpt_dir)
                eff_chain = {}
                eff_model = {}
                for name, target in [("efficiency_chain.json", eff_chain), ("efficiency_model.json", eff_model)]:
                    p = os.path.join(ckpt_dir, name)
                    if os.path.isfile(p):
                        with open(p, "r") as jf:
                            target.update(json.load(jf))

                best_ckpt = _best_ckpt_path(ckpt_dir)
                avg_best_ckpt = _best_avg_ckpt_path(ckpt_dir)
                design_json = os.path.join(ckpt_dir, "camera_design_best.json")
                if not os.path.isfile(design_json):
                    design_json = ""

                writer.writerow([
                    exp_name,
                    seed,
                    ckpt_dir,
                    best_ckpt,
                    avg_best_ckpt,
                    design_json,
                    metrics.get("best_val", ""),
                    metrics.get("pixel_acc", ""),
                    metrics.get("mIoU", ""),
                    avg_metrics.get("mIoU", ""),
                    avg_metrics.get("pixel_acc", ""),
                    avg_metrics.get("metrics_path", ""),
                    eff_model.get("params_m", ""),
                    eff_chain.get("latency_ms", ""),
                ])
                f.flush()

    print(f"Saved matrix summary to {summary_path}")
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", required=True, help="YAML file describing base_config, out_root, seeds, experiments.")
    parser.add_argument("--only", default="", help="Comma-separated experiment names to run.")
    parser.add_argument("--seeds", default="", help="Comma-separated seeds overriding the matrix.")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved runs without training.")
    parser.add_argument("--skip-existing", action="store_true", help="Reuse run dirs that already contain last.pt.")
    parser.add_argument("--fresh-summary", action="store_true", help="Overwrite summary.csv instead of appending.")
    args = parser.parse_args()

    seeds = [int(x) for x in _parse_csv_list(args.seeds)] if args.seeds else None
    run_matrix(
        matrix_path=args.matrix,
        only=_parse_csv_list(args.only),
        seeds_override=seeds,
        dry_run=args.dry_run,
        skip_existing=args.skip_existing,
        fresh_summary=args.fresh_summary,
    )


if __name__ == "__main__":
    main()
