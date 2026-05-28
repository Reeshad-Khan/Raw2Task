# deeplens/projects/raw2task/analyze_ablations.py
# Copyright (c) 2025.
# Analyze ablation_summary.csv → Pareto front + top-k table.

import argparse
import os
import sys
from typing import List, Tuple

import pandas as pd
import numpy as np


def _detect_metric(df: pd.DataFrame) -> Tuple[str, bool]:
    """Return (metric_col, larger_is_better)."""
    for col in ["mIoU", "pixel_acc", "best_val", "score", "acc"]:
        if col in df.columns:
            return col, True
    # fallback: try to infer a numeric column that looks like a metric
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric:
        raise ValueError("Could not detect any numeric metric columns.")
    return numeric[-1], True


def _available_cols(df: pd.DataFrame) -> List[str]:
    cands = [
        "eff_sensor_params_m",
        "eff_model_params_m",
        "eff_chain_flops_g",
        "eff_chain_macs_g",
        "eff_chain_latency_ms",
    ]
    return [c for c in cands if c in df.columns]


def _apply_constraints(
    df: pd.DataFrame,
    max_params: float = None,
    max_latency: float = None,
    use_model_params: bool = False,
) -> pd.DataFrame:
    out = df.copy()
    params_col = "eff_model_params_m" if (use_model_params and "eff_model_params_m" in out.columns) else "eff_sensor_params_m"
    if params_col not in out.columns and "eff_sensor_params_m" in out.columns:
        params_col = "eff_sensor_params_m"

    if max_params is not None and params_col in out.columns:
        out = out[out[params_col] <= max_params]

    if max_latency is not None and "eff_chain_latency_ms" in out.columns:
        out = out[out["eff_chain_latency_ms"] <= max_latency]

    return out, params_col


def _pareto_front(
    df: pd.DataFrame,
    metric_col: str,
    params_col: str = "eff_sensor_params_m",
    latency_col: str = "eff_chain_latency_ms",
) -> pd.DataFrame:
    """
    Pareto: maximize metric_col; minimize params_col & latency_col.
    Missing cols are ignored in dominance check.
    """
    use_params = params_col in df.columns
    use_latency = latency_col in df.columns

    vals = df[[metric_col] + ([params_col] if use_params else []) + ([latency_col] if use_latency else [])].to_numpy()
    n = len(df)
    dominated = np.zeros(n, dtype=bool)

    # indices for columns
    i_metric = 0
    i_params = 1 if use_params else None
    i_latency = (1 if (not use_params and use_latency) else 2) if use_latency else None

    for i in range(n):
        if dominated[i]:
            continue
        mi = vals[i, i_metric]
        pi = vals[i, i_params] if use_params else None
        li = vals[i, i_latency] if use_latency else None

        for j in range(n):
            if i == j:
                continue
            mj = vals[j, i_metric]
            pj = vals[j, i_params] if use_params else None
            lj = vals[j, i_latency] if use_latency else None

            better_or_equal = (mj >= mi)
            strictly_better = (mj > mi)

            if use_params:
                better_or_equal &= (pj <= pi)
                strictly_better |= (pj < pi)

            if use_latency:
                better_or_equal &= (lj <= li)
                strictly_better |= (lj < li)

            if better_or_equal and strictly_better:
                dominated[i] = True
                break

    return df[~dominated].copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to ablation_summary.csv")
    ap.add_argument("--out_dir", default=None, help="Directory to write pareto_front.csv and topk.csv")
    ap.add_argument("--topk", type=int, default=20, help="Save top-k by metric (after constraints)")
    ap.add_argument("--max-params", type=float, default=None, help="Constrain params (in millions)")
    ap.add_argument("--max-latency", type=float, default=None, help="Constrain end-to-end latency (ms)")
    ap.add_argument("--use-model-params", action="store_true", help="Use model params instead of sensor params for constraint/printing")
    ap.add_argument("--metric", default=None, help="Override metric column (default: auto-detect)")
    args = ap.parse_args()

    if not os.path.exists(args.csv):
        print(f"File not found: {args.csv}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(args.csv)
    if df.empty:
        print("CSV is empty.", file=sys.stderr)
        sys.exit(1)

    metric_col, larger_better = _detect_metric(df) if args.metric is None else (args.metric, True)
    cols = _available_cols(df)
    if not cols:
        print("Warning: efficiency columns not found; Pareto will use metric only.", file=sys.stderr)

    filtered, params_col = _apply_constraints(
        df, max_params=args.max_params, max_latency=args.max_latency, use_model_params=args.use_model_params
    )
    if filtered.empty:
        print("No rows remain after applying constraints.", file=sys.stderr)
        sys.exit(2)

    front = _pareto_front(filtered, metric_col=metric_col, params_col=params_col, latency_col="eff_chain_latency_ms")
    front = front.sort_values(by=[metric_col], ascending=not larger_better)

    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.csv))
    os.makedirs(out_dir, exist_ok=True)

    pareto_csv = os.path.join(out_dir, "pareto_front.csv")
    topk_csv = os.path.join(out_dir, "topk.csv")

    front.to_csv(pareto_csv, index=False)
    filtered.sort_values(by=[metric_col], ascending=not larger_better).head(args.topk).to_csv(topk_csv, index=False)

    # Pretty print a compact summary
    print("\n=== Detected Metric:", metric_col, "(higher is better) ===")
    show_cols = [metric_col]
    if params_col in filtered.columns:
        show_cols.append(params_col)
    if "eff_chain_latency_ms" in filtered.columns:
        show_cols.append("eff_chain_latency_ms")
    if "eff_chain_macs_g" in filtered.columns:
        show_cols.append("eff_chain_macs_g")
    if "eff_chain_flops_g" in filtered.columns:
        show_cols.append("eff_chain_flops_g")

    def _shorten(path):
        return os.path.basename(str(path))

    print("\n-- Pareto Front --")
    cols_to_show = show_cols + ["ckpt_dir"]
    print(front[cols_to_show].assign(ckpt_dir=front["ckpt_dir"].apply(_shorten)).to_string(index=False))

    print("\nSaved:")
    print(" -", pareto_csv)
    print(" -", topk_csv)
    print("\nTip: open in pandas and filter on your design knobs (e.g., specific bit_depth/width) to compare apples to apples.")


if __name__ == "__main__":
    main()