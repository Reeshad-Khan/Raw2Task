#!/usr/bin/env python3
"""Live experiment dashboard. Run with: watch -n 60 python3 monitor.py"""
import os, glob, csv, re, subprocess
from datetime import datetime

RUNS_DIR    = os.path.expanduser("~/Raw2Task/runs/kitti360_m2f")
LOGS_DIR    = os.path.expanduser("~/Raw2Task/runs/logs")
MATRIX_FILE = os.path.expanduser("~/Raw2Task/raw2task/configs/kitti360_m2f_matrix.yaml")
USER        = os.environ.get("USER", "re141872")

# ── helpers ───────────────────────────────────────────────────────────────────

def matrix_exps(path):
    """Read experiment names from matrix yaml (no pyyaml needed)."""
    exps = []
    try:
        with open(path) as f:
            for line in f:
                m = re.match(r'\s*-\s*name:\s*(\S+)', line)
                if m:
                    exps.append(m.group(1))
    except Exception:
        pass
    return exps

def squeue_jobs():
    try:
        out = subprocess.check_output(
            ["squeue", "-u", USER, "--noheader", "-o", "%i %T %M"],
            text=True, stderr=subprocess.DEVNULL
        )
        return {parts[0]: parts[1] if len(parts) > 1 else "?"
                for line in out.strip().splitlines()
                for parts in [line.split()] if parts}
    except Exception:
        return {}

def is_recent(path, minutes=20):
    try:
        return (datetime.now().timestamp() - os.path.getmtime(path)) < minutes * 60
    except Exception:
        return False

def job_for_exp(exp):
    """Find the most recent job ID whose .out file mentions this experiment."""
    for f in sorted(glob.glob(f"{LOGS_DIR}/r2t_*.out"), reverse=True):
        try:
            with open(f) as fh:
                for line in fh:
                    if f"Experiment: {exp}" in line:
                        m = re.search(r"r2t_(\d+)\.out", f)
                        return m.group(1) if m else None
        except Exception:
            pass
    return None

def last_live_line(job_id):
    """Return the most recent step-log line from .err, or progress from .out."""
    for ext, pattern in [("err", r"Epoch \d+ \| \d+/\d+"), ("out", r"TRAIN epoch")]:
        path = f"{LOGS_DIR}/r2t_{job_id}.{ext}"
        if not os.path.exists(path):
            continue
        try:
            lines = subprocess.check_output(["tail", "-120", path], text=True).splitlines()
            for line in reversed(lines):
                if re.search(pattern, line):
                    return line, ext
        except Exception:
            pass
    return "", ""

def parse_line(line, ext):
    """Parse a training progress line from either .err or .out format."""
    info = {}
    if ext == "err":
        # format: "... | Epoch 3 | 1200/3663 | loss 0.12 | acc 0.94 | batch_mIoU 0.55 | lr 6e-05"
        m = re.search(r'Epoch (\d+) \| (\d+)/(\d+)', line)
        if m:
            info["epoch"] = int(m.group(1))
            info["step"]  = int(m.group(2))
            info["total_steps"] = int(m.group(3))
            info["pct"] = 100.0 * int(m.group(2)) / max(1, int(m.group(3)))
    else:
        # format: "TRAIN epoch 3/40 [...] ETA_total 5h30m loss 0.12 ..."
        m = re.search(r'epoch (\d+)/(\d+)', line)
        if m:
            info["epoch"] = int(m.group(1))
            info["total_epochs"] = int(m.group(2))
        m = re.search(r'ETA_total (\S+)', line)
        if m:
            info["eta"] = m.group(1)
        m = re.search(r'(\d+\.\d+)%', line)
        if m:
            info["pct"] = float(m.group(1))

    m = re.search(r'loss (\d+\.\d+)', line)
    if m:
        info["loss"] = float(m.group(1))
    m = re.search(r'\bacc (\d+\.\d+)', line)
    if m:
        info["acc"] = float(m.group(1))
    m = re.search(r'batch_mIoU (\d+\.\d+)', line)
    if m:
        info["bmiou"] = float(m.group(1))
    return info

def best_metrics(seed_dir):
    path = os.path.join(seed_dir, "metrics_log.csv")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            rows = [r for r in csv.DictReader(f) if r.get("mIoU")]
        if not rows:
            return None
        best = max(rows, key=lambda r: float(r["mIoU"]))
        mious = [float(r["mIoU"]) for r in rows]
        trend = ("↑" if mious[-1] > mious[-2] else "↓" if mious[-1] < mious[-2] else "→") \
                if len(mious) >= 2 else "→"
        return {
            "best_miou": float(best["mIoU"]),
            "best_acc":  float(best["pixel_acc"]),
            "epoch":     int(rows[-1]["epoch"]),
            "trend":     trend,
        }
    except Exception:
        return None

# ── main ──────────────────────────────────────────────────────────────────────
exps         = matrix_exps(MATRIX_FILE)
running_jobs = squeue_jobs()
now          = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

print(f"\n  Raw2Task · SegFormer-B2 ({len(exps)} exps)   [{now}]\n")
print(f"  {'Experiment':<44} {'Status':<10} {'Epoch':>8}  {'BestmIoU':>9}  {'BestAcc':>8}  {'T':>2}  {'Loss':>7}  {'batchmIoU':>10}  {'ETA'}")
print("  " + "─" * 115)

for exp in exps:
    seed_dirs = sorted(glob.glob(f"{RUNS_DIR}/{exp}_seed*"))
    metrics   = [m for m in (best_metrics(d) for d in seed_dirs) if m]

    job_id   = job_for_exp(exp)
    in_sq    = job_id in running_jobs if job_id else False
    log_live = job_id and is_recent(f"{LOGS_DIR}/r2t_{job_id}.err")
    is_run   = in_sq or log_live

    status = "RUNNING" if is_run else ("DONE" if metrics else "PENDING")

    # live info (available regardless of squeue)
    live_loss = live_bmiou = live_eta = live_ep = ""
    if job_id:
        line, ext = last_live_line(job_id)
        p = parse_line(line, ext)
        if p:
            live_loss  = f"{p['loss']:.3f}"  if "loss"  in p else ""
            live_bmiou = f"{p['bmiou']:.3f}" if "bmiou" in p else ""
            live_eta   = p.get("eta", "")
            if "epoch" in p:
                pct = f" {p['pct']:4.1f}%" if "pct" in p else ""
                tot = p.get("total_epochs", p.get("total_steps", "?"))
                live_ep = f"{p['epoch']}/{tot}{pct}"

    if metrics:
        bm   = max(m["best_miou"] for m in metrics)
        ba   = max(m["best_acc"]  for m in metrics)
        trnd = metrics[-1]["trend"]
        ep   = live_ep or str(metrics[-1]["epoch"])
        label = f"{exp} (s{len(metrics)})"
        print(f"  {label:<44} {status:<10} {ep:>8}  {bm:>9.4f}  {ba:>8.4f}  {trnd:>2}  {live_loss:>7}  {live_bmiou:>10}  {live_eta}")
    else:
        ep = live_ep or "—"
        print(f"  {exp:<44} {status:<10} {ep:>8}  {'—':>9}  {'—':>8}  {'?':>2}  {live_loss:>7}  {live_bmiou:>10}  {live_eta}")

print()
sq_status = f"{len(running_jobs)} in squeue" if running_jobs else "squeue unavailable (using mtime fallback)"
print(f"  {sq_status}   Runs: {RUNS_DIR}")
print()
