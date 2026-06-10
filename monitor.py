#!/usr/bin/env python3
"""Live experiment dashboard. Run with: watch -n 30 python3 monitor.py"""
import os, glob, csv, re, subprocess, statistics
from datetime import datetime

RUNS_DIR = os.path.expanduser("~/Raw2Task/runs/kitti360_sfb2")
LOGS_DIR = os.path.expanduser("~/Raw2Task/runs/logs")
USER     = os.environ.get("USER", "re141872")

def squeue_jobs():
    try:
        out = subprocess.check_output(
            ["squeue", "-u", USER, "--format=%i %T %M", "--noheader"], text=True
        )
        jobs = {}
        for line in out.strip().splitlines():
            parts = line.split()
            if len(parts) >= 3:
                jobs[parts[0]] = {"state": parts[1], "time": parts[2]}
        return jobs
    except Exception:
        return {}

def job_for_exp(exp, log_dir):
    """Scan log files to find the job ID running a given experiment."""
    for f in sorted(glob.glob(f"{log_dir}/r2t_*.out"), reverse=True):
        try:
            with open(f) as fh:
                for line in fh:
                    if f"Experiment: {exp}" in line:
                        jid = re.search(r"r2t_(\d+)\.out", f)
                        return jid.group(1) if jid else None
        except Exception:
            pass
    return None

def last_train_line(job_id, log_dir):
    path = f"{log_dir}/r2t_{job_id}.out"
    if not os.path.exists(path):
        return ""
    try:
        out = subprocess.check_output(["tail", "-80", path], text=True)
        for line in reversed(out.splitlines()):
            if "TRAIN epoch" in line:
                return line
    except Exception:
        pass
    return ""

def parse_train(line):
    info = {}
    for pat, key, cast in [
        (r"epoch (\d+)/(\d+)", "epoch", None),
        (r"(\d+\.\d+)%",       "pct",   float),
        (r"ETA_total (\S+)",   "eta",   str),
        (r"loss (\d+\.\d+)",   "loss",  float),
        (r" acc (\d+\.\d+)",   "acc",   float),
        (r"batch_mIoU (\d+\.\d+)", "bmiou", float),
    ]:
        m = re.search(pat, line)
        if m:
            if key == "epoch":
                info["epoch"] = int(m.group(1))
                info["total_epochs"] = int(m.group(2))
            else:
                info[key] = cast(m.group(1))
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
        best_row = max(rows, key=lambda r: float(r["mIoU"]))
        mious = [float(r["mIoU"]) for r in rows]
        trend = "↑" if len(mious) >= 2 and mious[-1] > mious[-2] else \
                "↓" if len(mious) >= 2 and mious[-1] < mious[-2] else "→"
        return {
            "best_miou": float(best_row["mIoU"]),
            "best_acc":  float(best_row["pixel_acc"]),
            "epoch":     int(rows[-1]["epoch"]),
            "n":         len(rows),
            "trend":     trend,
        }
    except Exception:
        return None

# ── gather data ──────────────────────────────────────────────────────────────
running_jobs = squeue_jobs()

# collect all experiment names from completed seed dirs + running logs
exps = set()
for d in glob.glob(f"{RUNS_DIR}/*_seed*"):
    exps.add(os.path.basename(d).rsplit("_seed", 1)[0])

# also pick up experiments that only exist in logs (no completed epochs yet)
for f in glob.glob(f"{LOGS_DIR}/r2t_*.out"):
    try:
        with open(f) as fh:
            for line in fh:
                m = re.search(r"Experiment: (\S+)", line)
                if m:
                    exps.add(m.group(1))
                    break
    except Exception:
        pass

# ── print dashboard ──────────────────────────────────────────────────────────
now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print(f"\n  Raw2Task · SegFormer-B2 Experiments   [{now}]\n")
hdr = f"  {'Experiment':<44} {'Status':<10} {'Epoch':>6}  {'BestmIoU':>9}  {'BestAcc':>8}  {'Trend':>5}  {'Loss':>7}  {'ETA':>9}"
print(hdr)
print("  " + "─" * 103)

for exp in sorted(exps):
    seed_dirs = sorted(glob.glob(f"{RUNS_DIR}/{exp}_seed*"))
    metrics   = [best_metrics(d) for d in seed_dirs]
    metrics   = [m for m in metrics if m]

    job_id   = job_for_exp(exp, LOGS_DIR)
    is_run   = job_id in running_jobs if job_id else False
    job_info = running_jobs.get(job_id, {})

    if is_run:
        status = "RUNNING"
    elif metrics:
        status = "DONE"
    else:
        status = "PENDING"

    # live step info
    live_loss = live_eta = live_epoch = ""
    if is_run:
        tline = last_train_line(job_id, LOGS_DIR)
        p = parse_train(tline)
        if p:
            live_loss  = f"{p['loss']:.3f}" if "loss" in p else ""
            live_eta   = p.get("eta", "")
            live_epoch = f"{p['epoch']}/{p.get('total_epochs','?')}" if "epoch" in p else ""

    if metrics:
        bm   = max(m["best_miou"] for m in metrics)
        ba   = max(m["best_acc"]  for m in metrics)
        trnd = metrics[-1]["trend"]
        ep   = live_epoch or str(metrics[-1]["epoch"])
        seeds_done = f"(s{len(metrics)})"
        print(f"  {exp+' '+seeds_done:<44} {status:<10} {ep:>6}  {bm:>9.4f}  {ba:>8.4f}  {trnd:>5}  {live_loss:>7}  {live_eta:>9}")
    else:
        ep = live_epoch or "—"
        print(f"  {exp:<44} {status:<10} {ep:>6}  {'—':>9}  {'—':>8}  {'?':>5}  {live_loss:>7}  {live_eta:>9}")

print()
print(f"  Jobs in queue: {len(running_jobs)}   Runs dir: {RUNS_DIR}")
print()
