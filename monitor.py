#!/usr/bin/env python3
"""Live experiment dashboard. Run with: watch -n 60 python3 monitor.py"""
import os, glob, csv, re, subprocess
from datetime import datetime

DATASETS = [
    {
        "label":       "KITTI-360 v2",
        "runs_dir":    os.path.expanduser("~/Raw2Task/runs/kitti360_sfb4"),
        "matrix_file": os.path.expanduser("~/Raw2Task/raw2task/configs/kitti360_sfb4_matrix.yaml"),
        "total_epochs": 40,
        "steps_per_ep": 7325,
    },
    {
        "label":       "KITTI-360 v3 (2-stage)",
        "runs_dir":    os.path.expanduser("~/Raw2Task/runs/kitti360_sfb4_v3"),
        "matrix_file": os.path.expanduser("~/Raw2Task/raw2task/configs/kitti360_sfb4_v3_matrix.yaml"),
        "total_epochs": 40,
        "steps_per_ep": 7325,
    },
    {
        "label":       "ACDC v2",
        "runs_dir":    os.path.expanduser("~/Raw2Task/runs/acdc_sfb4"),
        "matrix_file": os.path.expanduser("~/Raw2Task/raw2task/configs/acdc_sfb4_matrix.yaml"),
        "total_epochs": 60,
        "steps_per_ep": 400,
    },
    {
        "label":       "ACDC v3 (2-stage)",
        "runs_dir":    os.path.expanduser("~/Raw2Task/runs/acdc_sfb4_v3"),
        "matrix_file": os.path.expanduser("~/Raw2Task/raw2task/configs/acdc_sfb4_v3_matrix.yaml"),
        "total_epochs": 60,
        "steps_per_ep": 400,
    },
]

LOGS_DIR = os.path.expanduser("~/Raw2Task/runs/logs")
USER     = os.environ.get("USER", "re141872")

# ── helpers ───────────────────────────────────────────────────────────────────

def matrix_exps(path):
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

def is_recent(path, minutes=15):
    try:
        return (datetime.now().timestamp() - os.path.getmtime(path)) < minutes * 60
    except Exception:
        return False

def job_for_exp(exp):
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

def compute_eta(job_id, epoch, step, total_steps, total_epochs):
    path = f"{LOGS_DIR}/r2t_{job_id}.err"
    try:
        all_lines = subprocess.check_output(
            ["grep", "-E", r"Epoch [0-9]+ \| [0-9]+/[0-9]+", path],
            text=True, stderr=subprocess.DEVNULL
        ).splitlines()
        if len(all_lines) < 2:
            return ""
        ts_pat = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})'
        ep_pat = r'Epoch (\d+) \| (\d+)/(\d+)'
        t1 = re.search(ts_pat, all_lines[0])
        t2 = re.search(ts_pat, all_lines[-1])
        m1 = re.search(ep_pat, all_lines[0])
        m2 = re.search(ep_pat, all_lines[-1])
        if not (t1 and t2 and m1 and m2):
            return ""
        from datetime import datetime as _dt
        dt1 = _dt.strptime(t1.group(1), "%Y-%m-%d %H:%M:%S")
        dt2 = _dt.strptime(t2.group(1), "%Y-%m-%d %H:%M:%S")
        elapsed = (dt2 - dt1).total_seconds()
        if elapsed <= 0:
            return ""
        ep1, s1, tot = int(m1.group(1)), int(m1.group(2)), int(m1.group(3))
        ep2, s2      = int(m2.group(1)), int(m2.group(2))
        done  = (ep2 - ep1) * tot + (s2 - s1)
        if done <= 0:
            return ""
        rate  = done / elapsed
        left  = (total_epochs - ep2) * tot + (tot - s2)
        secs  = left / rate
        h, rem = divmod(int(secs), 3600)
        m_     = rem // 60
        return f"{h}h{m_:02d}m"
    except Exception:
        return ""

def parse_line(line, ext):
    info = {}
    if ext == "err":
        m = re.search(r'Epoch (\d+) \| (\d+)/(\d+)', line)
        if m:
            info["epoch"] = int(m.group(1))
            info["step"]  = int(m.group(2))
            info["total_steps"] = int(m.group(3))
            info["pct"] = 100.0 * int(m.group(2)) / max(1, int(m.group(3)))
    else:
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

running_jobs = squeue_jobs()
now          = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

print(f"\n  Raw2Task · SegFormer-B4   [{now}]")

for ds in DATASETS:
    runs_dir     = ds["runs_dir"]
    matrix_file  = ds["matrix_file"]
    total_epochs = ds["total_epochs"]
    steps_per_ep = ds["steps_per_ep"]
    exps         = matrix_exps(matrix_file)

    print(f"\n  ── {ds['label']} ({len(exps)} exps, {total_epochs} epochs) " + "─" * 60)
    print(f"  {'Experiment':<44} {'Status':<10} {'Epoch':>8}  {'BestmIoU':>9}  {'BestAcc':>8}  {'T':>2}  {'Loss':>7}  {'batchmIoU':>10}  {'ETA'}")
    print("  " + "─" * 115)

    for exp in exps:
        seed_dirs = sorted(glob.glob(f"{runs_dir}/{exp}_seed*"))
        metrics   = [m for m in (best_metrics(d) for d in seed_dirs) if m]

        job_id   = job_for_exp(exp)
        in_sq    = job_id in running_jobs if job_id else False
        log_live = job_id and is_recent(f"{LOGS_DIR}/r2t_{job_id}.err")
        is_run   = in_sq or log_live

        status = "RUNNING" if is_run else ("DONE" if metrics else "PENDING")

        live_loss = live_bmiou = live_eta = live_ep = ""
        if job_id:
            line, ext = last_live_line(job_id)
            p = parse_line(line, ext)
            if p:
                live_loss  = f"{p['loss']:.3f}"  if "loss"  in p else ""
                live_bmiou = f"{p['bmiou']:.3f}" if "bmiou" in p else ""
                live_eta   = p.get("eta", "")
                if "epoch" in p:
                    pct = f"({p['pct']:4.1f}%)" if "pct" in p else ""
                    live_ep = f"ep{p['epoch']} {pct}"
                    if not live_eta:
                        live_eta = compute_eta(
                            job_id, p["epoch"], p.get("step", 0),
                            p.get("total_steps", steps_per_ep),
                            total_epochs
                        )

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
sq_status = f"{len(running_jobs)} in squeue" if running_jobs else "squeue unavailable"
print(f"  {sq_status}")
print()
