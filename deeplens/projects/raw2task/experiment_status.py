"""Live status dashboard for raw2task paper experiment matrices.

The training runners write per-job folders incrementally. This script reads the
matrix definition, the run directories, summary CSVs, metrics logs, and active
process command lines to answer:

- how many jobs are done, running, waiting, or partial;
- best/latest mIoU and pixel accuracy for finished jobs;
- current epoch, latest train/eval metrics, and ETA for active jobs.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from deeplens.projects.raw2task.run_paper_experiments import (
    Job,
    _expand_jobs,
    _is_complete,
    _last_logged_epoch,
    _matrix_out_root,
    _parse_csv_list,
    _target_epochs,
)


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def _as_float(value: Any) -> Optional[float]:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except Exception:
        return None


def _fmt_float(value: Any, digits: int = 4) -> str:
    val = _as_float(value)
    return "n/a" if val is None else f"{val:.{digits}f}"


def _fmt_time(seconds: Optional[float]) -> str:
    if seconds is None or seconds < 0:
        return "n/a"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:d}h{m:02d}m"
    if m:
        return f"{m:d}m{s:02d}s"
    return f"{s:d}s"


def _tail_text(path: Path, max_bytes: int = 160_000) -> str:
    if not path.is_file():
        return ""
    try:
        size = path.stat().st_size
        with path.open("rb") as f:
            if size > max_bytes:
                f.seek(size - max_bytes)
            return f.read().decode("utf-8", errors="ignore")
    except Exception:
        return ""


def _active_config_paths() -> Dict[str, Dict[str, str]]:
    """Return active train_extended config paths keyed by absolute path."""
    active: Dict[str, Dict[str, str]] = {}
    try:
        out = subprocess.check_output(["ps", "-eo", "pid=,etimes=,args="], text=True)
    except Exception:
        return active
    for line in out.splitlines():
        if "deeplens.projects.raw2task.train_extended" not in line or "--config" not in line:
            continue
        parts = line.strip().split(None, 2)
        if len(parts) < 3:
            continue
        pid, etimes, args = parts
        match = re.search(r"--config\s+(\S+)", args)
        if not match:
            continue
        cfg_path = os.path.abspath(os.path.expanduser(match.group(1)))
        active[cfg_path] = {"pid": pid, "elapsed": etimes, "args": args}
    return active


def _summary_by_job(out_root: Path) -> Dict[tuple[str, str], Dict[str, str]]:
    rows = _read_csv(out_root / "summary.csv")
    return {(r.get("experiment", ""), str(r.get("seed", ""))): r for r in rows}


def _best_latest_metrics(run_dir: Path) -> Dict[str, Any]:
    rows = _read_csv(run_dir / "metrics_log.csv")
    if not rows:
        return {}
    latest = rows[-1]
    best = max(rows, key=lambda r: _as_float(r.get("mIoU")) if _as_float(r.get("mIoU")) is not None else -1e9)
    return {
        "latest_epoch": latest.get("epoch", ""),
        "latest_miou": latest.get("mIoU", ""),
        "latest_acc": latest.get("pixel_acc", ""),
        "best_epoch": best.get("epoch", ""),
        "best_miou": best.get("mIoU", ""),
        "best_acc": best.get("pixel_acc", ""),
    }


def _latest_train_state(log_path: Path) -> Dict[str, str]:
    text = _tail_text(log_path)
    state: Dict[str, str] = {}
    train_re = re.compile(
        r"TRAIN epoch (?P<epoch>\d+)/(?P<epochs>\d+).*?"
        r"step (?P<step>\d+)/(?P<steps>\d+).*?"
        r"ETA_epoch (?P<eta_epoch>\S+) ETA_total (?P<eta_total>\S+).*?"
        r"loss (?P<loss>[0-9.]+).*?acc (?P<acc>[0-9.]+).*?"
        r"batch_mIoU (?P<miou>[0-9.]+)",
    )
    eval_re = re.compile(
        r"EVAL(?:\[[^\]]+\])? progress .*?"
        r"batch (?P<step>\d+)/(?P<steps>\d+|\?).*?"
        r"ETA (?P<eta>\S+).*?"
        r"pixel_acc (?P<acc>[0-9.]+).*?"
        r"running_mIoU (?P<miou>[0-9.]+)",
    )
    final_eval_re = re.compile(
        r"EVAL epoch (?P<epoch>\d+)/(?P<epochs>\d+) "
        r"pixel_acc (?P<acc>[0-9.]+) mIoU (?P<miou>[0-9.]+)"
    )
    for match in train_re.finditer(text):
        state.update({f"train_{k}": v for k, v in match.groupdict().items()})
        state["phase"] = "train"
    for match in eval_re.finditer(text):
        state.update({f"eval_{k}": v for k, v in match.groupdict().items()})
        state["phase"] = "eval"
    for match in final_eval_re.finditer(text):
        state.update({f"last_eval_{k}": v for k, v in match.groupdict().items()})
    if "phase" not in state and text:
        state["phase"] = "starting"
    return state


def _runtime_from_log(run_dir: Path, running_elapsed: Optional[str]) -> str:
    if running_elapsed:
        try:
            return _fmt_time(float(running_elapsed))
        except Exception:
            pass
    complete_json = run_dir / "train_complete.json"
    if complete_json.is_file():
        try:
            payload = json.loads(complete_json.read_text())
            if "wall_time_seconds" in payload:
                return _fmt_time(float(payload["wall_time_seconds"]))
        except Exception:
            pass
    log_path = run_dir / "train.log"
    if not log_path.is_file():
        return "n/a"
    try:
        stat = log_path.stat()
        return _fmt_time(max(0.0, stat.st_mtime - stat.st_ctime))
    except Exception:
        return "n/a"


def _status_for_job(job: Job, active: Dict[str, Dict[str, str]]) -> Dict[str, Any]:
    run_dir = Path(job.ckpt_dir)
    cfg_path = os.path.abspath(os.path.expanduser(job.config_path))
    proc = active.get(cfg_path)
    target_epochs = _target_epochs(job)
    logged_epoch = _last_logged_epoch(job)
    complete = _is_complete(job)
    metrics = _best_latest_metrics(run_dir)
    log_state = _latest_train_state(run_dir / "train.log")

    if proc:
        status = "running"
    elif complete:
        status = "done"
    elif logged_epoch > 0 or (run_dir / "last.pt").is_file():
        status = "partial"
    else:
        status = "waiting"

    eta = "n/a"
    if proc:
        if log_state.get("phase") == "train":
            eta = log_state.get("train_eta_total", "n/a")
        elif log_state.get("phase") == "eval":
            eta = log_state.get("eval_eta", "n/a")

    return {
        "experiment": job.experiment,
        "seed": str(job.seed),
        "status": status,
        "pid": proc.get("pid", "") if proc else "",
        "epoch": logged_epoch,
        "target_epochs": target_epochs,
        "runtime": _runtime_from_log(run_dir, proc.get("elapsed") if proc else None),
        "eta": eta,
        "phase": log_state.get("phase", ""),
        "latest_miou": metrics.get("latest_miou", ""),
        "latest_acc": metrics.get("latest_acc", ""),
        "best_miou": metrics.get("best_miou", ""),
        "best_acc": metrics.get("best_acc", ""),
        "best_epoch": metrics.get("best_epoch", ""),
        "train_loss": log_state.get("train_loss", ""),
        "train_acc": log_state.get("train_acc", ""),
        "train_batch_miou": log_state.get("train_miou", ""),
        "eval_acc": log_state.get("eval_acc", log_state.get("last_eval_acc", "")),
        "eval_miou": log_state.get("eval_miou", log_state.get("last_eval_miou", "")),
        "ckpt_dir": str(run_dir),
    }


def _print_table(title: str, rows: List[Dict[str, Any]], fields: List[str], limit: int = 0) -> None:
    if not rows:
        return
    shown = rows[:limit] if limit and limit > 0 else rows
    print(f"\n{title}")
    widths = {
        field: max(len(field), *(len(str(row.get(field, ""))) for row in shown))
        for field in fields
    }
    print("  ".join(field.ljust(widths[field]) for field in fields))
    print("  ".join("-" * widths[field] for field in fields))
    for row in shown:
        print("  ".join(str(row.get(field, "")).ljust(widths[field]) for field in fields))
    if limit and len(rows) > limit:
        print(f"... {len(rows) - limit} more")


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", action="append", default=[], help="Matrix YAML. Can be repeated.")
    parser.add_argument("--only", default="", help="Comma-separated experiment filter.")
    parser.add_argument("--seeds", default="", help="Comma-separated seed override.")
    parser.add_argument("--order-policy", default="codesign_first", choices=["codesign_first", "matrix"])
    parser.add_argument("--watch", type=int, default=0, help="Refresh every N seconds.")
    parser.add_argument(
        "--no-clear",
        action="store_true",
        help="In watch mode, append each refresh instead of updating the same terminal screen.",
    )
    parser.add_argument("--exit-when-complete", action="store_true", help="In watch mode, exit once all jobs are done.")
    parser.add_argument("--limit-waiting", type=int, default=20)
    parser.add_argument("--out-csv", default="", help="Optional status CSV output path.")
    args = parser.parse_args()

    matrices = args.matrix or [
        "deeplens/projects/raw2task/configs/dataset_compare_kitti360_fast.yaml",
        "deeplens/projects/raw2task/configs/dataset_compare_cityscapes_fast.yaml",
    ]
    only = _parse_csv_list(args.only)
    seeds = [int(x) for x in _parse_csv_list(args.seeds)] if args.seeds else None

    def render() -> List[Dict[str, Any]]:
        all_rows: List[Dict[str, Any]] = []
        active = _active_config_paths()
        for matrix in matrices:
            matrix_path = os.path.abspath(os.path.expanduser(matrix))
            jobs = _expand_jobs(matrix_path, only=only, seeds_override=seeds, order_policy=args.order_policy)
            out_root = Path(_matrix_out_root(matrix_path))
            summary = _summary_by_job(out_root)
            for job in jobs:
                row = _status_for_job(job, active)
                row["matrix"] = os.path.basename(matrix_path)
                srow = summary.get((job.experiment, str(job.seed)))
                if srow:
                    row["summary_miou"] = srow.get("mIoU", "")
                    row["summary_acc"] = srow.get("pixel_acc", "")
                else:
                    row["summary_miou"] = ""
                    row["summary_acc"] = ""
                all_rows.append(row)
        return all_rows

    while True:
        rows = render()
        counts: Dict[str, int] = {}
        for row in rows:
            counts[row["status"]] = counts.get(row["status"], 0) + 1
        if args.watch > 0 and not args.no_clear:
            print("\033[2J\033[3J\033[H", end="", flush=True)
        print(f"Raw2Task Experiment Status | {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(
            "Total {total} | done {done} | running {running} | partial {partial} | waiting {waiting}".format(
                total=len(rows),
                done=counts.get("done", 0),
                running=counts.get("running", 0),
                partial=counts.get("partial", 0),
                waiting=counts.get("waiting", 0),
            )
        )

        running = [r for r in rows if r["status"] == "running"]
        done = [r for r in rows if r["status"] == "done"]
        partial = [r for r in rows if r["status"] == "partial"]
        waiting = [r for r in rows if r["status"] == "waiting"]

        _print_table(
            "Running",
            running,
            ["experiment", "seed", "pid", "phase", "epoch", "target_epochs", "best_miou", "latest_miou", "eval_miou", "train_batch_miou", "eta", "runtime"],
        )
        _print_table(
            "Done",
            done,
            ["experiment", "seed", "best_miou", "best_acc", "best_epoch", "latest_miou", "latest_acc", "runtime"],
        )
        _print_table(
            "Partial / Interrupted",
            partial,
            ["experiment", "seed", "epoch", "target_epochs", "best_miou", "latest_miou", "runtime"],
        )
        _print_table(
            "Waiting",
            waiting,
            ["experiment", "seed", "epoch", "target_epochs"],
            limit=args.limit_waiting,
        )

        if args.out_csv:
            _write_csv(Path(args.out_csv), rows)
            print(f"\nWrote CSV: {args.out_csv}")

        if args.exit_when_complete and rows and all(row["status"] == "done" for row in rows):
            break
        if args.watch <= 0:
            break
        sys.stdout.flush()
        time.sleep(max(1, int(args.watch)))


if __name__ == "__main__":
    main()
