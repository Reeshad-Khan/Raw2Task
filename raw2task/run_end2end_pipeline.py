"""End-to-end reviewer-facing experiment pipeline.

This script runs the named experiment matrix, aggregates seed statistics,
executes robustness sweeps for selected methods, and writes paper-ready CSV and
Markdown tables under ``paper_tables``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, Iterable, List, Tuple

import yaml

from raw2task.curate_paper_figures import curate as curate_story_figures
from raw2task.animate_camera_learning import make_learning_gifs
from raw2task.run_review_matrix import run_matrix


NUMERIC_COLS = ("best_val", "pixel_acc", "mIoU", "params_model_m", "latency_chain_ms")


def _parse_csv_list(value: str) -> List[str]:
    if not value:
        return []
    return [x.strip() for x in value.split(",") if x.strip()]


def _as_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def _fmt_float(value: float | None, digits: int = 4) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def _mean_std(values: Iterable[float | None]) -> Tuple[float | None, float | None, int]:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return None, None, 0
    return mean(vals), (stdev(vals) if len(vals) > 1 else 0.0), len(vals)


def _read_csv(path: str) -> List[Dict[str, str]]:
    if not os.path.isfile(path):
        return []
    with open(path, "r", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: str, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _write_markdown_table(path: str, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("| " + " | ".join(fieldnames) + " |\n")
        f.write("| " + " | ".join(["---"] * len(fieldnames)) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(str(row.get(k, "")) for k in fieldnames) + " |\n")


def _dedupe_summary(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    latest: Dict[Tuple[str, str], Dict[str, str]] = {}
    for row in rows:
        latest[(row.get("experiment", ""), row.get("seed", ""))] = row
    return list(latest.values())


def aggregate_main_results(summary_csv: str, tables_dir: str) -> List[Dict[str, str]]:
    rows = _dedupe_summary(_read_csv(summary_csv))
    groups: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row.get("experiment", "")].append(row)

    out_rows: List[Dict[str, str]] = []
    for experiment, exp_rows in sorted(groups.items()):
        agg: Dict[str, str] = {"experiment": experiment}
        metric_n = 0
        for col in NUMERIC_COLS:
            m, s, n = _mean_std(_as_float(r.get(col)) for r in exp_rows)
            agg[f"{col}_mean"] = _fmt_float(m)
            agg[f"{col}_std"] = _fmt_float(s)
            if col == "mIoU":
                metric_n = n
        agg["n"] = str(metric_n)
        agg["seeds"] = ",".join(str(r.get("seed", "")) for r in sorted(exp_rows, key=lambda r: str(r.get("seed", ""))))
        out_rows.append(agg)

    fields = [
        "experiment",
        "n",
        "seeds",
        "mIoU_mean",
        "mIoU_std",
        "pixel_acc_mean",
        "pixel_acc_std",
        "params_model_m_mean",
        "latency_chain_ms_mean",
        "best_val_mean",
        "best_val_std",
    ]
    _write_csv(os.path.join(tables_dir, "main_results.csv"), out_rows, fields)
    _write_markdown_table(os.path.join(tables_dir, "main_results.md"), out_rows, fields)
    return rows


def run_robustness(
    summary_rows: List[Dict[str, str]],
    experiments: Iterable[str],
    sweep: str,
    max_batches: int,
    force: bool,
) -> None:
    exp_set = set(experiments)
    if not exp_set:
        return

    for row in summary_rows:
        if row.get("experiment") not in exp_set:
            continue
        ckpt = row.get("best_ckpt", "")
        ckpt_dir = row.get("ckpt_dir", "")
        if not ckpt or not os.path.isfile(ckpt):
            print(f"[Robustness] Missing best checkpoint for {row.get('experiment')} seed={row.get('seed')}; skipping.")
            continue
        out_dir = os.path.join(ckpt_dir, "robustness")
        out_csv = os.path.join(out_dir, "robustness.csv")
        if os.path.isfile(out_csv) and not force:
            print(f"[Robustness] Reusing {out_csv}")
            continue

        cmd = [
            sys.executable,
            "-m",
            "raw2task.eval_robustness",
            "--ckpt",
            ckpt,
            "--out-dir",
            out_dir,
            "--max-batches",
            str(max_batches),
        ]
        if sweep:
            cmd.extend(["--sweep", sweep])
        print("[Robustness] " + " ".join(cmd))
        subprocess.run(cmd, check=True)


def collect_robustness(summary_rows: List[Dict[str, str]], tables_dir: str) -> None:
    all_rows: List[Dict[str, Any]] = []
    for row in summary_rows:
        path = os.path.join(row.get("ckpt_dir", ""), "robustness", "robustness.csv")
        for r in _read_csv(path):
            r = dict(r)
            r["experiment"] = row.get("experiment", "")
            r["seed"] = row.get("seed", "")
            all_rows.append(r)
    if not all_rows:
        return

    fields = ["experiment", "seed", "name", "kind", "level", "pixel_acc", "mIoU"]
    _write_csv(os.path.join(tables_dir, "robustness_all.csv"), all_rows, fields)

    groups: Dict[Tuple[str, str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        groups[(row["experiment"], row["name"], row["kind"], str(row["level"]))].append(row)

    summary: List[Dict[str, str]] = []
    for (experiment, name, kind, level), rows in sorted(groups.items()):
        miou_m, miou_s, n = _mean_std(_as_float(r.get("mIoU")) for r in rows)
        acc_m, acc_s, _ = _mean_std(_as_float(r.get("pixel_acc")) for r in rows)
        summary.append(
            {
                "experiment": experiment,
                "name": name,
                "kind": kind,
                "level": level,
                "n": str(n),
                "mIoU_mean": _fmt_float(miou_m),
                "mIoU_std": _fmt_float(miou_s),
                "pixel_acc_mean": _fmt_float(acc_m),
                "pixel_acc_std": _fmt_float(acc_s),
            }
        )
    fields_summary = ["experiment", "name", "kind", "level", "n", "mIoU_mean", "mIoU_std", "pixel_acc_mean", "pixel_acc_std"]
    _write_csv(os.path.join(tables_dir, "robustness_summary.csv"), summary, fields_summary)
    _write_markdown_table(os.path.join(tables_dir, "robustness_summary.md"), summary, fields_summary)


def collect_design_inventory(summary_rows: List[Dict[str, str]], tables_dir: str) -> None:
    def _load_json(path: str) -> Dict[str, Any]:
        if not path or not os.path.isfile(path):
            return {}
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            return {}

    def _flatten_numbers(value: Any) -> List[float]:
        if value is None:
            return []
        if isinstance(value, (int, float)):
            return [float(value)]
        if isinstance(value, list):
            out: List[float] = []
            for item in value:
                out.extend(_flatten_numbers(item))
            return out
        return []

    def _mean_abs_delta(a: Any, b: Any) -> float | None:
        av = _flatten_numbers(a)
        bv = _flatten_numbers(b)
        if not av or not bv or len(av) != len(bv):
            return None
        return sum(abs(x - y) for x, y in zip(av, bv)) / len(av)

    def _rms_delta(a: Any, b: Any) -> float | None:
        av = _flatten_numbers(a)
        bv = _flatten_numbers(b)
        if not av or not bv or len(av) != len(bv):
            return None
        return (sum((x - y) ** 2 for x, y in zip(av, bv)) / len(av)) ** 0.5

    rows: List[Dict[str, Any]] = []
    for row in summary_rows:
        path = row.get("design_json", "")
        if not path or not os.path.isfile(path):
            continue
        design = _load_json(path)
        if not design:
            continue
        initial = _load_json(os.path.join(row.get("ckpt_dir", ""), "camera_design_initial.json"))
        optics = design.get("optics", {}) or {}
        noise_quant = (design.get("noise_quantization", {}) or {})
        deploy_constraints = design.get("deployment_constraints", {}) or {}
        deploy_margins = design.get("constraint_margins", {}) or {}
        optics_constraints = (optics.get("physical_constraints", {}) or {}) if isinstance(optics, dict) else {}
        optics_margins = (optics.get("constraint_margins", {}) or {}) if isinstance(optics, dict) else {}
        init_optics = (initial.get("optics", {}) or {}) if initial else {}
        init_noise = (initial.get("noise_quantization", {}) or {}) if initial else {}
        flags = design.get("trainable_flags", {}) or {}
        rows.append(
            {
                "experiment": row.get("experiment", ""),
                "seed": row.get("seed", ""),
                "design_json": path,
                "initial_design_json": os.path.join(row.get("ckpt_dir", ""), "camera_design_initial.json")
                if initial
                else "",
                "optics_type": optics.get("type", ""),
                "optics_mode": design.get("optics_mode", ""),
                "num_zones": optics.get("num_zones", ""),
                "kernel_size": optics.get("kernel_size", ""),
                "learn_optics": flags.get("optics", ""),
                "learn_cfa": flags.get("cfa", ""),
                "learn_exposure": flags.get("exposure", ""),
                "exposure_gain": _fmt_float(_as_float(design.get("exposure_gain")), 5),
                "delta_exposure_gain": _fmt_float(
                    abs((_as_float(design.get("exposure_gain")) or 0.0) - (_as_float(initial.get("exposure_gain")) or 0.0))
                    if initial
                    else None,
                    6,
                ),
                "delta_cfa_l1": _fmt_float(
                    _mean_abs_delta(initial.get("cfa_weights_rgb") if initial else None, design.get("cfa_weights_rgb")),
                    6,
                ),
                "delta_psf_coeff_rms": _fmt_float(
                    _rms_delta(init_optics.get("coefficients"), optics.get("coefficients")),
                    6,
                ),
                "bit_depth": noise_quant.get("bit_depth_deploy", noise_quant.get("bit_depth", "")),
                "bit_depth_continuous": _fmt_float(_as_float(noise_quant.get("bit_depth_continuous")), 4),
                "delta_bit_depth": _fmt_float(
                    abs(
                        (_as_float(noise_quant.get("bit_depth_continuous")) or 0.0)
                        - (_as_float(init_noise.get("bit_depth_continuous")) or 0.0)
                    )
                    if initial
                    else None,
                    6,
                ),
                "read_noise_std": _fmt_float(_as_float(noise_quant.get("read_noise_std")), 6),
                "delta_read_noise_std": _fmt_float(
                    abs(
                        (_as_float(noise_quant.get("read_noise_std")) or 0.0)
                        - (_as_float(init_noise.get("read_noise_std")) or 0.0)
                    )
                    if initial
                    else None,
                    6,
                ),
                "shot_noise_scale": _fmt_float(_as_float(noise_quant.get("shot_noise_scale")), 6),
                "delta_shot_noise_scale": _fmt_float(
                    abs(
                        (_as_float(noise_quant.get("shot_noise_scale")) or 0.0)
                        - (_as_float(init_noise.get("shot_noise_scale")) or 0.0)
                    )
                    if initial
                    else None,
                    6,
                ),
                "learn_noise": noise_quant.get("learn_read_shot_noise", ""),
                "learn_bit_depth": noise_quant.get("learn_bit_depth", ""),
                "fixed_at_inference": deploy_constraints.get("fixed_at_inference", ""),
                "raw_pipeline_before_task_head": deploy_constraints.get("raw_pipeline_before_task_head", ""),
                "nonnegative_unit_sum_cfa_rows": deploy_constraints.get("nonnegative_unit_sum_cfa_rows", ""),
                "bounded_noise_parameters": deploy_constraints.get("bounded_noise_parameters", ""),
                "bounded_bit_depth": deploy_constraints.get("bounded_bit_depth", ""),
                "simulated_not_hardware_validated": deploy_constraints.get("simulated_not_hardware_validated", ""),
                "min_cfa_weight": _fmt_float(_as_float(deploy_margins.get("min_cfa_weight")), 6),
                "max_cfa_row_sum_error": _fmt_float(_as_float(deploy_margins.get("max_cfa_row_sum_error")), 8),
                "nonnegative_energy_normalized_psf": optics_constraints.get("nonnegative_energy_normalized_psf", ""),
                "bounded_low_order_coefficients": optics_constraints.get("bounded_low_order_coefficients", ""),
                "smooth_field_interpolation": optics_constraints.get("smooth_field_interpolation", ""),
                "max_psf_sum_error": _fmt_float(_as_float(optics_margins.get("max_abs_kernel_sum_error")), 8),
                "max_coeff_bound_fraction": _fmt_float(_as_float(optics_margins.get("max_coeff_bound_fraction")), 6),
            }
        )
    if not rows:
        return
    fields = [
        "experiment",
        "seed",
        "optics_type",
        "optics_mode",
        "num_zones",
        "kernel_size",
        "learn_optics",
        "learn_cfa",
        "learn_exposure",
        "exposure_gain",
        "delta_exposure_gain",
        "delta_cfa_l1",
        "delta_psf_coeff_rms",
        "bit_depth",
        "bit_depth_continuous",
        "delta_bit_depth",
        "read_noise_std",
        "delta_read_noise_std",
        "shot_noise_scale",
        "delta_shot_noise_scale",
        "learn_noise",
        "learn_bit_depth",
        "fixed_at_inference",
        "raw_pipeline_before_task_head",
        "nonnegative_unit_sum_cfa_rows",
        "bounded_noise_parameters",
        "bounded_bit_depth",
        "simulated_not_hardware_validated",
        "min_cfa_weight",
        "max_cfa_row_sum_error",
        "nonnegative_energy_normalized_psf",
        "bounded_low_order_coefficients",
        "smooth_field_interpolation",
        "max_psf_sum_error",
        "max_coeff_bound_fraction",
        "initial_design_json",
        "design_json",
    ]
    _write_csv(os.path.join(tables_dir, "design_inventory.csv"), rows, fields)


def collect_camera_measurements(summary_rows: List[Dict[str, str]], tables_dir: str) -> None:
    """Collect per-run camera visuals and measurement CSVs into paper_tables."""
    out_dir = os.path.join(tables_dir, "camera_measurements")
    os.makedirs(out_dir, exist_ok=True)
    manifest: List[Dict[str, str]] = []
    combined: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    suffix_to_table = {
        "_camera_summary.csv": "camera_summary_all.csv",
        "_psf_metrics.csv": "psf_metrics_all.csv",
        "_cfa_weights.csv": "cfa_weights_all.csv",
        "_noise_adc_curve.csv": "noise_adc_curve_all.csv",
    }

    for row in summary_rows:
        exp = row.get("experiment", "")
        seed = str(row.get("seed", ""))
        ckpt_dir = row.get("ckpt_dir", "")
        preview_dir = os.path.join(ckpt_dir, "design_preview")
        if not os.path.isdir(preview_dir):
            continue
        run_tag = f"{exp}_seed{seed}"
        run_out = os.path.join(out_dir, run_tag)
        os.makedirs(run_out, exist_ok=True)
        try:
            gif_paths = make_learning_gifs(
                Path(preview_dir),
                Path(run_out) / "learning_gifs",
                imgs_dir=None,
                duration_ms=420,
            )
            for gif_path in gif_paths:
                manifest.append(
                    {
                        "experiment": exp,
                        "seed": seed,
                        "artifact_type": "camera_learning_gif",
                        "source": str(gif_path),
                        "artifact": str(gif_path),
                    }
                )
        except Exception:
            pass
        for name in sorted(os.listdir(preview_dir)):
            src = os.path.join(preview_dir, name)
            if not os.path.isfile(src):
                continue
            dst = os.path.join(run_out, name)
            shutil.copy2(src, dst)
            manifest.append(
                {
                    "experiment": exp,
                    "seed": seed,
                    "artifact_type": "camera_measurement",
                    "source": src,
                    "artifact": dst,
                }
            )
            if name.endswith(".csv"):
                for suffix, table_name in suffix_to_table.items():
                    if name.endswith(suffix):
                        for item in _read_csv(src):
                            item = dict(item)
                            item["experiment"] = exp
                            item["seed"] = seed
                            item["source_file"] = name
                            combined[table_name].append(item)

    if manifest:
        _write_csv(
            os.path.join(out_dir, "manifest.csv"),
            manifest,
            ["experiment", "seed", "artifact_type", "source", "artifact"],
        )

    for table_name, rows in combined.items():
        if not rows:
            continue
        fields = ["experiment", "seed", "source_file"] + [
            k for k in rows[0].keys() if k not in ("experiment", "seed", "source_file")
        ]
        _write_csv(os.path.join(out_dir, table_name), rows, fields)


def collect_claim_readiness(summary_rows: List[Dict[str, str]], tables_dir: str) -> None:
    """Write explicit claim-audit tables so paper language follows evidence."""

    def _truthy(value: Any) -> bool:
        return str(value).strip().lower() in ("1", "true", "yes")

    main_rows = _read_csv(os.path.join(tables_dir, "main_results.csv"))
    design_rows = _read_csv(os.path.join(tables_dir, "design_inventory.csv"))
    stress_rows = _read_csv(os.path.join(tables_dir, "sensor_stress_deltas.csv"))
    rows_by_exp = {r.get("experiment", ""): r for r in main_rows}

    claim_rows: List[Dict[str, str]] = []

    matched_rows = []
    for row in main_rows:
        name = row.get("experiment", "")
        if "codesign" not in name or "robust" in name:
            continue
        fixed_name = name.replace("codesign", "fixed_camera")
        fixed = rows_by_exp.get(fixed_name)
        if fixed:
            delta = (_as_float(row.get("mIoU_mean")) or 0.0) - (_as_float(fixed.get("mIoU_mean")) or 0.0)
            matched_rows.append((name, fixed_name, delta))
    if matched_rows:
        best = max(matched_rows, key=lambda x: x[2])
        status = "supported" if best[2] > 0 else "not supported yet"
        evidence = f"{best[0]} vs {best[1]}: delta mIoU={best[2]:.4f}"
    else:
        status = "pending"
        evidence = "No matched co-design/fixed-camera pair found in main_results.csv."
    claim_rows.append(
        {
            "claim": "Matched task-driven camera co-design improves over a fixed camera frontend.",
            "status": status,
            "evidence": evidence,
            "safe_language": "Claim only on matched split/backbone/resolution pairs.",
            "blocked_language": "Do not claim general segmentation SOTA from this evidence.",
        }
    )

    stress_supported = []
    for row in stress_rows:
        delta_fixed = _as_float(row.get("delta_codesign_minus_fixed"))
        optics = _as_float(row.get("optics_only_mIoU"))
        codesign = _as_float(row.get("codesign_mIoU"))
        if delta_fixed is not None:
            stress_supported.append(delta_fixed > 0)
        if optics is not None and codesign is not None:
            stress_supported.append(codesign > optics)
    if stress_supported:
        status = "supported" if any(stress_supported) else "not supported yet"
        evidence = os.path.join(tables_dir, "sensor_stress_deltas.csv")
    else:
        status = "pending"
        evidence = "Run sensor_stress_matrix.yaml or sensor_stress_matrix_fast.yaml."
    claim_rows.append(
        {
            "claim": "The full optics+sensor method advances DeepLens-style optics-only co-design for driving tasks.",
            "status": status,
            "evidence": evidence,
            "safe_language": "Use 'advances DeepLens-style co-design' or 'outperforms optics-only under our driving stress matrix' when deltas are positive.",
            "blocked_language": "Do not write 'outperforms DeepLens' unless a named DeepLens baseline is reproduced and beaten.",
        }
    )

    constrained_designs = []
    for row in design_rows:
        if "codesign" not in row.get("experiment", ""):
            continue
        required = [
            _truthy(row.get("fixed_at_inference")),
            _truthy(row.get("nonnegative_unit_sum_cfa_rows")),
            _truthy(row.get("bounded_noise_parameters")),
            _truthy(row.get("bounded_bit_depth")),
            _truthy(row.get("nonnegative_energy_normalized_psf")),
            _truthy(row.get("bounded_low_order_coefficients")),
        ]
        constrained_designs.append(all(required))
    if constrained_designs:
        status = "supported" if all(constrained_designs) else "partial"
        evidence = os.path.join(tables_dir, "design_inventory.csv")
    else:
        status = "pending"
        evidence = "No co-design camera_design_best.json found."
    claim_rows.append(
        {
            "claim": "The learned camera is a fixed-at-inference, physically constrained simulated camera specification.",
            "status": status,
            "evidence": evidence,
            "safe_language": "Use 'deployment-oriented constrained simulation' and cite exported camera_design_best.json.",
            "blocked_language": "Do not claim validated real hardware without RAW/prototype/calibration experiments.",
        }
    )

    robust_rows = [r for r in stress_rows if r.get("delta_robust_minus_nominal", "") not in ("", None)]
    robust_ok = [
        (_as_float(r.get("delta_robust_minus_nominal")) or 0.0) >= 0.0
        for r in robust_rows
    ]
    claim_rows.append(
        {
            "claim": "Tolerance-aware co-design improves robustness to camera manufacturing/sensor perturbations.",
            "status": "supported" if robust_ok and any(robust_ok) else ("not supported yet" if robust_rows else "pending"),
            "evidence": os.path.join(tables_dir, "sensor_stress_deltas.csv") if robust_rows else "Run robust co-design rows in the stress matrix.",
            "safe_language": "Claim tolerance robustness only for stress regimes with positive robust-minus-nominal delta.",
            "blocked_language": "Do not imply hardware tolerance validation without measured device data.",
        }
    )

    claim_rows.append(
        {
            "claim": "State-of-the-art semantic segmentation.",
            "status": "not supported by this pipeline",
            "evidence": "Requires official leaderboard or matched reproduction of modern segmentation baselines.",
            "safe_language": "Use 'competitive lightweight accuracy-efficiency under a learned sensing frontend' only if numbers support it.",
            "blocked_language": "Do not claim SOTA mIoU unless same split, same metric, same resolution, and modern baselines are beaten.",
        }
    )

    fields = ["claim", "status", "evidence", "safe_language", "blocked_language"]
    _write_csv(os.path.join(tables_dir, "claim_readiness.csv"), claim_rows, fields)
    _write_markdown_table(os.path.join(tables_dir, "claim_readiness.md"), claim_rows, fields)
    with open(os.path.join(tables_dir, "claim_safe_language.md"), "w") as f:
        f.write("# Claim-Safe Paper Language\n\n")
        for row in claim_rows:
            f.write(f"## {row['claim']}\n\n")
            f.write(f"Status: **{row['status']}**\n\n")
            f.write(f"Safe language: {row['safe_language']}\n\n")
            f.write(f"Blocked language: {row['blocked_language']}\n\n")


def collect_qualitative_outputs(summary_rows: List[Dict[str, str]], tables_dir: str, max_per_run: int = 4) -> None:
    """Curate representative panels into one paper-facing folder with captions."""
    qual_dir = os.path.join(tables_dir, "qualitative")
    os.makedirs(qual_dir, exist_ok=True)
    run_dirs = [row.get("ckpt_dir", "") for row in summary_rows if row.get("ckpt_dir")]
    common_root = os.path.commonpath(run_dirs) if run_dirs else ""
    if common_root and os.path.isdir(common_root):
        story_rows = curate_story_figures(
            root=Path(os.path.abspath(common_root)),
            out_dir=Path(os.path.abspath(qual_dir)),
            max_per_story=max(1, max_per_run),
        )
        if story_rows:
            _write_csv(
                os.path.join(qual_dir, "manifest.csv"),
                [asdict(row) if hasattr(row, "__dataclass_fields__") else row for row in story_rows],
                [
                    "source",
                    "artifact",
                    "caption_txt",
                    "caption_json",
                    "experiment",
                    "seed",
                    "story_type",
                    "score",
                    "classes",
                    "rationale",
                ],
            )
            return

    manifest_rows: List[Dict[str, str]] = []
    for row in summary_rows:
        exp = row.get("experiment", "")
        seed = str(row.get("seed", ""))
        ckpt_dir = row.get("ckpt_dir", "")
        viz_dir = os.path.join(ckpt_dir, "viz")
        if not os.path.isdir(viz_dir):
            continue
        candidates = sorted(
            os.path.join(viz_dir, name)
            for name in os.listdir(viz_dir)
            if name.lower().endswith((".png", ".jpg", ".jpeg"))
        )
        if not candidates:
            continue
        picked = candidates[:max_per_run]
        for idx, src in enumerate(picked, 1):
            dst_name = f"{exp}_seed{seed}_{idx:02d}_{os.path.basename(src)}"
            dst = os.path.join(qual_dir, dst_name)
            shutil.copy2(src, dst)
            manifest_rows.append(
                {
                    "experiment": exp,
                    "seed": seed,
                    "source": src,
                    "artifact": dst,
                }
            )
    if manifest_rows:
        _write_csv(os.path.join(qual_dir, "manifest.csv"), manifest_rows, ["experiment", "seed", "source", "artifact"])


def write_run_manifest(matrix_path: str, out_root: str, tables_dir: str) -> None:
    with open(os.path.expanduser(matrix_path), "r") as f:
        matrix = yaml.safe_load(f)
    manifest = {
        "matrix": os.path.abspath(os.path.expanduser(matrix_path)),
        "base_config": os.path.abspath(os.path.expanduser(matrix["base_config"])),
        "out_root": out_root,
        "tables_dir": tables_dir,
        "input_signal_note": (
            "KITTI-360 data_2d_raw frames are rectified processed RGB. "
            "Sensor-mode experiments therefore evaluate a physically constrained "
            "differentiable camera-simulation frontend applied to processed RGB as "
            "a proxy scene signal, not validation on genuine camera RAW."
        ),
        "critical_reviewer_rows": [
            "codesign_full_liteseg",
            "ablate_no_optics",
            "ablate_fixed_optics",
            "ablate_fixed_camera_frontend",
            "rgb_liteseg",
        ],
        "fixed_at_inference_artifact": "camera_design_best.json",
    }
    os.makedirs(tables_dir, exist_ok=True)
    with open(os.path.join(tables_dir, "experiment_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)


def plot_paper_figures(tables_dir: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    def _display_name(name: str) -> str:
        mapping = {
            "kitti360_codesign_liteseg": "Co-design",
            "kitti360_fixed_camera_frontend": "Fixed camera",
            "kitti360_no_optics": "No optics",
            "kitti360_fixed_optics": "Fixed optics",
            "kitti360_fixed_cfa": "Fixed CFA",
            "kitti360_fixed_exposure": "Fixed exposure",
            "kitti360_fixed_noise_quant": "Fixed noise/ADC",
            "kitti360_no_noise_quant": "No noise/ADC",
            "kitti360_rgb_liteseg": "RGB LiteSeg",
            "kitti360_rgb_lraspp_mobilenetv3": "RGB LR-ASPP",
            "kitti360_rgb_deeplabv3_mobilenetv3": "RGB DeepLabV3",
            "kitti360_rgb_segformer_b0": "RGB SegFormer-B0",
            "cityscapes_codesign_liteseg": "Cityscapes co-design",
            "cityscapes_rgb_liteseg": "Cityscapes RGB",
            "acdc_codesign_liteseg": "ACDC co-design",
            "acdc_rgb_liteseg": "ACDC RGB",
            "kitti360_codesign_liteseg_w96_highres": "Co-design LiteSeg-W96 HR",
            "kitti360_fixed_camera_liteseg_w96_highres": "Fixed camera LiteSeg-W96 HR",
            "kitti360_rgb_liteseg_w96_highres": "RGB LiteSeg-W96 HR",
            "kitti360_rgb_segformer_b0_cityscapes_highres": "RGB SegFormer-B0 HR",
            "kitti360_codesign_segformer_b0_cityscapes_highres": "Co-design SegFormer-B0 HR",
            "codesign_full_liteseg": "Co-design",
            "ablate_fixed_camera_frontend": "Fixed camera",
            "rgb_liteseg": "RGB LiteSeg",
        }
        return mapping.get(name, name.replace("kitti360_", "").replace("_", " "))

    def _row_value(row: Dict[str, str], key: str) -> float:
        return _as_float(row.get(key)) or 0.0

    def _stress_parts(name: str) -> tuple[str, str] | None:
        """Return (regime, method) for stress_* experiment names."""
        if not name.startswith("stress_"):
            return None
        known = [
            ("_fixed_camera_", "fixed camera"),
            ("_optics_only_", "optics only"),
            ("_sensor_only_", "sensor only"),
            ("_robust_codesign_", "robust co-design"),
            ("_codesign_", "co-design"),
            ("_rgb_reference_", "RGB reference"),
            ("_rgb_", "RGB reference"),
        ]
        for token, method in known:
            if token in name:
                regime = name[len("stress_") : name.index(token)]
                return regime, method
        if name.startswith("stress_clean_rgb_reference"):
            return "clean", "RGB reference"
        return None

    main_rows = _read_csv(os.path.join(tables_dir, "main_results.csv"))
    if main_rows:
        labels = [r["experiment"] for r in main_rows]
        miou = [_as_float(r.get("mIoU_mean")) or 0.0 for r in main_rows]
        miou_std = [_as_float(r.get("mIoU_std")) or 0.0 for r in main_rows]
        fig, ax = plt.subplots(figsize=(max(8.0, 0.65 * len(labels)), 4.5))
        ax.bar(range(len(labels)), miou, yerr=miou_std, capsize=3)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("mIoU")
        ax.set_ylim(0.0, max(0.75, max(miou + [0.0]) + 0.05))
        ax.set_title("Reviewer matrix: mean validation mIoU")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(tables_dir, "main_results_miou.png"), dpi=180)
        plt.close(fig)

        stress_by_regime: Dict[str, Dict[str, Dict[str, str]]] = defaultdict(dict)
        for row in main_rows:
            parsed = _stress_parts(row.get("experiment", ""))
            if parsed is None:
                continue
            regime, method = parsed
            stress_by_regime[regime][method] = row
        stress_delta_rows: List[Dict[str, Any]] = []
        stress_regimes = sorted(r for r, methods in stress_by_regime.items() if "co-design" in methods or "robust co-design" in methods)
        if stress_regimes:
            methods_order = ["fixed camera", "optics only", "sensor only", "co-design", "robust co-design", "RGB reference"]
            xs = list(range(len(stress_regimes)))
            width = 0.13
            colors = {
                "fixed camera": "#8c939b",
                "optics only": "#6baed6",
                "sensor only": "#74c476",
                "co-design": "#1f77b4",
                "robust co-design": "#54278f",
                "RGB reference": "#d9822b",
            }
            fig, ax = plt.subplots(figsize=(max(7.0, 1.8 * len(stress_regimes)), 4.6))
            for midx, method in enumerate(methods_order):
                vals = []
                present_x = []
                for x, regime in zip(xs, stress_regimes):
                    row = stress_by_regime[regime].get(method)
                    if not row:
                        continue
                    vals.append(_row_value(row, "mIoU_mean"))
                    present_x.append(x + (midx - 2.5) * width)
                if vals:
                    ax.bar(present_x, vals, width=width, label=method, color=colors.get(method, None))
            ax.set_ylabel("Validation mIoU")
            ax.set_xticks(xs)
            ax.set_xticklabels([r.replace("_", " ") for r in stress_regimes])
            max_val = max(
                [_row_value(row, "mIoU_mean") for methods in stress_by_regime.values() for row in methods.values()]
                + [0.0]
            )
            ax.set_ylim(0.0, max(0.7, max_val + 0.05))
            ax.set_title("Constrained Sensing: Fixed vs Learned Camera")
            ax.grid(axis="y", alpha=0.25)
            ax.legend(frameon=False, ncol=2)
            fig.tight_layout()
            fig.savefig(os.path.join(tables_dir, "sensor_stress_grouped_miou.png"), dpi=240)
            plt.close(fig)

            for regime in stress_regimes:
                methods = stress_by_regime[regime]
                codesign = methods.get("robust co-design") or methods.get("co-design")
                fixed = methods.get("fixed camera")
                if not codesign or not fixed:
                    continue
                code_m = _row_value(codesign, "mIoU_mean")
                fixed_m = _row_value(fixed, "mIoU_mean")
                row = {
                    "regime": regime,
                    "codesign_mIoU": _fmt_float(code_m),
                    "fixed_camera_mIoU": _fmt_float(fixed_m),
                    "delta_codesign_minus_fixed": _fmt_float(code_m - fixed_m),
                    "codesign_experiment": codesign.get("experiment", ""),
                    "fixed_camera_experiment": fixed.get("experiment", ""),
                }
                if "co-design" in methods and "robust co-design" in methods:
                    row["nominal_codesign_mIoU"] = _fmt_float(_row_value(methods["co-design"], "mIoU_mean"))
                    row["delta_robust_minus_nominal"] = _fmt_float(code_m - _row_value(methods["co-design"], "mIoU_mean"))
                for method in ("optics only", "sensor only", "RGB reference"):
                    if method in methods:
                        row[f"{method.replace(' ', '_')}_mIoU"] = _fmt_float(_row_value(methods[method], "mIoU_mean"))
                stress_delta_rows.append(row)
            if stress_delta_rows:
                fields = [
                    "regime",
                    "codesign_mIoU",
                    "fixed_camera_mIoU",
                    "delta_codesign_minus_fixed",
                    "optics_only_mIoU",
                    "sensor_only_mIoU",
                    "nominal_codesign_mIoU",
                    "delta_robust_minus_nominal",
                    "RGB_reference_mIoU",
                    "codesign_experiment",
                    "fixed_camera_experiment",
                ]
                _write_csv(os.path.join(tables_dir, "sensor_stress_deltas.csv"), stress_delta_rows, fields)
                _write_markdown_table(os.path.join(tables_dir, "sensor_stress_deltas.md"), stress_delta_rows, fields)

                labels_delta = [r["regime"].replace("_", " ") for r in stress_delta_rows]
                deltas = [_as_float(r.get("delta_codesign_minus_fixed")) or 0.0 for r in stress_delta_rows]
                fig, ax = plt.subplots(figsize=(max(6.0, 1.8 * len(labels_delta)), 3.8))
                ax.axhline(0.0, color="#333333", linewidth=1.0)
                ax.bar(range(len(labels_delta)), deltas, color=["#1f77b4" if v >= 0 else "#b04a4a" for v in deltas])
                ax.set_xticks(range(len(labels_delta)))
                ax.set_xticklabels(labels_delta)
                ax.set_ylabel("mIoU gain over fixed camera")
                ax.set_title("Learned Optics-Sensor Gain Under Hardware Stress")
                ax.grid(axis="y", alpha=0.25)
                fig.tight_layout()
                fig.savefig(os.path.join(tables_dir, "sensor_stress_codesign_gain.png"), dpi=240)
                plt.close(fig)

        claim_order = [
            "kitti360_codesign_liteseg",
            "kitti360_fixed_camera_frontend",
            "kitti360_no_optics",
            "kitti360_fixed_optics",
            "kitti360_fixed_cfa",
            "kitti360_fixed_exposure",
            "kitti360_fixed_noise_quant",
            "kitti360_rgb_liteseg",
        ]
        rows_by_exp = {r.get("experiment", ""): r for r in main_rows}
        claim_rows = [rows_by_exp[e] for e in claim_order if e in rows_by_exp]
        if len(claim_rows) >= 2:
            labels = [_display_name(r["experiment"]) for r in claim_rows]
            vals = [_row_value(r, "mIoU_mean") for r in claim_rows]
            errs = [_row_value(r, "mIoU_std") for r in claim_rows]
            colors = ["#1f77b4"] + ["#9aa3ad"] * (len(labels) - 2) + ["#d9822b"]
            fig, ax = plt.subplots(figsize=(8.0, 4.2))
            ax.bar(range(len(labels)), vals, yerr=errs, capsize=3, color=colors[: len(labels)])
            ax.set_ylabel("Validation mIoU")
            ax.set_ylim(0.0, max(0.7, max(vals) + max(errs + [0.0]) + 0.05))
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=25, ha="right")
            ax.set_title("KITTI-360 Controlled Optics-Sensor Ablations")
            ax.grid(axis="y", alpha=0.25)
            fig.tight_layout()
            fig.savefig(os.path.join(tables_dir, "claim_kitti360_ablation_miou.png"), dpi=240)
            plt.close(fig)

        dataset_pairs = [
            ("kitti360_codesign_liteseg", "kitti360_rgb_liteseg", "KITTI-360"),
            ("cityscapes_codesign_liteseg", "cityscapes_rgb_liteseg", "Cityscapes"),
            ("acdc_codesign_liteseg", "acdc_rgb_liteseg", "ACDC"),
        ]
        pair_rows = [(rows_by_exp.get(a), rows_by_exp.get(b), name) for a, b, name in dataset_pairs]
        pair_rows = [(a, b, name) for a, b, name in pair_rows if a and b]
        if pair_rows:
            labels = [name for _, _, name in pair_rows]
            codesign = [_row_value(a, "mIoU_mean") for a, _, _ in pair_rows]
            rgb = [_row_value(b, "mIoU_mean") for _, b, _ in pair_rows]
            xs = list(range(len(labels)))
            width = 0.36
            fig, ax = plt.subplots(figsize=(6.8, 4.0))
            ax.bar([x - width / 2 for x in xs], codesign, width=width, label="Co-design", color="#1f77b4")
            ax.bar([x + width / 2 for x in xs], rgb, width=width, label="RGB", color="#d9822b")
            ax.set_ylabel("Validation mIoU")
            ax.set_xticks(xs)
            ax.set_xticklabels(labels)
            ax.set_ylim(0.0, max(0.7, max(codesign + rgb) + 0.05))
            ax.set_title("Co-design vs RGB Across Datasets")
            ax.legend(frameon=False)
            ax.grid(axis="y", alpha=0.25)
            fig.tight_layout()
            fig.savefig(os.path.join(tables_dir, "codesign_vs_rgb_by_dataset.png"), dpi=240)
            plt.close(fig)

        params = [_as_float(r.get("params_model_m_mean")) for r in main_rows]
        latency = [_as_float(r.get("latency_chain_ms_mean")) for r in main_rows]
        if any(v is not None for v in params):
            fig, ax = plt.subplots(figsize=(7.0, 4.8))
            for label, x, y in zip(labels, params, miou):
                if x is None:
                    continue
                ax.scatter(x, y, s=48)
                ax.annotate(label, (x, y), xytext=(4, 3), textcoords="offset points", fontsize=7)
            ax.set_xlabel("model parameters (M)")
            ax.set_ylabel("mIoU")
            ax.set_title("Accuracy vs model capacity")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(tables_dir, "accuracy_vs_params.png"), dpi=180)
            plt.close(fig)

        if any(v is not None for v in latency):
            fig, ax = plt.subplots(figsize=(7.0, 4.8))
            for label, x, y in zip(labels, latency, miou):
                if x is None:
                    continue
                ax.scatter(x, y, s=48)
                ax.annotate(label, (x, y), xytext=(4, 3), textcoords="offset points", fontsize=7)
            ax.set_xlabel("sensor+model latency (ms)")
            ax.set_ylabel("mIoU")
            ax.set_title("Accuracy vs end-to-end latency")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(tables_dir, "accuracy_vs_latency.png"), dpi=180)
            plt.close(fig)

    robust_rows = _read_csv(os.path.join(tables_dir, "robustness_summary.csv"))
    if robust_rows:
        by_kind: Dict[str, List[Dict[str, str]]] = defaultdict(list)
        for row in robust_rows:
            by_kind[row.get("kind", "")].append(row)
        for kind, rows in by_kind.items():
            if kind in ("", "clean"):
                continue
            fig, ax = plt.subplots(figsize=(7.0, 4.5))
            by_exp: Dict[str, List[Dict[str, str]]] = defaultdict(list)
            for row in rows:
                by_exp[row.get("experiment", "")].append(row)
            for exp, exp_rows in sorted(by_exp.items()):
                exp_rows = sorted(exp_rows, key=lambda r: _as_float(r.get("level")) or 0.0)
                xs = [_as_float(r.get("level")) or 0.0 for r in exp_rows]
                ys = [_as_float(r.get("mIoU_mean")) or 0.0 for r in exp_rows]
                ax.plot(xs, ys, marker="o", linewidth=1.5, label=exp)
            ax.set_xlabel("corruption level")
            ax.set_ylabel("mIoU")
            ax.set_ylim(0.0, 1.0)
            ax.set_title(f"Robustness: {kind}")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
            fig.tight_layout()
            safe_kind = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in kind)
            fig.savefig(os.path.join(tables_dir, f"robustness_{safe_kind}.png"), dpi=180)
            plt.close(fig)

    measurement_manifest = _read_csv(os.path.join(tables_dir, "camera_measurements", "manifest.csv"))
    if measurement_manifest:
        by_run: Dict[Tuple[str, str], Dict[str, str]] = defaultdict(dict)
        for row in measurement_manifest:
            artifact = row.get("artifact", "")
            if not artifact:
                continue
            by_run[(row.get("experiment", ""), row.get("seed", ""))][os.path.basename(artifact)] = artifact

        preferred: Tuple[str, str] | None = None
        preferred_score = -1
        for key, artifacts in by_run.items():
            exp, _seed = key
            score = 0
            if "codesign" in exp:
                score += 10
            if "robust" in exp:
                score += 3
            if any(name.startswith("best_") for name in artifacts):
                score += 2
            if score > preferred_score:
                preferred = key
                preferred_score = score

        if preferred is not None:
            artifacts = by_run[preferred]

            def _artifact(*names: str) -> str:
                for name in names:
                    if name in artifacts:
                        return artifacts[name]
                return ""

            panels = [
                ("Learned field PSFs", _artifact("best_psf_kernels.png", "epoch1_psf_kernels.png", "initial_psf_kernels.png")),
                ("Field PSF width", _artifact("best_psf_rms_by_field.png", "epoch1_psf_rms_by_field.png", "initial_psf_rms_by_field.png")),
                ("Learned CFA response", _artifact("best_cfa_bars.png", "best_cfa_weights.png", "epoch1_cfa_bars.png", "initial_cfa_weights.png")),
                ("Noise and ADC response", _artifact("best_noise_adc_curve.png", "epoch1_noise_adc_curve.png", "initial_noise_adc_curve.png")),
            ]
            panels = [(title, path) for title, path in panels if path and os.path.isfile(path)]
            if panels:
                fig = plt.figure(figsize=(14.0, 7.6))
                gs = fig.add_gridspec(2, max(4, len(panels)), height_ratios=[0.55, 1.45], hspace=0.28, wspace=0.18)
                ax = fig.add_subplot(gs[0, :])
                ax.axis("off")
                blocks = [
                    ("RGB proxy", "#f2f2f2"),
                    ("RAW-linear\nunprocess", "#d9ead3"),
                    ("learned field PSF", "#cfe2f3"),
                    ("exposure + CFA", "#fff2cc"),
                    ("noise + n-bit ADC", "#f4cccc"),
                    ("task backbone", "#d0e0e3"),
                    ("segmentation\nmetrics", "#eeeeee"),
                ]
                for i, (text, color) in enumerate(blocks):
                    x = 0.015 + i * 0.14
                    rect = plt.Rectangle((x, 0.36), 0.112, 0.34, facecolor=color, edgecolor="#222222", linewidth=1.0)
                    ax.add_patch(rect)
                    ax.text(x + 0.056, 0.53, text, ha="center", va="center", fontsize=9)
                    if i < len(blocks) - 1:
                        ax.annotate("", xy=(x + 0.134, 0.53), xytext=(x + 0.114, 0.53), arrowprops=dict(arrowstyle="->", lw=1.2))
                ax.text(
                    0.5,
                    0.12,
                    "Evidence exported from the learned fixed-at-inference camera: PSF field response, CFA spectrum, exposure/noise/ADC numbers.",
                    ha="center",
                    va="center",
                    fontsize=10,
                )

                for idx, (title, path) in enumerate(panels[:4]):
                    panel_ax = fig.add_subplot(gs[1, idx])
                    try:
                        panel_ax.imshow(plt.imread(path))
                    except Exception:
                        panel_ax.text(0.5, 0.5, os.path.basename(path), ha="center", va="center")
                    panel_ax.set_title(title, fontsize=10)
                    panel_ax.set_xticks([])
                    panel_ax.set_yticks([])
                    for spine in panel_ax.spines.values():
                        spine.set_color("#444444")
                        spine.set_linewidth(0.8)
                fig.suptitle(f"Optics-Sensor Co-Design Measurement Panel: {preferred[0]} seed {preferred[1]}", fontsize=13)
                fig.tight_layout()
                evidence_path = os.path.join(tables_dir, "learned_camera_evidence.png")
                fig.savefig(evidence_path, dpi=240)
                plt.close(fig)

                table_root = Path(tables_dir).resolve()
                for parent in [table_root] + list(table_root.parents):
                    imgs_dir = parent / "imgs"
                    if imgs_dir.is_dir():
                        shutil.copy2(evidence_path, imgs_dir / "raw2task_learned_camera_evidence.png")
                        break

    fig, ax = plt.subplots(figsize=(11.0, 2.6))
    ax.axis("off")
    blocks = [
        ("processed RGB\nproxy signal", "#d9ead3"),
        ("bounded field PSF\nlearned/fixed", "#cfe2f3"),
        ("exposure\nsingle gain", "#fff2cc"),
        ("2x2 CFA\nspectral weights", "#eadcf8"),
        ("noise + ADC\nPoisson/Gaussian/bit", "#f4cccc"),
        ("segmentation\nbackbone", "#d0e0e3"),
        ("mask + metrics\nmIoU/IoU/latency", "#eeeeee"),
    ]
    for i, (text, color) in enumerate(blocks):
        x = 0.02 + i * 0.14
        rect = plt.Rectangle((x, 0.34), 0.115, 0.34, facecolor=color, edgecolor="#333333", linewidth=1.0)
        ax.add_patch(rect)
        ax.text(x + 0.0575, 0.51, text, ha="center", va="center", fontsize=9)
        if i < len(blocks) - 1:
            ax.annotate("", xy=(x + 0.135, 0.51), xytext=(x + 0.115, 0.51), arrowprops=dict(arrowstyle="->", lw=1.2))
    ax.text(
        0.5,
        0.13,
        "Training optimizes camera and model jointly; inference uses the exported fixed camera_design_best.json.",
        ha="center",
        va="center",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(os.path.join(tables_dir, "pipeline_diagram.png"), dpi=220)
    plt.close(fig)


def _matrix_out_root(matrix_path: str) -> str:
    with open(os.path.expanduser(matrix_path), "r") as f:
        matrix = yaml.safe_load(f)
    return os.path.abspath(os.path.expanduser(matrix.get("out_root", "./runs/raw2task_review_matrix")))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--only", default="", help="Comma-separated experiment names for the matrix.")
    parser.add_argument("--seeds", default="", help="Comma-separated seed override.")
    parser.add_argument("--skip-matrix", action="store_true", help="Only aggregate existing results.")
    parser.add_argument("--skip-existing", action="store_true", help="Do not retrain run dirs that already have last.pt.")
    parser.add_argument("--append-summary", action="store_true", help="Append to summary.csv instead of overwriting it.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--tables-dir", default="")
    parser.add_argument(
        "--robustness-experiments",
        default=(
            "codesign_full_liteseg,rgb_liteseg,ablate_no_optics,ablate_fixed_camera_frontend,"
            "kitti360_codesign_liteseg,kitti360_rgb_liteseg,kitti360_no_optics,kitti360_fixed_camera_frontend"
        ),
        help="Comma-separated experiments to robustness-sweep. Empty string skips robustness.",
    )
    parser.add_argument("--robustness-sweep", default="")
    parser.add_argument("--max-robustness-batches", type=int, default=0)
    parser.add_argument("--force-robustness", action="store_true")
    args = parser.parse_args()

    out_root = _matrix_out_root(args.matrix)
    tables_dir = os.path.abspath(os.path.expanduser(args.tables_dir or os.path.join(out_root, "paper_tables")))
    os.makedirs(tables_dir, exist_ok=True)

    seeds = [int(x) for x in _parse_csv_list(args.seeds)] if args.seeds else None
    if not args.skip_matrix:
        run_matrix(
            matrix_path=args.matrix,
            only=_parse_csv_list(args.only),
            seeds_override=seeds,
            dry_run=args.dry_run,
            skip_existing=args.skip_existing,
            fresh_summary=not args.append_summary,
        )

    summary_csv = os.path.join(out_root, "summary.csv")
    if args.dry_run:
        print(f"Dry run complete. Tables would be written under {tables_dir}")
        return

    summary_rows = aggregate_main_results(summary_csv, tables_dir)
    run_robustness(
        summary_rows,
        experiments=_parse_csv_list(args.robustness_experiments),
        sweep=args.robustness_sweep,
        max_batches=args.max_robustness_batches,
        force=args.force_robustness,
    )
    collect_robustness(summary_rows, tables_dir)
    collect_design_inventory(summary_rows, tables_dir)
    collect_camera_measurements(summary_rows, tables_dir)
    collect_claim_readiness(summary_rows, tables_dir)
    collect_qualitative_outputs(summary_rows, tables_dir)
    write_run_manifest(args.matrix, out_root, tables_dir)
    plot_paper_figures(tables_dir)
    print(f"Saved paper-ready tables to {tables_dir}")


if __name__ == "__main__":
    main()
