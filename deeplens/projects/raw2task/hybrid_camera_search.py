"""Hybrid physical-AI camera search for task-driven co-design.

This module implements the outer loop that plain backprop is missing:

1. Start from a physically meaningful co-design experiment template.
2. Generate manufacturable-ish optics/sensor candidates with deterministic
   quasi-random sampling.
3. Train each candidate with the existing paper experiment runner.
4. Rank candidates by validation task score and export the best camera designs.

The goal is not to replace differentiable training. It gives differentiable
training better starting points and tests discrete/non-smooth sensor choices
such as ADC bit depth, exposure regime, noise scale, and PSF bounds.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch
import yaml

from deeplens.projects.raw2task.run_review_matrix import _deep_update


DEFAULT_KITTI_MATRIX = "deeplens/projects/raw2task/configs/constrained_codesign_matrix_fast.yaml"
DEFAULT_CITY_MATRIX = "deeplens/projects/raw2task/configs/cityscapes_codesign_matrix_fast.yaml"
PHYSICS_LOCAL_ANCHORS = [
    {
        "source": "broad_cand004",
        "pixel_pitch_um": 2.5415012285113336,
        "f_number": 3.7082095623016356,
        "exposure_ms": 9.614285759795592,
        "scene_lux": 18.986306882128552,
        "quantum_efficiency": 0.6366339206695557,
        "full_well_e": 7921.146411833057,
        "read_noise_e": 1.523036523507045,
        "defocus_um": 1.2963,
        "field_aberration": 0.17078200355172157,
        "cfa_init_floor": 0.059826560020446784,
        "dark_current_e_s": 2.6074909436790317,
    },
    {
        "source": "broad_cand003",
        "pixel_pitch_um": 2.265866383910179,
        "f_number": 1.7793955445289613,
        "exposure_ms": 16.50741610788805,
        "scene_lux": 13.919342577831026,
        "quantum_efficiency": 0.5954985618591309,
        "full_well_e": 14124.871428502604,
        "read_noise_e": 3.57666804462879,
        "defocus_um": 0.3835,
        "field_aberration": 0.5629454970359802,
        "cfa_init_floor": 0.07416150569915772,
        "dark_current_e_s": 8.270717560080696,
    },
]


SEARCH_FIELDS = [
    "candidate",
    "experiment",
    "seed",
    "objective",
    "mIoU",
    "pixel_acc",
    "best_val",
    "candidate_prior",
    "local_anchor",
    "exposure_init",
    "bit_depth",
    "read_noise_std",
    "shot_noise_scale",
    "cfa_init_floor",
    "base_sigma_px",
    "max_sigma_px",
    "max_shift_px",
    "field_sigma",
    "pixel_pitch_um",
    "f_number",
    "exposure_ms",
    "scene_lux",
    "quantum_efficiency",
    "full_well_e",
    "read_noise_e",
    "dark_current_e_s",
    "wavelength_nm",
    "electrons_per_pixel",
    "shot_noise_e",
    "dark_noise_e",
    "snr_db",
    "dynamic_range_db",
    "diffraction_sigma_px",
    "defocus_sigma_px",
    "pixel_aperture_sigma_px",
    "center_mtf50_cyc_px",
    "edge_mtf50_cyc_px",
    "ckpt_dir",
    "best_ckpt",
    "avg_best_ckpt",
    "design_json",
    "avg_bestk_mIoU",
    "avg_bestk_pixel_acc",
    "avg_bestk_metrics",
    "rank_metric",
]


def _load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(os.path.expanduser(str(path)), "r") as f:
        return yaml.safe_load(f) or {}


def _write_yaml(path: str | Path, payload: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def _read_csv(path: str | Path) -> List[Dict[str, str]]:
    if not Path(path).is_file():
        return []
    with open(path, "r", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: str | Path, rows: List[Dict[str, Any]], fields: Iterable[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return default
        return float(value)
    except Exception:
        return default


def _candidate_name(prefix: str, idx: int) -> str:
    return f"{prefix}_cand{idx:03d}"


def _find_template(matrix: Dict[str, Any], template_name: str) -> Dict[str, Any]:
    experiments = list(matrix.get("experiments", []))
    if template_name:
        for exp in experiments:
            if str(exp.get("name", "")) == template_name:
                return exp
        raise ValueError(f"Template experiment not found: {template_name}")
    preferred = [
        "lowlight_codesign",
        "lowbit4_codesign",
        "clean_codesign",
        "codesign",
    ]
    for needle in preferred:
        for exp in experiments:
            name = str(exp.get("name", "")).lower()
            if needle in name and "fixed" not in name and "rgb" not in name:
                return exp
    raise ValueError("No co-design template experiment found in matrix.")


def _sobol_candidates(num_candidates: int, seed: int, dimension: int) -> torch.Tensor:
    engine = torch.quasirandom.SobolEngine(dimension=int(dimension), scramble=True, seed=int(seed))
    return engine.draw(int(num_candidates)).clamp(1e-6, 1.0 - 1e-6)


def _lerp(u: float, lo: float, hi: float) -> float:
    return float(lo + (hi - lo) * float(u))


def _clamp(value: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, value)))


def _log_lerp(u: float, lo: float, hi: float) -> float:
    return float(math.exp(_lerp(float(u), math.log(lo), math.log(hi))))


def _gaussian_mtf50(sigma_px: float) -> float:
    sigma_px = max(float(sigma_px), 1e-6)
    return float(math.sqrt(math.log(2.0)) / (2.0 * math.pi * sigma_px))


def _physics_params_from_latents(latents: Dict[str, float], prior_name: str, anchor_name: str = "") -> Dict[str, Any]:
    pixel_pitch_um = float(latents["pixel_pitch_um"])
    f_number = float(latents["f_number"])
    exposure_ms = float(latents["exposure_ms"])
    scene_lux = float(latents["scene_lux"])
    quantum_efficiency = float(latents["quantum_efficiency"])
    full_well_e = float(latents["full_well_e"])
    read_noise_e = float(latents["read_noise_e"])
    defocus_um = float(latents["defocus_um"])
    field_aberration = float(latents["field_aberration"])
    cfa_init_floor = float(latents["cfa_init_floor"])
    dark_current_e_s = float(latents["dark_current_e_s"])

    wavelength_nm = 550.0
    exposure_s = exposure_ms * 1e-3

    # Scene-referred electron proxy. The constant only fixes units for our
    # simulator scale; the sampled latents determine candidate ordering.
    lux_ms_um2_to_e = 0.85
    incident_e = scene_lux * exposure_ms * (pixel_pitch_um**2) * quantum_efficiency * lux_ms_um2_to_e
    electrons_per_pixel = _clamp(incident_e, 1.0, full_well_e)
    dark_noise_e = math.sqrt(max(dark_current_e_s * exposure_s, 0.0))
    shot_noise_e = math.sqrt(max(electrons_per_pixel, 1.0))
    total_noise_e = math.sqrt(electrons_per_pixel + read_noise_e**2 + dark_noise_e**2)
    snr = electrons_per_pixel / max(total_noise_e, 1e-6)
    snr_db = 20.0 * math.log10(max(snr, 1e-6))
    dynamic_range_db = 20.0 * math.log10(max(full_well_e / max(read_noise_e, 1e-6), 1e-6))

    exposure_ratio = _clamp(electrons_per_pixel / max(0.45 * full_well_e, 1.0), 0.015, 0.95)
    exposure_init = math.log(exposure_ratio)

    if dynamic_range_db >= 75.0:
        bit_depth = 8
    elif dynamic_range_db >= 65.0:
        bit_depth = 7
    elif dynamic_range_db >= 55.0:
        bit_depth = 6
    else:
        bit_depth = 5

    adc_levels = float((1 << bit_depth) - 1)
    electrons_per_dn = full_well_e / adc_levels
    read_noise_std = _clamp(read_noise_e / max(electrons_per_dn, 1e-6), 0.05, 3.25)
    shot_noise_scale = _clamp(0.35 * math.sqrt(100.0 / max(electrons_per_pixel, 1.0)), 0.05, 0.85)

    wavelength_um = wavelength_nm * 1e-3
    airy_radius_px = 1.22 * wavelength_um * f_number / pixel_pitch_um
    diffraction_sigma_px = airy_radius_px / 2.355
    pixel_aperture_sigma_px = 1.0 / math.sqrt(12.0)
    defocus_sigma_px = (defocus_um / pixel_pitch_um) / 2.355
    base_sigma_px = math.sqrt(
        diffraction_sigma_px**2 + pixel_aperture_sigma_px**2 + defocus_sigma_px**2
    )
    base_sigma_px = _clamp(base_sigma_px, 0.28, 1.15)
    max_sigma_px = _clamp(base_sigma_px * (1.0 + 1.55 * field_aberration) + 0.12, base_sigma_px + 0.20, 2.40)
    max_shift_px = _clamp(0.10 + 0.78 * field_aberration + 0.04 * (f_number - 1.6), 0.10, 0.85)
    field_sigma = _clamp(field_aberration, 0.15, 0.85)

    return {
        "candidate_prior": prior_name,
        "local_anchor": anchor_name,
        "exposure_init": exposure_init,
        "bit_depth": float(bit_depth),
        "read_noise_std": read_noise_std,
        "shot_noise_scale": shot_noise_scale,
        "cfa_init_floor": cfa_init_floor,
        "base_sigma_px": base_sigma_px,
        "max_sigma_px": max_sigma_px,
        "max_shift_px": max_shift_px,
        "field_sigma": field_sigma,
        "pixel_pitch_um": pixel_pitch_um,
        "f_number": f_number,
        "exposure_ms": exposure_ms,
        "scene_lux": scene_lux,
        "quantum_efficiency": quantum_efficiency,
        "full_well_e": full_well_e,
        "read_noise_e": read_noise_e,
        "dark_current_e_s": dark_current_e_s,
        "wavelength_nm": wavelength_nm,
        "electrons_per_pixel": electrons_per_pixel,
        "shot_noise_e": shot_noise_e,
        "dark_noise_e": dark_noise_e,
        "snr_db": snr_db,
        "dynamic_range_db": dynamic_range_db,
        "diffraction_sigma_px": diffraction_sigma_px,
        "defocus_sigma_px": defocus_sigma_px,
        "pixel_aperture_sigma_px": pixel_aperture_sigma_px,
        "center_mtf50_cyc_px": _gaussian_mtf50(base_sigma_px),
        "edge_mtf50_cyc_px": _gaussian_mtf50(max_sigma_px),
    }


def _heuristic_candidate_params(u: torch.Tensor) -> Dict[str, Any]:
    vals = [float(x) for x in u.tolist()]
    bit_options = [4, 5, 6, 7, 8]
    bit = bit_options[min(len(bit_options) - 1, int(vals[1] * len(bit_options)))]
    base_sigma = _lerp(vals[5], 0.35, 0.85)
    max_sigma = max(base_sigma + 0.45, _lerp(vals[6], 1.25, 2.40))
    return {
        "exposure_init": _lerp(vals[0], -3.40, -0.60),
        "bit_depth": float(bit),
        "read_noise_std": _lerp(vals[2], 0.50, 3.25),
        "shot_noise_scale": _lerp(vals[3], 0.12, 0.85),
        "cfa_init_floor": _lerp(vals[4], 0.02, 0.16),
        "base_sigma_px": base_sigma,
        "max_sigma_px": max_sigma,
        "max_shift_px": _lerp(vals[7], 0.15, 0.85),
        "field_sigma": _lerp(vals[8], 0.20, 0.85),
    }


def _physics_candidate_params(u: torch.Tensor) -> Dict[str, Any]:
    vals = [float(x) for x in u.tolist()]

    # Automotive perception cameras are usually constrained by small pixels,
    # fast lenses, low-light exposure time, full-well capacity, and ADC/read
    # noise. We sample those physical latents, then derive the simulator knobs.
    pixel_pitch_um = _lerp(vals[0], 2.1, 4.2)
    f_number = _lerp(vals[1], 1.6, 4.0)
    exposure_ms = _log_lerp(vals[2], 2.0, 20.0)
    scene_lux = _log_lerp(vals[3], 0.3, 30.0)
    quantum_efficiency = _lerp(vals[4], 0.35, 0.75)
    full_well_e = _log_lerp(vals[5], 6000.0, 35000.0)
    read_noise_e = _log_lerp(vals[6], 0.8, 5.0)
    defocus_um = _lerp(vals[7], 0.0, 1.6)
    field_aberration = _lerp(vals[8], 0.15, 0.75)
    cfa_init_floor = _lerp(vals[9], 0.02, 0.10)
    dark_current_e_s = _log_lerp(vals[10], 0.5, 30.0) if len(vals) > 10 else 5.0

    return _physics_params_from_latents(
        {
            "pixel_pitch_um": pixel_pitch_um,
            "f_number": f_number,
            "exposure_ms": exposure_ms,
            "scene_lux": scene_lux,
            "quantum_efficiency": quantum_efficiency,
            "full_well_e": full_well_e,
            "read_noise_e": read_noise_e,
            "defocus_um": defocus_um,
            "field_aberration": field_aberration,
            "cfa_init_floor": cfa_init_floor,
            "dark_current_e_s": dark_current_e_s,
        },
        "physics",
    )


def _local_multiplier(u: float, span: float) -> float:
    return float(math.exp((float(u) - 0.5) * math.log(span)))


def _physics_local_candidate_params(u: torch.Tensor) -> Dict[str, Any]:
    vals = [float(x) for x in u.tolist()]
    anchor_idx = min(len(PHYSICS_LOCAL_ANCHORS) - 1, int(vals[0] * len(PHYSICS_LOCAL_ANCHORS)))
    anchor = PHYSICS_LOCAL_ANCHORS[anchor_idx]
    latents = {
        "pixel_pitch_um": _clamp(anchor["pixel_pitch_um"] + _lerp(vals[1], -0.35, 0.35), 2.1, 4.2),
        "f_number": _clamp(anchor["f_number"] + _lerp(vals[2], -0.55, 0.55), 1.6, 4.0),
        "exposure_ms": _clamp(anchor["exposure_ms"] * _local_multiplier(vals[3], 1.75), 2.0, 20.0),
        "scene_lux": _clamp(anchor["scene_lux"] * _local_multiplier(vals[4], 2.25), 0.3, 30.0),
        "quantum_efficiency": _clamp(anchor["quantum_efficiency"] + _lerp(vals[5], -0.07, 0.07), 0.35, 0.75),
        "full_well_e": _clamp(anchor["full_well_e"] * _local_multiplier(vals[6], 1.75), 6000.0, 35000.0),
        "read_noise_e": _clamp(anchor["read_noise_e"] * _local_multiplier(vals[7], 1.65), 0.8, 5.0),
        "defocus_um": _clamp(anchor["defocus_um"] + _lerp(vals[8], -0.35, 0.35), 0.0, 1.6),
        "field_aberration": _clamp(anchor["field_aberration"] + _lerp(vals[9], -0.16, 0.16), 0.15, 0.75),
        "cfa_init_floor": _clamp(anchor["cfa_init_floor"] + _lerp(vals[10], -0.025, 0.025), 0.02, 0.10),
        "dark_current_e_s": _clamp(anchor["dark_current_e_s"] * _local_multiplier(vals[11], 2.0), 0.5, 30.0),
    }
    return _physics_params_from_latents(latents, "physics_local", anchor["source"])


def _candidate_params(u: torch.Tensor, candidate_prior: str) -> Dict[str, Any]:
    if candidate_prior == "physics":
        return _physics_candidate_params(u)
    if candidate_prior == "physics_local":
        return _physics_local_candidate_params(u)
    params = _heuristic_candidate_params(u)
    params["candidate_prior"] = "heuristic"
    params["local_anchor"] = ""
    return params


def _patch_candidate(overrides: Dict[str, Any], params: Dict[str, float], args: argparse.Namespace) -> Dict[str, Any]:
    patched = copy.deepcopy(overrides)
    sensor = patched.setdefault("sensor", {})
    lens = patched.setdefault("lens", {})
    trainable_psf = lens.setdefault("trainable_psf", {})
    data = patched.setdefault("data", {})
    train = patched.setdefault("train", {})

    sensor.update(
        {
            "raw_output": "soft_rgb",
            "sensor_model": sensor.get("sensor_model", "deeplens_nbit"),
            "exposure_init": float(params["exposure_init"]),
            "bit_depth": int(round(params["bit_depth"])),
            "read_noise_std": float(params["read_noise_std"]),
            "shot_noise_scale": float(params["shot_noise_scale"]),
            "cfa_init_floor": float(params["cfa_init_floor"]),
            "learn_cfa": True,
            "learn_exposure": True,
            "learn_noise": True,
            "learn_bit_depth": True,
        }
    )
    trainable_psf.update(
        {
            "base_sigma_px": float(params["base_sigma_px"]),
            "max_sigma_px": float(params["max_sigma_px"]),
            "max_shift_px": float(params["max_shift_px"]),
            "field_sigma": float(params["field_sigma"]),
        }
    )
    lens["learn_optics"] = True

    train["epochs"] = int(args.epochs)
    train["early_stop_patience"] = int(args.early_stop_patience)
    train["early_stop_min_delta"] = float(args.early_stop_min_delta)
    train["semantic_snr_weight"] = float(args.semantic_snr_weight)
    train["semantic_snr_nonboundary_weight"] = float(args.semantic_snr_nonboundary_weight)
    train["sensor_warmup_epochs"] = int(args.sensor_warmup_epochs)
    train["camera_curriculum_epochs"] = int(args.camera_curriculum_epochs)
    train["deployable_task_weight"] = float(args.deployable_task_weight)
    train["deployable_task_start_epoch"] = int(args.deployable_task_start_epoch)
    train["aux_losses_update_adapter"] = True
    train["freeze_backbone_during_sensor_stage"] = True
    train["sensor_freeze_after_best_patience"] = int(args.sensor_freeze_after_best_patience)
    train["sensor_freeze_restore_best"] = True
    trust = train.setdefault("sensor_trust_region", {})
    trust["revert_on_harm"] = bool(args.revert_on_harm)
    trust["revert_threshold"] = float(args.revert_threshold)
    train["deploy_guard"] = {
        "enabled": bool(args.deploy_guard),
        "start_alpha": float(args.deploy_guard_start_alpha),
        "batch_miou_ratio": float(args.deploy_guard_batch_miou_ratio),
        "val_drop_tolerance": float(args.deploy_guard_val_drop_tolerance),
        "probe_drop_tolerance": float(args.deploy_guard_probe_drop_tolerance),
        "patience": int(args.deploy_guard_patience),
        "shrink_factor": float(args.deploy_guard_shrink_factor),
        "min_lr_mult": float(args.deploy_guard_min_lr_mult),
        "freeze_after_actions": int(args.deploy_guard_freeze_after_actions),
    }
    train["progress_bar"] = True

    if args.max_train_samples > 0:
        data["max_train_samples"] = int(args.max_train_samples)
    if args.max_val_samples > 0:
        data["max_val_samples"] = int(args.max_val_samples)
    if args.img_height > 0 and args.img_width > 0:
        data["img_size"] = [int(args.img_height), int(args.img_width)]
    return patched


def build_search_matrix(args: argparse.Namespace) -> Path:
    matrix = _load_yaml(args.base_matrix)
    template = _find_template(matrix, args.template)
    template_overrides = copy.deepcopy(template.get("overrides", {}))
    prefix = args.prefix or str(template.get("name", "hybrid_codesign"))
    sobol_dim = {"heuristic": 9, "physics": 11, "physics_local": 12}[args.candidate_prior]
    draws = _sobol_candidates(args.num_candidates, args.seed, sobol_dim)

    experiments = []
    manifest_rows: List[Dict[str, Any]] = []
    for idx, u in enumerate(draws):
        params = _candidate_params(u, args.candidate_prior)
        name = _candidate_name(prefix, idx)
        overrides = _patch_candidate(template_overrides, params, args)
        experiments.append({"name": name, "overrides": overrides})
        manifest_rows.append({"candidate": idx, "experiment": name, **params})

    out_root = Path(args.out_root).expanduser().resolve()
    search_matrix = {
        "base_config": matrix["base_config"],
        "out_root": str(out_root / "candidates"),
        "seeds": [int(args.seed)],
        "experiments": experiments,
    }
    matrix_path = out_root / "candidate_matrix.yaml"
    _write_yaml(matrix_path, search_matrix)
    _write_csv(out_root / "candidate_manifest.csv", manifest_rows, manifest_rows[0].keys() if manifest_rows else [])
    return matrix_path


def _run(cmd: List[str], cwd: Path, dry_run: bool = False) -> None:
    print("[hybrid-search] " + " ".join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _rank_results(args: argparse.Namespace, matrix_path: Path) -> List[Dict[str, Any]]:
    out_root = Path(args.out_root).expanduser().resolve()
    summary_rows = _read_csv(out_root / "candidates" / "summary.csv")
    manifest = {r["experiment"]: r for r in _read_csv(out_root / "candidate_manifest.csv")}
    rank_metric = str(getattr(args, "rank_metric", "regular")).lower()
    ranked: List[Dict[str, Any]] = []
    for row in summary_rows:
        exp = row.get("experiment", "")
        params = manifest.get(exp, {})
        miou = _as_float(row.get("mIoU"), _as_float(row.get("best_val")))
        avg_miou = _as_float(row.get("avg_bestk_mIoU"), default=-1.0)
        pixel_acc = _as_float(row.get("pixel_acc"))
        avg_pixel_acc = _as_float(row.get("avg_bestk_pixel_acc"), default=-1.0)
        score_miou = avg_miou if rank_metric == "avg_bestk" and avg_miou >= 0.0 else miou
        score_acc = avg_pixel_acc if rank_metric == "avg_bestk" and avg_pixel_acc >= 0.0 else pixel_acc
        objective = score_miou + float(args.pixel_acc_weight) * score_acc
        ranked.append(
            {
                **params,
                "seed": row.get("seed", args.seed),
                "objective": objective,
                "mIoU": miou,
                "pixel_acc": pixel_acc,
                "best_val": row.get("best_val", ""),
                "ckpt_dir": row.get("ckpt_dir", ""),
                "best_ckpt": row.get("best_ckpt", ""),
                "avg_best_ckpt": row.get("avg_best_ckpt", ""),
                "design_json": row.get("design_json", ""),
                "avg_bestk_mIoU": row.get("avg_bestk_mIoU", ""),
                "avg_bestk_pixel_acc": row.get("avg_bestk_pixel_acc", ""),
                "avg_bestk_metrics": row.get("avg_bestk_metrics", ""),
                "rank_metric": rank_metric,
            }
        )
    ranked.sort(key=lambda r: _as_float(r.get("objective")), reverse=True)
    ranked_path = out_root / f"camera_search_ranked_{rank_metric}.csv"
    _write_csv(ranked_path, ranked, SEARCH_FIELDS)
    _write_csv(out_root / "camera_search_ranked.csv", ranked, SEARCH_FIELDS)
    if ranked:
        best = ranked[0]
        best_payload = {
            "source_matrix": str(matrix_path),
            "ranked_csv": str(ranked_path),
            "canonical_ranked_csv": str(out_root / "camera_search_ranked.csv"),
            "best": best,
        }
        _write_yaml(out_root / "best_candidate.yaml", best_payload)
        design_json = str(best.get("design_json", "") or "")
        if design_json and Path(design_json).is_file():
            shutil.copy2(design_json, out_root / "best_camera_design.json")
    return ranked


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-matrix", default=DEFAULT_KITTI_MATRIX)
    parser.add_argument("--template", default="", help="Template co-design experiment name. Default: first constrained co-design row.")
    parser.add_argument("--prefix", default="", help="Candidate experiment prefix. Default: template name.")
    parser.add_argument("--out-root", default="runs/hybrid_camera_search_fast/kitti360")
    parser.add_argument("--num-candidates", type=int, default=8)
    parser.add_argument(
        "--candidate-prior",
        choices=["physics", "physics_local", "heuristic"],
        default="physics",
        help=(
            "Candidate generator. 'physics' samples camera latents such as pixel pitch, "
            "f-number, exposure, QE, full well, and read noise, then derives simulator knobs. "
            "'physics_local' samples near the best broad-search physics candidates. "
            "'heuristic' reproduces the older direct simulator-knob Sobol search."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--early-stop-patience", type=int, default=2)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.001)
    parser.add_argument("--max-train-samples", type=int, default=2500)
    parser.add_argument("--max-val-samples", type=int, default=600)
    parser.add_argument("--img-height", type=int, default=0)
    parser.add_argument("--img-width", type=int, default=0)
    parser.add_argument("--semantic-snr-weight", type=float, default=0.04)
    parser.add_argument("--semantic-snr-nonboundary-weight", type=float, default=0.20)
    parser.add_argument("--sensor-warmup-epochs", type=int, default=2)
    parser.add_argument("--camera-curriculum-epochs", type=int, default=4)
    parser.add_argument("--deployable-task-weight", type=float, default=0.12)
    parser.add_argument("--deployable-task-start-epoch", type=int, default=3)
    parser.add_argument("--sensor-freeze-after-best-patience", type=int, default=2)
    parser.add_argument("--revert-on-harm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--revert-threshold", type=float, default=0.012)
    parser.add_argument("--deploy-guard", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--deploy-guard-start-alpha", type=float, default=0.999)
    parser.add_argument("--deploy-guard-batch-miou-ratio", type=float, default=0.45)
    parser.add_argument("--deploy-guard-val-drop-tolerance", type=float, default=0.008)
    parser.add_argument("--deploy-guard-probe-drop-tolerance", type=float, default=0.08)
    parser.add_argument("--deploy-guard-patience", type=int, default=2)
    parser.add_argument("--deploy-guard-shrink-factor", type=float, default=0.35)
    parser.add_argument("--deploy-guard-min-lr-mult", type=float, default=0.01)
    parser.add_argument("--deploy-guard-freeze-after-actions", type=int, default=3)
    parser.add_argument("--pixel-acc-weight", type=float, default=0.0)
    parser.add_argument(
        "--rank-metric",
        choices=["regular", "avg_bestk"],
        default="regular",
        help="Metric used for camera_search_ranked.csv objective. Both regular and averaged metrics are exported.",
    )
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--max-parallel", type=int, default=1)
    parser.add_argument("--batch-plan", default="")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--rank-only", action="store_true", help="Regenerate ranked CSV/YAML from existing candidate outputs.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    out_root = Path(args.out_root).expanduser().resolve()
    if args.fresh and out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.rank_only:
        matrix_path = out_root / "candidate_matrix.yaml"
        ranked = _rank_results(args, matrix_path)
        if ranked:
            best = ranked[0]
            print(
                "[hybrid-search] best "
                f"{best.get('experiment')} mIoU={_as_float(best.get('mIoU')):.4f} "
                f"avg_bestk_mIoU={_as_float(best.get('avg_bestk_mIoU')):.4f} "
                f"objective={_as_float(best.get('objective')):.4f}",
                flush=True,
            )
        else:
            print("[hybrid-search] no completed candidate rows found yet", flush=True)
        return

    matrix_path = build_search_matrix(args)
    print(f"[hybrid-search] candidate matrix: {matrix_path}", flush=True)
    if args.generate_only:
        return

    cmd = [
        args.python,
        "-u",
        "-m",
        "deeplens.projects.raw2task.run_paper_experiments",
        "--matrix",
        str(matrix_path),
        "--seeds",
        str(args.seed),
        "--gpus",
        args.gpus,
        "--max-parallel",
        str(max(1, int(args.max_parallel))),
        "--order-policy",
        "matrix",
        "--skip-existing",
        "--robustness-experiments",
        "",
    ]
    if args.batch_plan:
        cmd.extend(["--batch-plan", args.batch_plan])
    if args.dry_run:
        cmd.append("--dry-run")
    _run(cmd, cwd=repo_root, dry_run=False)
    if not args.dry_run:
        ranked = _rank_results(args, matrix_path)
        if ranked:
            best = ranked[0]
            print(
                "[hybrid-search] best "
                f"{best.get('experiment')} mIoU={_as_float(best.get('mIoU')):.4f} "
                f"avg_bestk_mIoU={_as_float(best.get('avg_bestk_mIoU')):.4f} "
                f"objective={_as_float(best.get('objective')):.4f}",
                flush=True,
            )
        else:
            print("[hybrid-search] no completed candidate rows found yet", flush=True)


if __name__ == "__main__":
    main()
