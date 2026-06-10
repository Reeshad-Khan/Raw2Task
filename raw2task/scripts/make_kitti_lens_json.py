#!/usr/bin/env python3
# Robust GeoLens JSON maker for KITTI-360 geometry

import argparse, os, math, json, logging, torch

def compute_f_from_hfov(sensor_w_m: float, hfov_deg: float) -> float:
    hfov = math.radians(hfov_deg)
    return sensor_w_m / (2.0 * math.tan(hfov * 0.5))  # thin-lens approx

def set_if_has(obj, name, value):
    if hasattr(obj, name):
        try:
            setattr(obj, name, value)
            return True
        except Exception:
            pass
    return False

def maybe_set_sensor(lens, H, W, pixel_pitch_m):
    # handle different DeepLens attribute names
    if hasattr(lens, "set_sensor_res"):
        try: lens.set_sensor_res((H, W))
        except Exception: pass
    set_if_has(lens, "sensor_res", (H, W))
    for nm in ("pixel", "pixsize", "pixel_size"):
        if set_if_has(lens, nm, float(pixel_pitch_m)):
            break

def maybe_set_aperture(lens, aper_radius_m):
    # best-effort set aperture on a surface
    try:
        surfs = getattr(lens, "surfaces", None)
        if surfs:
            for s in surfs:
                tname = getattr(s, "type", s.__class__.__name__).lower()
                if "aper" in tname:
                    if hasattr(s, "update_r"):
                        try:
                            s.update_r(float(aper_radius_m)); return True
                        except Exception: pass
                    for nm in ("r", "radius", "aper_r", "aperture_radius"):
                        if set_if_has(s, nm, float(aper_radius_m)):
                            return True
        # or directly on lens
        for nm in ("aper_r", "aperture_radius", "aperture_r"):
            if set_if_has(lens, nm, float(aper_radius_m)):
                return True
    except Exception:
        pass
    return False

def robust_set_fov_fnum(lens, hfov_deg, fnum, W, pixel_pitch_m):
    """
    Try GeoLens.set_target_fov_fnum; if it fails, fall back to thin-lens f and aperture.
    """
    # attempt native helper
    try:
        lens.set_target_fov_fnum(hfov=hfov_deg, fnum=fnum)
        return True, None
    except Exception as e:
        # fallback path
        sensor_w_m = W * pixel_pitch_m
        f_m = compute_f_from_hfov(sensor_w_m, hfov_deg)
        aper_d_m = f_m / fnum
        ok_f = set_if_has(lens, "foclen", float(f_m)) or set_if_has(lens, "focal_length", float(f_m))
        ok_a = maybe_set_aperture(lens, aper_d_m * 0.5)
        return False, dict(f_m=f_m, aper_d_m=aper_d_m, ok_f=ok_f, ok_a=ok_a, err=str(e))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--hfov_deg", type=float, default=80.0)
    ap.add_argument("--fnum", type=float, default=2.8)
    ap.add_argument("--H", type=int, default=384)
    ap.add_argument("--W", type=int, default=1280)
    ap.add_argument("--pixel_pitch_m", type=float, default=1.4e-6)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if torch.cuda.is_available():
        logging.info(f"Using CUDA: {torch.cuda.get_device_name(0)}")

    from deeplens import GeoLens
    lens = GeoLens()

    # match the reference flow: set sensor first
    maybe_set_sensor(lens, args.H, args.W, args.pixel_pitch_m)

    ok, info = robust_set_fov_fnum(lens, args.hfov_deg, args.fnum, args.W, args.pixel_pitch_m)
    if ok:
        logging.info(f"set_target_fov_fnum OK (HFOV={args.hfov_deg}°, f/{args.fnum}).")
    else:
        logging.warning("set_target_fov_fnum not available on this build; used thin-lens fallback.")
        logging.info(f"Fallback f={info['f_m']*1e3:.2f} mm, aperture={info['aper_d_m']*1e3:.2f} mm "
                     f"(set_f={info['ok_f']}, set_aper={info['ok_a']}). Reason: {info['err']}")

    out_path = os.path.abspath(os.path.expanduser(args.out))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    try:
        lens.write_lens_json(out_path)
        logging.info(f"Saved lens JSON to {out_path}")
    except Exception as e:
        # final safety: minimal JSON
        logging.warning(f"lens.write_lens_json failed ({e}); writing minimal JSON.")
        minimal = {}
        for k in ("sensor_res", "pixel", "pixsize", "pixel_size", "foclen", "focal_length"):
            if hasattr(lens, k):
                v = getattr(lens, k)
                try:
                    minimal[k] = float(v) if isinstance(v, (int, float)) else v
                except Exception:
                    pass
        with open(out_path, "w") as f:
            json.dump(minimal, f, indent=2)
        logging.info(f"Wrote minimal JSON to {out_path}")

if __name__ == "__main__":
    main()