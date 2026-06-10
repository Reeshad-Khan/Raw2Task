#!/usr/bin/env python3
import argparse, json, math, os, sys

def compute_f_from_hfov(sensor_w_m: float, hfov_deg: float) -> float:
    hfov = math.radians(hfov_deg)
    return sensor_w_m / (2.0 * math.tan(0.5 * hfov))

def set_if(d, key, val):
    if key in d:
        d[key] = val
        return True
    return False

def set_first(d, keys, val):
    for k in keys:
        if k in d:
            d[k] = val
            return k
    return None

def set_inplace(obj, keys, val):
    if isinstance(obj, dict):
        for k in keys:
            if k in obj:
                obj[k] = val
                return k
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", required=True, help="Path to a WORKING GeoLens JSON (e.g., epoch11.json)")
    ap.add_argument("--out",       required=True, help="Output JSON path (will be overwritten)")
    ap.add_argument("--H", type=int, default=384)
    ap.add_argument("--W", type=int, default=1280)
    ap.add_argument("--pixel_pitch_m", type=float, default=1.4e-6)
    ap.add_argument("--hfov_deg", type=float, default=80.0)
    ap.add_argument("--fnum", type=float, default=2.8)
    args = ap.parse_args()

    with open(args.template, "r") as f:
        data = json.load(f)

    # ---- Update sensor resolution ----
    # common places: top-level "sensor_res": [H, W] or nested "sensor": {"H":..., "W":...}
    updated = False
    if isinstance(data, dict):
        if set_if(data, "sensor_res", [args.H, args.W]):
            updated = True
        if "sensor" in data and isinstance(data["sensor"], dict):
            s = data["sensor"]
            set_if(s, "H", args.H); set_if(s, "W", args.W)
            # some schemas store as dict: {"res": [H, W]}
            set_if(s, "res", [args.H, args.W])
            updated = True or updated

    # ---- Update pixel size ----
    # common keys: "pixel", "pixsize", "pixel_size"
    set_first(data, ["pixel","pixsize","pixel_size","pixel_size_m"], float(args.pixel_pitch_m))

    # ---- Compute thin-lens focal length from HFOV ----
    f_m = compute_f_from_hfov(args.W * args.pixel_pitch_m, args.hfov_deg)
    # store where possible
    set_first(data, ["focal_length","foclen","f_mm"], f_m)

    # ---- Aperture radius from f-number ----
    aper_radius_m = 0.5 * (f_m / args.fnum)

    # Try to update a surface that looks like an aperture
    def update_aper_in_surfaces(surfaces):
        hit = False
        for s in surfaces:
            t = str(s.get("type", "")).lower()
            if "aper" in t or "aperture" in t:
                # common radius field names
                if set_inplace(s, ["r","radius","aper_r","aperture_radius"], aper_radius_m):
                    hit = True
                # some store diameter; if you see "d" or "diameter"
                set_inplace(s, ["d","diameter","aper_d","aperture_diameter"], 2.0*aper_radius_m)
        return hit

    if "surfaces" in data and isinstance(data["surfaces"], list):
        _ = update_aper_in_surfaces(data["surfaces"])
    else:
        # Some schemas nest surfaces
        for k in list(data.keys()):
            if isinstance(data.get(k), dict) and isinstance(data[k].get("surfaces"), list):
                _ = update_aper_in_surfaces(data[k]["surfaces"])

    # Also try top-level aperture fields if present
    set_first(data, ["aperture_radius","aper_r"], aper_radius_m)
    set_first(data, ["aperture_diameter","aper_d"], 2.0*aper_radius_m)

    # Optionally record what we changed (non-critical)
    data.setdefault("_kitti360_override", {})
    data["_kitti360_override"].update(dict(
        sensor_res=[args.H, args.W],
        pixel_pitch_m=args.pixel_pitch_m,
        hfov_deg=args.hfov_deg,
        fnum=args.fnum,
        focal_length_m=f_m,
        aperture_radius_m=aper_radius_m,
    ))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved cloned lens JSON -> {args.out}")

if __name__ == "__main__":
    main()