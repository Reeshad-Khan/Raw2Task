from pathlib import Path
import sys
import torch
import torch.nn as nn

# ensure repo root on path
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

def _import_geolens_or_die():
    try:
        import deeplens.geolens as geolens
        return geolens
    except Exception as e:
        raise ImportError(
            "[DeepLens adapter] Failed to import deeplens.geolens. "
            "Run from the repo root and set PYTHONPATH.\n"
            f"Original error: {e}"
        )

def _call_render(lens, rgb, depth_m=None):
    """
    rgb: (B,3,H,W) in [0,1]
    We set lens.sensor_res to (H,W) if supported, then call render.
    Fallback: loop over batch if vectorized render isn't supported.
    """
    B, C, H, W = rgb.shape

    # Set sensor resolution on lens if possible
    if hasattr(lens, "set_sensor_res"):
        try:
            lens.set_sensor_res((H, W))
        except Exception:
            pass
    elif hasattr(lens, "sensor_res"):
        try:
            lens.sensor_res = (H, W)
        except Exception:
            pass

    # Optional: set device/pixel size aliases if present
    if hasattr(lens, "device"):
        try:
            lens.device = rgb.device
        except Exception:
            pass

    # Try vectorized render (B,3,H,W) -> (B,3,H,W)
    if hasattr(lens, "render"):
        try:
            return torch.clamp(lens.render(rgb, depth=depth_m), 0.0, 1.0)
        except TypeError:
            return torch.clamp(lens.render(rgb), 0.0, 1.0)
        except Exception:
            pass  # fall back to per-image loop

    # Fallbacks
    outs = []
    for i in range(B):
        x = rgb[i:i+1]
        if hasattr(lens, "render"):
            try:
                y = lens.render(x, depth=depth_m)
            except TypeError:
                y = lens.render(x)
        elif callable(lens):
            y = lens(x)
        else:
            raise RuntimeError("[DeepLens adapter] Lens has no supported render(...)")
        outs.append(y)
    return torch.clamp(torch.cat(outs, dim=0), 0.0, 1.0)


class DeepLensPSF(nn.Module):
    """
    Thin-lens integration:
      - Uses deeplens.geolens.Lens (NOT GeoLens)
      - Sets common attributes: foclen, fnum, aperture, pixel/pixsize
      - Renders with focus distance as 'depth'
    Trainable: focal_length, focus_distance, (optionally) aperture_diameter
    """
    def __init__(self, cfg):
        super().__init__()
        geolens = _import_geolens_or_die()
        if not hasattr(geolens, "Lens"):
            raise RuntimeError("[DeepLens adapter] geolens.Lens not found; update adapter to your version.")

        self._Lens = geolens.Lens

        dlcfg = cfg['sensor'].get('deeplens', {})
        # meters
        aperture_diameter = float(dlcfg.get('aperture_diameter', 4.0e-3))
        focal_length      = float(dlcfg.get('focal_length',      5.0e-3))
        focus_distance    = float(dlcfg.get('focus_distance',    2.0))
        sensor_pitch      = float(dlcfg.get('sensor_pitch',      1.4e-6))
        wavelengths_nm    = dlcfg.get('wavelengths_nm',          [460,540,620])  # unused by thin lens; kept for future

        train_flags = dlcfg.get('trainable', {})
        self.focal_length   = nn.Parameter(torch.tensor(focal_length),      requires_grad=bool(train_flags.get('focal_length', True)))
        self.focus_distance = nn.Parameter(torch.tensor(focus_distance),    requires_grad=bool(train_flags.get('focus_distance', True)))
        self.aperture_diam  = nn.Parameter(torch.tensor(aperture_diameter), requires_grad=bool(train_flags.get('aperture_diameter', False)))

        self.register_buffer('sensor_pitch',   torch.tensor(sensor_pitch))
        self.register_buffer('wavelengths_nm', torch.tensor(wavelengths_nm, dtype=torch.float32))

    def _build_lens(self):
        # Instantiate thin lens and set fields it uses
        lens = self._Lens()  # most versions allow empty init

        foclen   = float(self.focal_length.item())         # [m]
        aperture = float(self.aperture_diam.item())        # [m]
        fnum     = max(foclen / max(aperture, 1e-9), 1.4)  # sane clamp
        pix      = float(self.sensor_pitch.item())         # [m]

        # Set multiple aliases to be safe across versions
        for k, v in {
            "foclen": foclen,          # effective focal length
            "fnum": fnum,              # f-number
            "aperture": aperture,      # aperture diameter
            "pixel": pix,              # pixel pitch
            "pixsize": pix,            # alias
            "pixel_size": pix,         # alias
        }.items():
            try:
                setattr(lens, k, v)
            except Exception:
                pass
        return lens

    def forward(self, rgb):  # rgb: (B,3,H,W) in [0,1]
        lens = self._build_lens()
        depth_m = float(self.focus_distance.item())  # focus distance passed as depth
        return _call_render(lens, rgb, depth_m)
