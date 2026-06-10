"""Public DeepLens package surface.

The research project contains several optional stacks (reconstruction networks,
surrogate models, differentiable optics, and task-driven sensing).  Importing
``raw2task`` should not require every optional dependency used
by unrelated modules, so package-level imports are intentionally best-effort.
"""

def _optional_star_import(module_name: str):
    try:
        module = __import__(module_name, globals(), locals(), ["*"], 1)
    except Exception:
        return
    names = getattr(module, "__all__", None)
    if names is None:
        names = [k for k in module.__dict__ if not k.startswith("_")]
    globals().update({k: getattr(module, k) for k in names})


_optional_star_import("optics")
_optional_star_import("network")
_optional_star_import("geolens_pkg")
_optional_star_import("utils")

try:
    from .geolens import GeoLens
except Exception:
    GeoLens = None

try:
    from .psfnetlens import PSFNetLens
except Exception:
    PSFNetLens = None

try:
    from .camera import Camera
except Exception:
    Camera = None
