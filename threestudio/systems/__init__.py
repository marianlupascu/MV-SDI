from . import (
    dreamfusion,
    prolificdreamer,
)

import importlib, warnings

_optional_systems = [
    "control4d_multiview",
    "eff_dreamfusion",
    "fantasia3d",
    "imagedreamfusion",
    "instructnerf2nerf",
    "latentnerf",
    "magic3d",
    "magic123",
    "sdi",
    "mvsd",
    "sjc",
    "textmesh",
    "zero123",
    "zero123_simple",
]
for _mod in _optional_systems:
    try:
        importlib.import_module(f".{_mod}", __name__)
    except Exception as e:
        warnings.warn(f"Skipping system module {_mod}: {e}")
