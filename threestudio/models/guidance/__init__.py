from . import (
    stable_diffusion_guidance,
    stable_diffusion_ot_guidance,
    stable_diffusion_ot_sdi_guidance,
    stable_diffusion_ot_vsd_guidance,
    stable_diffusion_vsd_guidance,
)

import importlib, warnings

_optional_guidance = [
    "controlnet_guidance",
    "deep_floyd_guidance",
    "instructpix2pix_guidance",
    "stable_diffusion_sdi_guidance",
    "stable_diffusion_mvsd_guidance",
    "stable_diffusion_unified_guidance",
    "stable_zero123_guidance",
    "zero123_guidance",
    "zero123_unified_guidance",
    "flux_sdi_guidance",
]
for _mod in _optional_guidance:
    try:
        importlib.import_module(f".{_mod}", __name__)
    except Exception as e:
        warnings.warn(f"Skipping guidance module {_mod}: {e}")
