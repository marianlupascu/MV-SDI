from . import (
    base,
    stable_diffusion_prompt_processor,
)

import importlib, warnings

_optional_pp = [
    "deepfloyd_prompt_processor",
    "dummy_prompt_processor",
    "flux_prompt_processor",
]
for _mod in _optional_pp:
    try:
        importlib.import_module(f".{_mod}", __name__)
    except Exception as e:
        warnings.warn(f"Skipping prompt processor {_mod}: {e}")
