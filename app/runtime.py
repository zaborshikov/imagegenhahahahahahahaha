from __future__ import annotations

import torch


def cuda_info() -> dict:
    if not torch.cuda.is_available():
        return {
            "available": False,
            "name": None,
            "total_vram_gb": 0.0,
            "compute_capability": None,
            "preferred_dtype": "float32",
        }
    props = torch.cuda.get_device_properties(0)
    major, minor = torch.cuda.get_device_capability(0)
    # Native BF16 tensor-core support starts with Ampere (SM80).
    dtype = "bfloat16" if major >= 8 else "float16"
    return {
        "available": True,
        "name": props.name,
        "total_vram_gb": round(props.total_memory / (1024**3), 2),
        "compute_capability": f"{major}.{minor}",
        "preferred_dtype": dtype,
    }


def preferred_dtype() -> torch.dtype:
    if not torch.cuda.is_available():
        return torch.float32
    major, _ = torch.cuda.get_device_capability(0)
    return torch.bfloat16 if major >= 8 else torch.float16


def generator_for(seed: int | None) -> torch.Generator | None:
    if seed is None:
        return None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.Generator(device=device).manual_seed(seed)
