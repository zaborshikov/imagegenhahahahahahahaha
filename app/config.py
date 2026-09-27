from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

Quantization = Literal["none", "fp8", "nf4", "int8"]
Offload = Literal["none", "model", "sequential"]
SaveGpu = Literal["off", "ram", "disk"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    host: str = "127.0.0.1"
    port: int = 8000
    data_dir: Path = Path("./data")
    max_upload_mb: int = 40
    max_image_side: int = 2048

    # Default editing backend and default text-to-image backend.
    default_provider: Literal["qwen21", "flux"] = "qwen21"
    default_generate_provider: Literal["qwen21", "flux"] = "qwen21"

    # Default GPU-memory strategy. Can be overridden per request from the UI.
    #   off  - whole pipeline resident on the GPU (fastest).
    #   ram  - components are moved GPU<->CPU one at a time (diffusers model offload).
    #   disk - staged: text encoder runs and is fully unloaded, intermediate tensors go to disk,
    #          then transformer+VAE are loaded. Lowest peak VRAM, slowest.
    default_save_gpu: SaveGpu = "off"

    # Qwen-Image-2.1: unified text-to-image + editing (7B DiT + Qwen3-VL 8B text encoder, Qwen Research License).
    qwen21_model_id: str = "Qwen/Qwen-Image-2.1"
    qwen21_quantization: Quantization = "fp8"
    qwen21_offload: Offload = "none"
    qwen21_steps: int = 40
    qwen21_min_steps: int = 4
    qwen21_max_steps: int = 60
    # The model is meant to be sampled without guidance (true_cfg_scale=1.0 disables CFG).
    qwen21_true_cfg_scale: float = 1.0
    # Long side used for editing outputs; the pipeline derives height/width from the condition image.
    qwen21_edit_resolution: int = 1024

    # FLUX.2 klein 4B: distilled text-to-image + editing (Apache-2.0).
    flux_model_id: str = "black-forest-labs/FLUX.2-klein-4B"
    flux_quantization: Quantization = "fp8"
    flux_offload: Offload = "none"
    flux_steps: int = 4
    flux_min_steps: int = 1
    flux_max_steps: int = 12

    local_padding: int = 128
    mask_feather: int = 10
    local_min_side: int = 512
    local_max_side: int = 1536

    hf_token: str | None = None


settings = Settings()
settings.data_dir.mkdir(parents=True, exist_ok=True)
