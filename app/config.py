from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    host: str = "127.0.0.1"
    port: int = 8000
    data_dir: Path = Path("./data")
    max_upload_mb: int = 40
    max_image_side: int = 2048

    default_provider: Literal["qwen", "flux"] = "qwen"

    qwen_model_id: str = "Qwen/Qwen-Image-Edit-2511"
    qwen_quantization: Literal["none", "nf4", "int8"] = "nf4"
    qwen_offload: Literal["none", "model", "sequential"] = "none"
    qwen_steps: int = 40
    qwen_true_cfg_scale: float = 4.0

    flux_model_id: str = "black-forest-labs/FLUX.2-klein-4B"
    flux_quantization: Literal["none", "nf4", "int8"] = "none"
    flux_offload: Literal["none", "model", "sequential"] = "none"
    flux_steps: int = 4

    local_padding: int = 128
    mask_feather: int = 10
    local_min_side: int = 512
    local_max_side: int = 1536

    hf_token: str | None = None


settings = Settings()
settings.data_dir.mkdir(parents=True, exist_ok=True)
