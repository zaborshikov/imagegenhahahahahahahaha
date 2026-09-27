from enum import Enum
from pydantic import BaseModel, Field


class ProviderName(str, Enum):
    qwen21 = "qwen21"
    flux = "flux"


class EditMode(str, Enum):
    auto = "auto"
    local = "local"
    global_ = "global"


class SaveGpuMode(str, Enum):
    off = "off"
    ram = "ram"
    disk = "disk"


class ChatMessage(BaseModel):
    role: str
    content: str


class SessionCreateResponse(BaseModel):
    session_id: str


class AssetResponse(BaseModel):
    image_url: str
    width: int
    height: int


class JobResponse(BaseModel):
    job_id: str
    status: str


class JobState(BaseModel):
    job_id: str
    status: str
    image_url: str | None = None
    error: str | None = None
    provider: str | None = None
    mode: str | None = None
    seed: int | None = None
    steps: int | None = None
    save_gpu: str | None = None
    # Human-readable progress ("Encoding prompt", "Denoising 12/40", ...).
    stage: str | None = None


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=4000)
    width: int = Field(default=1024, ge=256, le=2048)
    height: int = Field(default=1024, ge=256, le=2048)
    seed: int | None = None
    # None -> settings.default_generate_provider
    provider: str | None = None
    # None -> provider default
    steps: int | None = Field(default=None, ge=1, le=200)
    # None -> settings.default_save_gpu
    save_gpu: SaveGpuMode | None = None
