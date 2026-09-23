from enum import Enum
from pydantic import BaseModel, Field


class ProviderName(str, Enum):
    qwen = "qwen"
    flux = "flux"


class EditMode(str, Enum):
    auto = "auto"
    local = "local"
    global_ = "global"


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


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=4000)
    width: int = Field(default=1024, ge=256, le=2048)
    height: int = Field(default=1024, ge=256, le=2048)
    seed: int | None = None
