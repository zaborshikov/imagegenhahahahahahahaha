from __future__ import annotations

import threading
from dataclasses import dataclass

from .base import ImageProvider
from .diffusers_base import DiffusersProvider
from .flux import FluxKleinProvider
from .qwen_image21 import QwenImage21Provider


@dataclass(frozen=True)
class ProviderInfo:
    """Static metadata exposed to the API/UI; does not touch model weights."""

    name: str
    label: str
    model_id: str
    quantization: str
    license: str
    supports_edit: bool
    supports_generate: bool
    default_steps: int
    min_steps: int
    max_steps: int


_LABELS = {
    "qwen21": ("Qwen Image 2.1", "Qwen Research License"),
    "flux": ("FLUX.2 klein 4B", "Apache-2.0"),
}


class ProviderManager:
    """Only one heavyweight provider is kept loaded at a time; GPU work is serialized."""

    def __init__(self) -> None:
        providers: list[DiffusersProvider] = [QwenImage21Provider(), FluxKleinProvider()]
        self._providers: dict[str, ImageProvider] = {p.name: p for p in providers}
        self._info: dict[str, ProviderInfo] = {}
        for p in providers:
            label, license_ = _LABELS[p.name]
            self._info[p.name] = ProviderInfo(
                name=p.name,
                label=label,
                model_id=p.model_id,
                quantization=p.quantization,
                license=license_,
                supports_edit=True,
                supports_generate=p.supports_generate,
                default_steps=p.default_steps,
                min_steps=p.min_steps,
                max_steps=p.max_steps,
            )
        self._active: str | None = None
        self.gpu_lock = threading.Lock()

    # -- metadata -------------------------------------------------------------------------------

    @property
    def names(self) -> list[str]:
        return list(self._providers)

    @property
    def edit_providers(self) -> list[str]:
        return [n for n, i in self._info.items() if i.supports_edit]

    @property
    def generate_providers(self) -> list[str]:
        return [n for n, i in self._info.items() if i.supports_generate]

    def info(self, name: str) -> ProviderInfo:
        return self._info[name]

    def describe(self) -> list[ProviderInfo]:
        return list(self._info.values())

    def validate(self, name: str, *, for_generate: bool = False) -> str:
        if name not in self._providers:
            raise ValueError(f"provider must be one of: {', '.join(self.names)}")
        if for_generate and not self._info[name].supports_generate:
            raise ValueError(
                f"provider '{name}' cannot generate from scratch; use one of: {', '.join(self.generate_providers)}"
            )
        return name

    # -- runtime --------------------------------------------------------------------------------

    def get(self, name: str) -> ImageProvider:
        if name not in self._providers:
            raise ValueError(f"Unknown provider: {name}")
        if self._active and self._active != name:
            self._providers[self._active].unload()
        self._active = name
        return self._providers[name]


provider_manager = ProviderManager()
