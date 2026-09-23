from __future__ import annotations

import threading

from .base import ImageProvider
from .flux import FluxKleinProvider
from .qwen import QwenEditProvider


class ProviderManager:
    """Only one heavyweight provider is kept loaded at a time; GPU work is serialized."""

    def __init__(self) -> None:
        self._providers = {
            "qwen": QwenEditProvider(),
            "flux": FluxKleinProvider(),
        }
        self._active: str | None = None
        self.gpu_lock = threading.Lock()

    def get(self, name: str) -> ImageProvider:
        if name not in self._providers:
            raise ValueError(f"Unknown provider: {name}")
        if self._active and self._active != name:
            self._providers[self._active].unload()
        self._active = name
        return self._providers[name]


provider_manager = ProviderManager()
