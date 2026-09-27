from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable

from PIL import Image

from ..config import SaveGpu

ProgressFn = Callable[[str], None]


@dataclass
class RunOptions:
    """Per-request runtime knobs chosen in the UI."""

    steps: int | None = None
    save_gpu: SaveGpu = "off"
    progress: ProgressFn | None = None

    def report(self, stage: str) -> None:
        if self.progress:
            self.progress(stage)


@dataclass
class EditInput:
    image: Image.Image
    prompt: str
    references: list[Image.Image] = field(default_factory=list)
    sketch_reference: Image.Image | None = None
    seed: int | None = None
    options: RunOptions = field(default_factory=RunOptions)


class ImageProvider(ABC):
    name: str
    # Whether generate() (text-to-image) is implemented.
    supports_generate: bool = False

    @abstractmethod
    def edit(self, req: EditInput) -> Image.Image:
        raise NotImplementedError

    def generate(
        self, prompt: str, width: int, height: int, seed: int | None = None, options: RunOptions | None = None
    ) -> Image.Image:
        raise NotImplementedError(f"{self.name} does not implement text-to-image")

    def unload(self) -> None:
        pass
