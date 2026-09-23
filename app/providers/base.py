from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from PIL import Image


@dataclass
class EditInput:
    image: Image.Image
    prompt: str
    references: list[Image.Image] = field(default_factory=list)
    sketch_reference: Image.Image | None = None
    seed: int | None = None


class ImageProvider(ABC):
    name: str

    @abstractmethod
    def edit(self, req: EditInput) -> Image.Image:
        raise NotImplementedError

    def generate(self, prompt: str, width: int, height: int, seed: int | None = None) -> Image.Image:
        raise NotImplementedError(f"{self.name} does not implement text-to-image")

    def unload(self) -> None:
        pass
