from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageFilter, ImageOps


@dataclass(frozen=True)
class Box:
    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top

    def pil(self) -> tuple[int, int, int, int]:
        return self.left, self.top, self.right, self.bottom


def normalize_rgb(image: Image.Image, max_side: int | None = None) -> Image.Image:
    image = ImageOps.exif_transpose(image).convert("RGB")
    if max_side and max(image.size) > max_side:
        scale = max_side / max(image.size)
        image = image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS)
    return image


def normalize_mask(mask: Image.Image, size: tuple[int, int]) -> Image.Image:
    mask = ImageOps.exif_transpose(mask).convert("L")
    if mask.size != size:
        mask = mask.resize(size, Image.Resampling.NEAREST)
    # Browser sends transparent/black background + white paint. Threshold tiny compression noise.
    return mask.point(lambda p: 255 if p >= 8 else 0)


def nonempty_mask(mask: Image.Image) -> bool:
    return mask.getbbox() is not None


def bbox_from_mask(mask: Image.Image) -> Box | None:
    bbox = mask.getbbox()
    return Box(*bbox) if bbox else None


def expand_box(box: Box, padding: int, image_size: tuple[int, int]) -> Box:
    w, h = image_size
    return Box(
        max(0, box.left - padding),
        max(0, box.top - padding),
        min(w, box.right + padding),
        min(h, box.bottom + padding),
    )


def ensure_min_crop(box: Box, image_size: tuple[int, int], min_side: int) -> Box:
    """Expand a crop around its center so tiny masks still provide scene context."""
    w, h = image_size
    target_w = min(w, max(box.width, min_side))
    target_h = min(h, max(box.height, min_side))
    cx = (box.left + box.right) / 2
    cy = (box.top + box.bottom) / 2
    left = int(round(cx - target_w / 2))
    top = int(round(cy - target_h / 2))
    left = max(0, min(left, w - target_w))
    top = max(0, min(top, h - target_h))
    return Box(left, top, left + target_w, top + target_h)


def fit_for_model(image: Image.Image, max_side: int, multiple: int = 16) -> tuple[Image.Image, tuple[int, int]]:
    """Resize down only; make dimensions model-friendly. Returns image and original size."""
    original = image.size
    scale = min(1.0, max_side / max(image.size))
    w = max(multiple, int(round(image.width * scale / multiple) * multiple))
    h = max(multiple, int(round(image.height * scale / multiple) * multiple))
    return image.resize((w, h), Image.Resampling.LANCZOS), original


def restore_size(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    return image.convert("RGB").resize(size, Image.Resampling.LANCZOS) if image.size != size else image.convert("RGB")


def feather_mask(mask: Image.Image, radius: int) -> Image.Image:
    mask = mask.convert("L")
    if radius <= 0:
        return mask
    return mask.filter(ImageFilter.GaussianBlur(radius=radius))


def composite_local(base: Image.Image, edited_crop: Image.Image, mask: Image.Image, box: Box, feather: int) -> Image.Image:
    base = base.convert("RGB").copy()
    old_crop = base.crop(box.pil())
    edited_crop = restore_size(edited_crop, old_crop.size)
    crop_mask = mask.crop(box.pil())
    soft = feather_mask(crop_mask, feather)
    merged = Image.composite(edited_crop, old_crop, soft)
    base.paste(merged, (box.left, box.top))
    return base


def make_sketch_reference(base: Image.Image, sketch: Image.Image | None) -> Image.Image | None:
    if sketch is None:
        return None
    sketch = sketch.convert("RGBA")
    if sketch.size != base.size:
        sketch = sketch.resize(base.size, Image.Resampling.NEAREST)
    canvas = base.convert("RGBA")
    # Preserve only explicitly drawn pixels from the sketch layer.
    canvas.alpha_composite(sketch)
    return canvas.convert("RGB")


def save_png(image: Image.Image, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", optimize=True)
    return path
