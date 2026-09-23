from __future__ import annotations

import random
from dataclasses import dataclass
from PIL import Image

from .config import settings
from .image_ops import (
    bbox_from_mask,
    composite_local,
    ensure_min_crop,
    expand_box,
    fit_for_model,
    make_sketch_reference,
    normalize_mask,
    restore_size,
)
from .providers.base import EditInput
from .providers.manager import provider_manager


@dataclass
class EditResult:
    image: Image.Image
    provider: str
    mode: str
    seed: int


def history_context(messages: list[dict[str, str]], latest: str) -> str:
    """Compact chat context. Avoids a second LLM while preserving multi-turn intent."""
    prior = [m for m in messages[-8:] if m.get("role") == "user" and m.get("content") != latest]
    if not prior:
        return latest
    summary = " | ".join(m["content"][:350] for m in prior[-4:])
    return (
        f"Current edit request: {latest}\n"
        f"Relevant previous user instructions in this editing conversation: {summary}\n"
        "Treat previous instructions as context only. Preserve the current input image as the source of truth and obey the current request first."
    )


def run_edit(
    *,
    base: Image.Image,
    prompt: str,
    provider_name: str,
    mode: str,
    strict_local: bool,
    history: list[dict[str, str]],
    mask: Image.Image | None = None,
    sketch: Image.Image | None = None,
    references: list[Image.Image] | None = None,
    seed: int | None = None,
) -> EditResult:
    references = references or []
    seed = seed if seed is not None else random.randint(0, 2**31 - 1)
    base = base.convert("RGB")
    mask = normalize_mask(mask, base.size) if mask is not None else None

    if mode == "auto":
        mode = "local" if mask is not None and mask.getbbox() else "global"
    if mode == "local" and (mask is None or not mask.getbbox()):
        raise ValueError("Local mode requires a non-empty mask.")

    prompt = history_context(history, prompt)
    with provider_manager.gpu_lock:
        provider = provider_manager.get(provider_name)
        if mode == "global":
            model_img, original_size = fit_for_model(base, settings.max_image_side)
            sketch_ref = make_sketch_reference(base, sketch)
            if sketch_ref is not None:
                sketch_ref = sketch_ref.resize(model_img.size, Image.Resampling.LANCZOS)
            refs = [r.convert("RGB") for r in references[:3]]
            out = provider.edit(EditInput(model_img, prompt, refs, sketch_ref, seed))
            out = restore_size(out, original_size)
            return EditResult(out, provider_name, mode, seed)

        box = bbox_from_mask(mask)
        assert box is not None
        box = expand_box(box, settings.local_padding, base.size)
        box = ensure_min_crop(box, base.size, settings.local_min_side)
        crop = base.crop(box.pil())
        crop_model, crop_original_size = fit_for_model(crop, settings.local_max_side)

        sketch_ref = None
        if sketch is not None:
            full_sketch_ref = make_sketch_reference(base, sketch)
            if full_sketch_ref is not None:
                sketch_ref = full_sketch_ref.crop(box.pil()).resize(crop_model.size, Image.Resampling.LANCZOS)

        local_prompt = (
            "Edit the source crop according to the request. Preserve camera geometry, perspective, lighting and unrequested content. "
            + prompt
        )
        refs = [r.convert("RGB") for r in references[:3]]
        edited = provider.edit(EditInput(crop_model, local_prompt, refs, sketch_ref, seed))
        edited = restore_size(edited, crop_original_size)

        # Even if the generative model changes the whole crop, strict compositing makes the visible change local.
        feather = settings.mask_feather if strict_local else max(settings.mask_feather, 18)
        out = composite_local(base, edited, mask, box, feather)
        return EditResult(out, provider_name, mode, seed)


def run_generate(prompt: str, width: int, height: int, seed: int | None) -> EditResult:
    seed = seed if seed is not None else random.randint(0, 2**31 - 1)
    with provider_manager.gpu_lock:
        provider = provider_manager.get("flux")
        image = provider.generate(prompt, width, height, seed)
    return EditResult(image, "flux", "generate", seed)
