from __future__ import annotations

import logging

from .base import EditInput
from .diffusers_base import DiffusersProvider
from ..config import settings

log = logging.getLogger(__name__)


class FluxKleinProvider(DiffusersProvider):
    """black-forest-labs/FLUX.2-klein-4B: distilled text-to-image + editing model."""

    name = "flux"
    supports_generate = True

    model_id = settings.flux_model_id
    quantization = settings.flux_quantization
    offload = settings.flux_offload
    default_steps = settings.flux_steps
    min_steps = settings.flux_min_steps
    max_steps = settings.flux_max_steps
    # 4B params: slow but feasible on CPU in full precision.
    allow_cpu_full_precision = True
    text_stage_components = ("text_encoder",)

    @staticmethod
    def _prompt(req: EditInput) -> str:
        extra = ""
        if req.sketch_reference is not None:
            extra += " Use the second image only as a sketch/placement guide and render the requested result naturally unless told otherwise."
        if req.references:
            extra += " Additional images are visual references. Preserve the source image except for the requested edit."
        return req.prompt.strip() + extra

    def _edit_kwargs(self, req: EditInput, steps: int) -> dict:
        images = [req.image.convert("RGB")]
        if req.sketch_reference is not None:
            images.append(req.sketch_reference.convert("RGB"))
        images.extend(x.convert("RGB") for x in req.references[:3])
        return {
            "image": images[0] if len(images) == 1 else images,
            "prompt": self._prompt(req),
            "num_inference_steps": steps,
            "guidance_scale": 1.0,
        }

    def _generate_kwargs(self, prompt: str, width: int, height: int, steps: int) -> dict:
        return {
            "prompt": prompt.strip(),
            "width": width,
            "height": height,
            "num_inference_steps": steps,
            "guidance_scale": 1.0,
        }

    def _encode_prompt_kwargs(self, pipe, call_kwargs: dict) -> dict:
        # klein is distilled: guidance is off, so encode_prompt runs exactly once with the positive prompt.
        return {
            "prompt": call_kwargs["prompt"],
            "num_images_per_prompt": 1,
            "max_sequence_length": call_kwargs.get("max_sequence_length", 512),
            "text_encoder_out_layers": call_kwargs.get("text_encoder_out_layers", (9, 18, 27)),
        }
