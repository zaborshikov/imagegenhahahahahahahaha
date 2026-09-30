from __future__ import annotations

import logging

from .base import EditInput
from .diffusers_base import DiffusersProvider
from ..config import settings

log = logging.getLogger(__name__)


def _ensure_pipeline_class_available() -> None:
    """Qwen-Image-2.1 ships its own pipeline class which only exists in recent diffusers."""
    import diffusers

    if not hasattr(diffusers, "QwenImage21Pipeline"):
        raise RuntimeError(
            f"{settings.qwen21_model_id} needs diffusers with QwenImage21Pipeline (>= 0.41 / git main); "
            f"installed diffusers=={diffusers.__version__}. "
            "Run: pip install -U git+https://github.com/huggingface/diffusers"
        )


class QwenImage21Provider(DiffusersProvider):
    """Qwen/Qwen-Image-2.1: unified text-to-image and multi-reference editing model."""

    name = "qwen21"
    supports_generate = True

    model_id = settings.qwen21_model_id
    quantization = settings.qwen21_quantization
    offload = settings.qwen21_offload
    default_steps = settings.qwen21_steps
    min_steps = settings.qwen21_min_steps
    max_steps = settings.qwen21_max_steps
    # 7B DiT + Qwen3-VL 8B text encoder: too heavy for CPU without quantization.
    allow_cpu_full_precision = False

    def _load_full(self, save_gpu):
        _ensure_pipeline_class_available()
        return super()._load_full(save_gpu)

    def _load_partial(self, exclude):
        _ensure_pipeline_class_available()
        return super()._load_partial(exclude)

    # -- prompts --------------------------------------------------------------------------------

    @staticmethod
    def _prompt(req: EditInput) -> str:
        extra = ""
        if req.sketch_reference is not None:
            extra += (
                " Image 2 is an annotation/sketch drawn over the source. Use its lines only as structural and placement guidance; "
                "do not reproduce guide strokes as visible paint unless explicitly requested."
            )
        if req.references:
            offset = 3 if req.sketch_reference is not None else 2
            extra += f" Images {offset} onward are visual references; borrow only the requested identity, object, material, or style from them."
        return req.prompt.strip() + extra

    @staticmethod
    def _cfg_kwargs() -> dict:
        # true_cfg_scale > 1 enables classifier-free guidance and requires a negative prompt.
        if settings.qwen21_true_cfg_scale > 1.0:
            return {"true_cfg_scale": settings.qwen21_true_cfg_scale, "negative_prompt": " "}
        return {"true_cfg_scale": 1.0}

    # -- pipeline kwargs ------------------------------------------------------------------------

    def _edit_kwargs(self, req: EditInput, steps: int) -> dict:
        images = [req.image.convert("RGB")]
        if req.sketch_reference is not None:
            images.append(req.sketch_reference.convert("RGB"))
        images.extend(x.convert("RGB") for x in req.references[:3])
        # height/width are derived from the first condition image's aspect ratio; output_resolution sets the scale.
        return {
            "prompt": self._prompt(req),
            "image": images,
            "num_inference_steps": steps,
            "output_resolution": settings.qwen21_edit_resolution,
            "num_images_per_prompt": 1,
            **self._cfg_kwargs(),
        }

    def _generate_kwargs(self, prompt: str, width: int, height: int, steps: int) -> dict:
        return {
            "prompt": prompt.strip(),
            "width": width,
            "height": height,
            "num_inference_steps": steps,
            "num_images_per_prompt": 1,
            **self._cfg_kwargs(),
        }

    # -- disk mode ------------------------------------------------------------------------------

    def _encode_prompt_kwargs(self, pipe, call_kwargs: dict) -> dict:
        """Mirror the condition-image preprocessing of ``QwenImage21Pipeline.__call__``."""
        from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_dimensions

        images = call_kwargs.get("image")
        input_images = None
        if images:
            resolution = call_kwargs.get("output_resolution", 1024)
            input_images = []
            for img in images:
                if img.mode != "RGBA":
                    img = img.convert("RGBA")
                w, h, _ = calculate_dimensions(resolution * resolution, img.size[0] / img.size[1])
                input_images.append(pipe.image_processor.resize(img, width=w, height=h))
        if call_kwargs.get("true_cfg_scale", 1.0) > 1.0:
            log.warning("disk mode ignores negative prompts; running without classifier-free guidance.")
        return {
            "prompt": call_kwargs["prompt"],
            "image": input_images,
            "num_images_per_prompt": call_kwargs.get("num_images_per_prompt", 1),
        }
