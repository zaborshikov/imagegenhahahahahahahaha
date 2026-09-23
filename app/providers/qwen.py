from __future__ import annotations

import gc
import logging

import torch
from PIL import Image

from .base import EditInput, ImageProvider
from ..config import settings
from ..runtime import generator_for, preferred_dtype

log = logging.getLogger(__name__)


class QwenEditProvider(ImageProvider):
    name = "qwen"

    def __init__(self) -> None:
        self.pipe = None

    def _load(self):
        if self.pipe is not None:
            return self.pipe

        from diffusers import DiffusionPipeline

        has_cuda = torch.cuda.is_available()
        dtype = preferred_dtype()
        kwargs = {"dtype": dtype}
        if settings.hf_token:
            kwargs["token"] = settings.hf_token

        quant = settings.qwen_quantization
        if quant != "none":
            from diffusers.quantizers import PipelineQuantizationConfig

            backend = "bitsandbytes_4bit" if quant == "nf4" else "bitsandbytes_8bit"
            qkwargs = (
                {
                    "load_in_4bit": True,
                    "bnb_4bit_quant_type": "nf4",
                    "bnb_4bit_compute_dtype": dtype,
                }
                if quant == "nf4"
                else {"load_in_8bit": True}
            )
            # device_map is a placement strategy, not an unconditional CUDA flag.
            # On CPU-only Windows use the CPU bitsandbytes backend; on CUDA place on GPU.
            if not has_cuda and quant == "nf4":
                qkwargs["bnb_4bit_compute_dtype"] = torch.float32
            kwargs["quantization_config"] = PipelineQuantizationConfig(
                quant_backend=backend,
                quant_kwargs=qkwargs,
                components_to_quantize=["transformer", "text_encoder"],
            )
            kwargs["device_map"] = "cuda" if has_cuda else "cpu"
        elif not has_cuda:
            raise RuntimeError(
                "Qwen-Image-Edit-2511 is too large for this 32 GB CPU setup without quantization. "
                "Set QWEN_QUANTIZATION=nf4 (experimental CPU mode) or use FLUX.2 klein for local CPU inference."
            )

        log.info(
            "Loading %s (quant=%s, offload=%s, dtype=%s)",
            settings.qwen_model_id,
            quant,
            settings.qwen_offload,
            dtype,
        )
        pipe = DiffusionPipeline.from_pretrained(settings.qwen_model_id, **kwargs)

        if has_cuda:
            if quant == "none":
                if settings.qwen_offload == "sequential":
                    pipe.enable_sequential_cpu_offload()
                elif settings.qwen_offload == "model":
                    pipe.enable_model_cpu_offload()
                else:
                    pipe.to("cuda")
            elif settings.qwen_offload != "none":
                try:
                    pipe.reset_device_map()
                    if settings.qwen_offload == "model":
                        pipe.enable_model_cpu_offload()
                    else:
                        pipe.enable_sequential_cpu_offload()
                except Exception as exc:
                    log.warning("Could not enable offload on quantized Qwen pipeline: %s", exc)
        else:
            log.warning("Running Qwen-Image-Edit on CPU. Expect very slow inference; FLUX.2 klein is recommended locally.")

        pipe.set_progress_bar_config(disable=False)
        self.pipe = pipe
        return pipe

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

    def edit(self, req: EditInput) -> Image.Image:
        pipe = self._load()
        images = [req.image.convert("RGB")]
        if req.sketch_reference is not None:
            images.append(req.sketch_reference.convert("RGB"))
        images.extend(x.convert("RGB") for x in req.references[:3])

        with torch.inference_mode():
            result = pipe(
                image=images,
                prompt=self._prompt(req),
                generator=generator_for(req.seed),
                true_cfg_scale=settings.qwen_true_cfg_scale,
                negative_prompt=" ",
                num_inference_steps=settings.qwen_steps,
                guidance_scale=1.0,
                num_images_per_prompt=1,
            )
        return result.images[0].convert("RGB")

    def unload(self) -> None:
        self.pipe = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
