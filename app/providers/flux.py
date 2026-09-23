from __future__ import annotations

import gc
import logging

import torch
from PIL import Image

from .base import EditInput, ImageProvider
from ..config import settings
from ..runtime import generator_for, preferred_dtype

log = logging.getLogger(__name__)


class FluxKleinProvider(ImageProvider):
    name = "flux"

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

        quant = settings.flux_quantization
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
            if not has_cuda and quant == "nf4":
                qkwargs["bnb_4bit_compute_dtype"] = torch.float32
            kwargs["quantization_config"] = PipelineQuantizationConfig(
                quant_backend=backend,
                quant_kwargs=qkwargs,
                components_to_quantize=["transformer", "text_encoder"],
            )
            kwargs["device_map"] = "cuda" if has_cuda else "cpu"
        elif not has_cuda:
            # Direct CPU placement avoids the old unconditional pipe.to("cuda") crash.
            kwargs["device_map"] = "cpu"

        log.info(
            "Loading %s (quant=%s, offload=%s, dtype=%s)",
            settings.flux_model_id,
            quant,
            settings.flux_offload,
            dtype,
        )
        pipe = DiffusionPipeline.from_pretrained(settings.flux_model_id, **kwargs)

        if has_cuda:
            if quant == "none":
                if settings.flux_offload == "sequential":
                    pipe.enable_sequential_cpu_offload()
                elif settings.flux_offload == "model":
                    pipe.enable_model_cpu_offload()
                else:
                    pipe.to("cuda")
            elif settings.flux_offload != "none":
                try:
                    pipe.reset_device_map()
                    if settings.flux_offload == "model":
                        pipe.enable_model_cpu_offload()
                    else:
                        pipe.enable_sequential_cpu_offload()
                except Exception as exc:
                    log.warning("Could not enable offload on quantized FLUX pipeline: %s", exc)
        else:
            log.warning("Running FLUX.2 klein on CPU (dtype=%s). This works but is much slower than CUDA.", dtype)

        pipe.set_progress_bar_config(disable=False)
        self.pipe = pipe
        return pipe

    @staticmethod
    def _prompt(req: EditInput) -> str:
        extra = ""
        if req.sketch_reference is not None:
            extra += " Use the second image only as a sketch/placement guide and render the requested result naturally unless told otherwise."
        if req.references:
            extra += " Additional images are visual references. Preserve the source image except for the requested edit."
        return req.prompt.strip() + extra

    def edit(self, req: EditInput) -> Image.Image:
        pipe = self._load()
        images = [req.image.convert("RGB")]
        if req.sketch_reference is not None:
            images.append(req.sketch_reference.convert("RGB"))
        images.extend(x.convert("RGB") for x in req.references[:3])
        image_arg = images[0] if len(images) == 1 else images
        with torch.inference_mode():
            result = pipe(
                image=image_arg,
                prompt=self._prompt(req),
                generator=generator_for(req.seed),
                num_inference_steps=settings.flux_steps,
                guidance_scale=1.0,
            )
        return result.images[0].convert("RGB")

    def generate(self, prompt: str, width: int, height: int, seed: int | None = None) -> Image.Image:
        pipe = self._load()
        with torch.inference_mode():
            result = pipe(
                prompt=prompt.strip(),
                width=width,
                height=height,
                generator=generator_for(seed),
                num_inference_steps=settings.flux_steps,
                guidance_scale=1.0,
            )
        return result.images[0].convert("RGB")

    def unload(self) -> None:
        self.pipe = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
