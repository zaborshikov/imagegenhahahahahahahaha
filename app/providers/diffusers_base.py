"""Base class for diffusers pipelines with the three "Save GPU" strategies.

off   The whole pipeline is resident on the GPU.
ram   ``enable_model_cpu_offload``: text_encoder -> transformer -> vae are moved
      to the GPU one at a time, the others wait in CPU RAM.
disk  Staged inference with a hard VRAM floor:
        stage 1  load the text encoder only, run ``pipe.encode_prompt`` (for
                 Qwen-Image-2.1 this also encodes the condition images through
                 the VL encoder), save its outputs to a .safetensors file, drop
                 the encoder from GPU and RAM.
        stage 2  load transformer + VAE without a text encoder, patch
                 ``encode_prompt`` to return the saved tensors, run denoising and
                 decoding, delete the temp file.
      Slowest (weights are re-read from disk for every request) but never keeps
      more than one heavyweight component alive.
"""

from __future__ import annotations

import logging
import uuid
from abc import abstractmethod
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from .base import EditInput, ImageProvider, RunOptions
from .loading import load_pipeline, release_gpu_memory
from ..config import Offload, Quantization, SaveGpu, settings
from ..runtime import generator_for

log = logging.getLogger(__name__)

_NONE_MARKER = "__none__"


class DiffusersProvider(ImageProvider):
    model_id: str
    quantization: Quantization
    offload: Offload
    default_steps: int
    min_steps: int
    max_steps: int
    # Whether the model is small enough to run on CPU without quantization.
    allow_cpu_full_precision: bool = False
    # Components that are only needed in the text-encoding stage (disk mode).
    text_stage_components: tuple[str, ...] = ("text_encoder",)
    # Components that are only needed in the denoising stage (disk mode).
    image_stage_components: tuple[str, ...] = ("transformer", "vae")

    def __init__(self) -> None:
        self.pipe = None
        self._pipe_mode: SaveGpu | None = None

    # -- loading --------------------------------------------------------------------------------

    def _offload_for(self, save_gpu: SaveGpu) -> Offload:
        if save_gpu == "ram":
            return "model"
        return self.offload

    def _load_full(self, save_gpu: SaveGpu):
        """off / ram: one resident pipeline, reloaded only if the placement strategy changed."""
        if self.pipe is not None and self._pipe_mode == save_gpu:
            return self.pipe
        self.unload()
        self.pipe = load_pipeline(
            self.model_id,
            quantization=self.quantization,
            offload=self._offload_for(save_gpu),
            allow_cpu_full_precision=self.allow_cpu_full_precision,
        )
        self._pipe_mode = save_gpu
        return self.pipe

    def _load_partial(self, exclude: tuple[str, ...]):
        """disk: load a pipeline with some components replaced by ``None``.

        The remaining components are still swapped GPU<->CPU one at a time (model
        offload), so e.g. the transformer is off the GPU while the VAE decodes.
        """
        return load_pipeline(
            self.model_id,
            quantization=self.quantization,
            offload="model",
            allow_cpu_full_precision=self.allow_cpu_full_precision,
            components={name: None for name in exclude},
        )

    def unload(self) -> None:
        self.pipe = None
        self._pipe_mode = None
        release_gpu_memory()

    # -- helpers --------------------------------------------------------------------------------

    def resolve_steps(self, requested: int | None) -> int:
        if requested is None:
            return self.default_steps
        return max(self.min_steps, min(self.max_steps, int(requested)))

    @staticmethod
    def _step_callback(options: RunOptions, total: int):
        def cb(pipe, i, t, kwargs):
            options.report(f"Denoising {i + 1}/{total}")
            return {}

        return cb

    @staticmethod
    def _tmp_path() -> Path:
        tmp_dir = settings.data_dir / "tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        return tmp_dir / f"stage_{uuid.uuid4().hex}.safetensors"

    # -- staged (disk) mode ---------------------------------------------------------------------

    @abstractmethod
    def _encode_prompt_kwargs(self, pipe, call_kwargs: dict) -> dict:
        """Arguments for ``pipe.encode_prompt`` equivalent to what ``pipe.__call__`` would pass."""

    @staticmethod
    def _pack(outputs: tuple) -> dict[str, torch.Tensor]:
        packed = {}
        for i, t in enumerate(outputs):
            if t is None:
                packed[f"{i}{_NONE_MARKER}"] = torch.zeros(1)
            else:
                packed[str(i)] = t.detach().to("cpu").contiguous()
        return packed

    @staticmethod
    def _unpack(tensors: dict[str, torch.Tensor]) -> tuple:
        out: dict[int, torch.Tensor | None] = {}
        for key, t in tensors.items():
            if key.endswith(_NONE_MARKER):
                out[int(key[: -len(_NONE_MARKER)])] = None
            else:
                out[int(key)] = t
        return tuple(out[i] for i in sorted(out))

    def _run_disk(self, call_kwargs: dict, options: RunOptions):
        # Stage 1: text encoder only.
        self.unload()
        options.report("Loading text encoder")
        pipe = self._load_partial(exclude=self.image_stage_components)
        options.report("Encoding prompt")
        with torch.inference_mode():
            outputs = pipe.encode_prompt(**self._encode_prompt_kwargs(pipe, call_kwargs))
        if not isinstance(outputs, tuple):
            outputs = (outputs,)
        path = self._tmp_path()
        save_file(self._pack(outputs), str(path))
        del outputs, pipe
        release_gpu_memory()
        log.info("disk mode: prompt embeddings saved to %s", path)

        try:
            # Stage 2: transformer + VAE only. encode_prompt is replaced by the saved result.
            options.report("Loading transformer + VAE")
            pipe = self._load_partial(exclude=self.text_stage_components)
            device = pipe._execution_device
            saved = self._unpack(load_file(str(path), device=str(device)))
            pipe.encode_prompt = lambda *args, **kwargs: saved
            with torch.inference_mode():
                result = pipe(**call_kwargs)
            del pipe, saved
            release_gpu_memory()
            return result
        finally:
            path.unlink(missing_ok=True)

    # -- dispatch -------------------------------------------------------------------------------

    def _run(self, call_kwargs: dict, options: RunOptions):
        steps = call_kwargs["num_inference_steps"]
        call_kwargs["callback_on_step_end"] = self._step_callback(options, steps)
        if options.save_gpu == "disk":
            return self._run_disk(call_kwargs, options)
        options.report("Loading model" if self.pipe is None else "Preparing")
        pipe = self._load_full(options.save_gpu)
        options.report("Encoding prompt")
        with torch.inference_mode():
            return pipe(**call_kwargs)

    # -- public API -----------------------------------------------------------------------------

    @abstractmethod
    def _edit_kwargs(self, req: EditInput, steps: int) -> dict: ...

    @abstractmethod
    def _generate_kwargs(self, prompt: str, width: int, height: int, steps: int) -> dict: ...

    def edit(self, req: EditInput):
        options = req.options
        steps = self.resolve_steps(options.steps)
        kwargs = self._edit_kwargs(req, steps)
        kwargs["generator"] = generator_for(req.seed)
        result = self._run(kwargs, options)
        return result.images[0].convert("RGB")

    def generate(self, prompt, width, height, seed=None, options: RunOptions | None = None):
        options = options or RunOptions()
        steps = self.resolve_steps(options.steps)
        kwargs = self._generate_kwargs(prompt, width, height, steps)
        kwargs["generator"] = generator_for(seed)
        result = self._run(kwargs, options)
        return result.images[0].convert("RGB")
