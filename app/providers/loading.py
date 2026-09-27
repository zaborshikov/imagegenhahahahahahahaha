"""Shared pipeline loading for all diffusers-based providers.

Handles dtype selection, weight quantization, CPU/GPU placement and CPU offload
in one place, and degrades gracefully when an optional quantization stack is
missing instead of crashing deep inside ``from_pretrained``.

Quantization modes
------------------
``fp8``
    Weights are stored as ``float8_e4m3fn`` and up-cast per layer to the compute
    dtype on the fly (diffusers *layerwise casting*). Halves VRAM versus BF16 with
    negligible quality loss and no extra dependencies. Works on any CUDA GPU.
    Applied to the transformer and the text encoder; the VAE stays in BF16
    because VAE quantization is visible in the output.
``nf4`` / ``int8``
    bitsandbytes. Requires the ``bitsandbytes`` package; falls back to ``none``
    on CUDA if it is missing.
``none``
    Plain BF16 / FP16.
"""

from __future__ import annotations

import gc
import importlib.util
import logging
from functools import lru_cache

import torch

from ..config import Offload, Quantization, settings
from ..runtime import preferred_dtype

log = logging.getLogger(__name__)

# Components we quantize. The VAE is intentionally excluded.
QUANTIZED_COMPONENTS = ("transformer", "text_encoder")

# Layers that are numerically sensitive and cheap; keep them in the compute dtype.
FP8_SKIP_PATTERNS = ("pos_embed", "patch_embed", "norm", "^proj_in$", "^proj_out$", "embed_tokens", "lm_head")


@lru_cache(maxsize=1)
def bitsandbytes_status() -> tuple[bool, str | None]:
    """Return (available, reason). Cached: the answer cannot change at runtime."""
    if importlib.util.find_spec("bitsandbytes") is None:
        return False, "bitsandbytes is not installed (pip install bitsandbytes)"
    try:
        import bitsandbytes  # noqa: F401
    except Exception as exc:  # pragma: no cover - depends on the local CUDA stack
        return False, f"bitsandbytes is installed but failed to import: {exc}"
    return True, None


def fp8_supported() -> bool:
    return hasattr(torch, "float8_e4m3fn")


def effective_quantization(requested: Quantization) -> Quantization:
    """Quantization that will actually be applied given the installed stack (no side effects)."""
    if requested == "none":
        return "none"
    if requested == "fp8":
        return "fp8" if fp8_supported() else "none"
    available, _ = bitsandbytes_status()
    if available:
        return requested
    return "none" if torch.cuda.is_available() else requested


def resolve_quantization(requested: Quantization, *, model_id: str) -> Quantization:
    """Like :func:`effective_quantization` but logs the fallback and refuses impossible CPU setups."""
    if requested == "none":
        return "none"
    if requested == "fp8":
        if fp8_supported():
            return "fp8"
        log.warning("%s: fp8 requested but this torch build has no float8 dtype. Using full precision.", model_id)
        return "none"
    available, reason = bitsandbytes_status()
    if available:
        return requested
    if torch.cuda.is_available():
        log.warning("%s: %s quantization requested but %s. Falling back to full precision on CUDA.", model_id, requested, reason)
        return "none"
    raise RuntimeError(
        f"{model_id}: {requested} quantization was requested but {reason}. "
        "On CPU this model cannot be loaded without quantization. Install bitsandbytes or use fp8/FLUX."
    )


def _bnb_pipeline_config(quant: Quantization, dtype: torch.dtype, has_cuda: bool):
    from diffusers.quantizers import PipelineQuantizationConfig

    if quant == "nf4":
        backend = "bitsandbytes_4bit"
        qkwargs = {
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            # bitsandbytes' CPU backend only supports fp32 compute.
            "bnb_4bit_compute_dtype": dtype if has_cuda else torch.float32,
        }
    else:
        backend = "bitsandbytes_8bit"
        qkwargs = {"load_in_8bit": True}
    return PipelineQuantizationConfig(
        quant_backend=backend, quant_kwargs=qkwargs, components_to_quantize=list(QUANTIZED_COMPONENTS)
    )


def apply_fp8(module: torch.nn.Module, compute_dtype: torch.dtype, *, name: str) -> None:
    """Store weights in float8_e4m3fn, compute in ``compute_dtype``."""
    from diffusers.hooks import apply_layerwise_casting

    apply_layerwise_casting(
        module,
        storage_dtype=torch.float8_e4m3fn,
        compute_dtype=compute_dtype,
        skip_modules_pattern=FP8_SKIP_PATTERNS,
    )
    log.info("%s: weights stored as fp8 (e4m3), compute %s", name, compute_dtype)


def _apply_offload(pipe, offload: Offload, *, quantized_bnb: bool, model_id: str) -> None:
    if offload == "none":
        if not quantized_bnb:
            pipe.to("cuda")
        return
    try:
        if quantized_bnb:
            # A bnb pipeline was loaded with device_map="cuda"; offload requires resetting it.
            pipe.reset_device_map()
        if offload == "model":
            pipe.enable_model_cpu_offload()
        else:
            pipe.enable_sequential_cpu_offload()
    except Exception as exc:
        log.warning("%s: could not enable %s offload (%s). Keeping the pipeline on CUDA.", model_id, offload, exc)
        if not quantized_bnb:
            pipe.to("cuda")


def load_pipeline(
    model_id: str,
    *,
    quantization: Quantization,
    offload: Offload,
    allow_cpu_full_precision: bool,
    components: dict | None = None,
):
    """Load a ``DiffusionPipeline`` with the project's placement/quantization policy.

    ``components`` lets the caller pass pre-built or ``None`` components (e.g.
    ``text_encoder=None`` to skip loading it in the staged disk mode).
    ``allow_cpu_full_precision`` says whether the model is small enough to run
    unquantized on CPU.
    """
    from diffusers import DiffusionPipeline

    has_cuda = torch.cuda.is_available()
    dtype = preferred_dtype()
    quant = resolve_quantization(quantization, model_id=model_id)
    use_bnb = quant in ("nf4", "int8")

    kwargs: dict = {"dtype": dtype}
    if settings.hf_token:
        kwargs["token"] = settings.hf_token
    if components:
        kwargs.update(components)

    if use_bnb:
        kwargs["quantization_config"] = _bnb_pipeline_config(quant, dtype, has_cuda)
        kwargs["device_map"] = "cuda" if has_cuda else "cpu"
    elif not has_cuda:
        if not allow_cpu_full_precision and quant != "fp8":
            raise RuntimeError(
                f"{model_id} is too large to run on CPU without quantization. "
                "Set *_QUANTIZATION=nf4 with bitsandbytes installed, or use the FLUX provider."
            )
        kwargs["device_map"] = "cpu"

    log.info("Loading %s (quant=%s, offload=%s, dtype=%s, cuda=%s)", model_id, quant, offload, dtype, has_cuda)
    pipe = DiffusionPipeline.from_pretrained(model_id, **kwargs)

    if quant == "fp8":
        for comp in QUANTIZED_COMPONENTS:
            module = getattr(pipe, comp, None)
            if isinstance(module, torch.nn.Module):
                apply_fp8(module, dtype, name=f"{model_id}/{comp}")

    if has_cuda:
        _apply_offload(pipe, offload, quantized_bnb=use_bnb, model_id=model_id)
    else:
        log.warning("%s is running on CPU. Expect very slow inference.", model_id)

    pipe.set_progress_bar_config(disable=False)
    return pipe


def release_gpu_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
