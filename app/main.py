from __future__ import annotations

import io
import logging
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps

from .config import settings
from .editor import run_edit, run_generate
from .image_ops import normalize_rgb, save_png
from .providers.base import RunOptions
from .providers.loading import bitsandbytes_status, effective_quantization
from .providers.manager import provider_manager
from .schemas import AssetResponse, GenerateRequest, JobResponse, JobState, SaveGpuMode, SessionCreateResponse
from .session import sessions
from .runtime import cuda_info

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)

app = FastAPI(title="Iterative Image Editor", version="0.1.0")
static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")
app.mount("/media", StaticFiles(directory=settings.data_dir), name="media")

executor = ThreadPoolExecutor(max_workers=2)
jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()


def media_url(path: Path) -> str:
    rel = path.resolve().relative_to(settings.data_dir.resolve()).as_posix()
    return f"/media/{rel}"


def read_upload(file: UploadFile | None) -> Image.Image | None:
    if file is None:
        return None
    raw = file.file.read(settings.max_upload_mb * 1024 * 1024 + 1)
    if len(raw) > settings.max_upload_mb * 1024 * 1024:
        raise ValueError(f"File too large; max {settings.max_upload_mb} MB")
    img = Image.open(io.BytesIO(raw))
    img.load()
    return ImageOps.exif_transpose(img)


@app.get("/")
def index():
    return FileResponse(static_dir / "index.html")


@app.get("/api/health")
def health():
    bnb_available, bnb_reason = bitsandbytes_status()
    providers = provider_manager.describe()
    return {
        "ok": True,
        "cuda_expected": True,
        "default_provider": settings.default_provider,
        "default_generate_provider": settings.default_generate_provider,
        "default_save_gpu": settings.default_save_gpu,
        "save_gpu_modes": [m.value for m in SaveGpuMode],
        "gpu": cuda_info(),
        "providers": [
            {
                "name": p.name,
                "label": p.label,
                "model_id": p.model_id,
                "license": p.license,
                "supports_edit": p.supports_edit,
                "supports_generate": p.supports_generate,
                "steps": {"default": p.default_steps, "min": p.min_steps, "max": p.max_steps},
                "quantization": {
                    "requested": p.quantization,
                    "effective": effective_quantization(p.quantization),
                },
            }
            for p in providers
        ],
        "quantization": {
            "bitsandbytes_available": bnb_available,
            "bitsandbytes_error": bnb_reason,
            **{
                p.name: {"requested": p.quantization, "effective": effective_quantization(p.quantization)}
                for p in providers
            },
        },
        "capabilities": {
            "upload": True,
            "generate": provider_manager.generate_providers,
            "edit": provider_manager.edit_providers,
        },
    }


@app.post("/api/sessions", response_model=SessionCreateResponse)
def create_session():
    state = sessions.create()
    return SessionCreateResponse(session_id=state.id)


@app.post("/api/sessions/{sid}/reset")
def reset_session(sid: str):
    try:
        state = sessions.get(sid)
    except KeyError:
        raise HTTPException(404, "Unknown session")
    state.current_image = None
    state.messages.clear()
    state.revision = 0
    state.persist_history()
    return {"ok": True}


@app.post("/api/sessions/{sid}/upload", response_model=AssetResponse)
def upload_source(sid: str, image: UploadFile = File(...)):
    try:
        state = sessions.get(sid)
        img = read_upload(image)
        assert img is not None
        img = normalize_rgb(img, settings.max_image_side)
        state.revision = 0
        path = state.root / "source.png"
        save_png(img, path)
        state.current_image = path
        state.messages.clear()
        state.persist_history()
        return AssetResponse(image_url=media_url(path), width=img.width, height=img.height)
    except KeyError:
        raise HTTPException(404, "Unknown session")
    except Exception as exc:
        raise HTTPException(400, str(exc))


def update_job(job_id: str, **values):
    with jobs_lock:
        jobs.setdefault(job_id, {}).update(values)


def run_options(job_id: str, steps: int | None, save_gpu: str | None) -> RunOptions:
    return RunOptions(
        steps=steps,
        save_gpu=save_gpu or settings.default_save_gpu,
        progress=lambda stage: update_job(job_id, stage=stage),
    )


def edit_worker(job_id: str, sid: str, payload: dict):
    try:
        state = sessions.get(sid)
        update_job(job_id, status="running")
        if not state.current_image:
            raise ValueError("Upload or generate an image first.")

        base = Image.open(state.current_image).convert("RGB")
        mask = payload.get("mask")
        sketch = payload.get("sketch")
        refs = payload.get("references", [])
        prompt = payload["prompt"]

        state.append("user", prompt)
        result = run_edit(
            base=base,
            prompt=prompt,
            provider_name=payload["provider"],
            mode=payload["mode"],
            strict_local=payload["strict_local"],
            history=state.messages,
            mask=mask,
            sketch=sketch,
            references=refs,
            seed=payload.get("seed"),
            options=run_options(job_id, payload.get("steps"), payload.get("save_gpu")),
        )
        state.revision += 1
        out_path = state.root / f"edit_{state.revision:04d}.png"
        save_png(result.image, out_path)
        state.current_image = out_path
        label = provider_manager.info(result.provider).label
        state.append(
            "assistant", f"Applied {result.mode} edit with {label}; steps={result.steps}, seed={result.seed}."
        )
        state.persist_history()
        update_job(
            job_id,
            status="done",
            stage="Done",
            image_url=media_url(out_path),
            provider=result.provider,
            mode=result.mode,
            seed=result.seed,
            steps=result.steps,
            save_gpu=result.save_gpu,
        )
    except Exception as exc:
        log.exception("Edit failed")
        update_job(job_id, status="error", error=f"{type(exc).__name__}: {exc}")


@app.post("/api/sessions/{sid}/edit", response_model=JobResponse)
def create_edit(
    sid: str,
    prompt: str = Form(...),
    provider: str = Form(settings.default_provider),
    mode: str = Form("auto"),
    strict_local: bool = Form(True),
    seed: int | None = Form(None),
    steps: int | None = Form(None, ge=1, le=200),
    save_gpu: SaveGpuMode | None = Form(None),
    mask: UploadFile | None = File(None),
    sketch: UploadFile | None = File(None),
    references: list[UploadFile] | None = File(None),
):
    try:
        sessions.get(sid)
        provider_manager.validate(provider)
        if mode not in {"auto", "local", "global"}:
            raise ValueError("mode must be auto, local, or global")
        payload = {
            "prompt": prompt,
            "provider": provider,
            "mode": mode,
            "strict_local": strict_local,
            "seed": seed,
            "steps": steps,
            "save_gpu": save_gpu.value if save_gpu else None,
            "mask": read_upload(mask),
            "sketch": read_upload(sketch),
            "references": [read_upload(x) for x in (references or [])][:3],
        }
        payload["references"] = [x for x in payload["references"] if x is not None]
    except KeyError:
        raise HTTPException(404, "Unknown session")
    except Exception as exc:
        raise HTTPException(400, str(exc))

    job_id = uuid.uuid4().hex
    update_job(job_id, status="queued")
    executor.submit(edit_worker, job_id, sid, payload)
    return JobResponse(job_id=job_id, status="queued")


def generate_worker(job_id: str, sid: str, req: GenerateRequest):
    try:
        state = sessions.get(sid)
        update_job(job_id, status="running")
        state.append("user", req.prompt)
        result = run_generate(
            req.prompt,
            req.width,
            req.height,
            req.seed,
            req.provider,
            options=run_options(job_id, req.steps, req.save_gpu.value if req.save_gpu else None),
        )
        state.revision += 1
        out_path = state.root / f"generated_{state.revision:04d}.png"
        save_png(result.image, out_path)
        state.current_image = out_path
        label = provider_manager.info(result.provider).label
        state.append("assistant", f"Generated image with {label}; steps={result.steps}, seed={result.seed}.")
        state.persist_history()
        update_job(
            job_id,
            status="done",
            stage="Done",
            image_url=media_url(out_path),
            provider=result.provider,
            mode="generate",
            seed=result.seed,
            steps=result.steps,
            save_gpu=result.save_gpu,
        )
    except Exception as exc:
        log.exception("Generation failed")
        update_job(job_id, status="error", error=f"{type(exc).__name__}: {exc}")


@app.post("/api/sessions/{sid}/generate", response_model=JobResponse)
def create_generation(sid: str, req: GenerateRequest):
    try:
        sessions.get(sid)
        provider_manager.validate(req.provider or settings.default_generate_provider, for_generate=True)
    except KeyError:
        raise HTTPException(404, "Unknown session")
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    job_id = uuid.uuid4().hex
    update_job(job_id, status="queued")
    executor.submit(generate_worker, job_id, sid, req)
    return JobResponse(job_id=job_id, status="queued")


@app.get("/api/jobs/{job_id}", response_model=JobState)
def get_job(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Unknown job")
    return JobState(job_id=job_id, **job)
