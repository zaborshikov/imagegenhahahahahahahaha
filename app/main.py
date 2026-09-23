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
from .schemas import AssetResponse, GenerateRequest, JobResponse, JobState, SessionCreateResponse
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
    return {
        "ok": True,
        "cuda_expected": True,
        "default_provider": settings.default_provider,
        "qwen_model": settings.qwen_model_id,
        "flux_model": settings.flux_model_id,
        "gpu": cuda_info(),
        "capabilities": {"upload": True, "generate": ["flux"], "edit": ["qwen", "flux"]},
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
        )
        state.revision += 1
        out_path = state.root / f"edit_{state.revision:04d}.png"
        save_png(result.image, out_path)
        state.current_image = out_path
        state.append("assistant", f"Applied {result.mode} edit with {result.provider}; seed={result.seed}.")
        state.persist_history()
        update_job(
            job_id,
            status="done",
            image_url=media_url(out_path),
            provider=result.provider,
            mode=result.mode,
            seed=result.seed,
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
    mask: UploadFile | None = File(None),
    sketch: UploadFile | None = File(None),
    references: list[UploadFile] | None = File(None),
):
    try:
        sessions.get(sid)
        if provider not in {"qwen", "flux"}:
            raise ValueError("provider must be qwen or flux")
        if mode not in {"auto", "local", "global"}:
            raise ValueError("mode must be auto, local, or global")
        payload = {
            "prompt": prompt,
            "provider": provider,
            "mode": mode,
            "strict_local": strict_local,
            "seed": seed,
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
        result = run_generate(req.prompt, req.width, req.height, req.seed)
        state.revision += 1
        out_path = state.root / f"generated_{state.revision:04d}.png"
        save_png(result.image, out_path)
        state.current_image = out_path
        state.append("assistant", f"Generated image with FLUX.2 [klein] 4B; seed={result.seed}.")
        state.persist_history()
        update_job(job_id, status="done", image_url=media_url(out_path), provider="flux", mode="generate", seed=result.seed)
    except Exception as exc:
        log.exception("Generation failed")
        update_job(job_id, status="error", error=f"{type(exc).__name__}: {exc}")


@app.post("/api/sessions/{sid}/generate", response_model=JobResponse)
def create_generation(sid: str, req: GenerateRequest):
    try:
        sessions.get(sid)
    except KeyError:
        raise HTTPException(404, "Unknown session")
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
