# Iterative Image Editor — MVP v2

Conversational open-weight image generation/editing service with manual mask painting, sketch guidance and multi-turn editing.

## Two first-class workflows

**Edit a photo**
1. Click the empty canvas / **Open photo**, or drag a PNG/JPEG/WebP onto it.
2. For local edits, paint the region with **Mask**.
3. Optionally draw a **Sketch** for geometry/placement and/or attach reference images.
4. Describe the edit and click **Apply edit**.
5. Keep editing the resulting image in the same chat session.

**Create from scratch**
1. Open **Создать с нуля**.
2. Choose the model, aspect ratio and target long-side size.
3. Enter a prompt and click **Generate image**.
4. The generated image becomes the current canvas and the UI switches to editing.

## Models

Both models are unified text-to-image **and** editing models, so either can be used in both workflows.

| Provider | Model | Params | License | Notes |
|---|---|---|---|---|
| `qwen21` (default) | [`Qwen/Qwen-Image-2.1`](https://huggingface.co/Qwen/Qwen-Image-2.1) | 7B DiT + 8B Qwen3-VL text encoder | Qwen Research License | Best quality; multi-reference editing; 40 steps by default |
| `flux` | [`black-forest-labs/FLUX.2-klein-4B`](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) | 4B | Apache-2.0 | Distilled, 4 steps; fast fallback for small GPUs / CPU |

The runtime automatically chooses FP16 on pre-Ampere CUDA GPUs such as T4 and BF16 on Ampere/Ada/Hopper.

### Weight precision

`*_QUANTIZATION` controls how weights are stored (compute is always BF16/FP16):

| Value | What it does | Qwen-Image-2.1 weights | Needs |
|---|---|---|---|
| `fp8` (default) | float8 e4m3 storage, per-layer up-cast on the fly (diffusers layerwise casting). Quality is visually identical to bf16. VAE stays bf16. | ~16 GB | nothing extra |
| `none` | plain bf16/fp16 | ~31 GB | — |
| `nf4` / `int8` | bitsandbytes | ~9 / ~16 GB | `pip install bitsandbytes` (falls back to `none` on CUDA if missing) |

### Inference controls in the UI

- **Steps** — slider with a per-model range; defaults come from `*_STEPS` in `.env`.
- **Save GPU** — how the pipeline uses memory:
  - `off`: everything resident on the GPU (fastest). Weights stay loaded between requests.
  - `ram`: components (text encoder → transformer → VAE) are swapped GPU↔CPU one at a time. Needs enough CPU RAM to hold the whole pipeline (~16 GB for Qwen-Image-2.1 in fp8).
  - `disk`: staged inference. The text encoder is loaded alone, encodes the prompt (and condition images), its output is written to `DATA_DIR/tmp/*.safetensors` and the encoder is fully unloaded. Then transformer + VAE are loaded without a text encoder and run from the saved tensors. Intermediate files are deleted afterwards. Lowest CPU RAM footprint (one stage at a time), slowest because weights are re-read per request.

Job progress (`Encoding prompt`, `Denoising 12/40`, …) is streamed to the chat status line via `/api/jobs/{id}`.

### GPU memory requirements

Measured peak VRAM on an H100 (includes ~0.7 GB CUDA context), fp8 weights, 1024×1024 output. Timings are wall-clock per request with weights already cached on local disk.

| Model | Save GPU | Peak VRAM | Time | Idle VRAM after | Fits |
|---|---|---|---|---|---|
| Qwen-Image-2.1 (30 steps) | `off` | **26.2 GB** | 7 s | 26.2 GB | 32 GB+ (A100 40/80, H100, RTX 5090/6000) |
| Qwen-Image-2.1 (30 steps) | `ram` | **10.5 GB** | 14 s | 0.8 GB | 12 GB+ GPU and ≥ 20 GB CPU RAM |
| Qwen-Image-2.1 (30 steps) | `disk` | **10.4 GB** | 17 s | 0.7 GB | 12 GB+ GPU (T4 16 GB, RTX 3060 12 GB, Colab), ~10 GB CPU RAM |
| FLUX.2 klein 4B (4 steps) | `off` | **14.1 GB** | 2 s | 14.1 GB | 16 GB+ |
| FLUX.2 klein 4B (4 steps) | `ram` | **5.7 GB** | 4 s | 0.8 GB | 8 GB+ |
| FLUX.2 klein 4B (4 steps) | `disk` | **5.7 GB** | 8 s | 0.7 GB | 8 GB+ |

Rules of thumb:
- Peak VRAM in `ram`/`disk` is dominated by the largest single component plus activations: the fp8 Qwen3-VL text encoder (~8 GB) for Qwen-Image-2.1, the transformer (~4 GB) for FLUX.
- With `none` (bf16) instead of `fp8`, roughly double every weight figure (Qwen-Image-2.1 `off` ≈ 31 GB of weights + activations).
- Larger outputs (2048×2048) add several GB of activations to every row.
- `ram` and `disk` have the same VRAM peak; choose `disk` when CPU RAM is the bottleneck (e.g. Colab with 12 GB RAM), otherwise `ram` is faster.

## Strict local editing

When a mask exists, local mode crops a padded ROI around the mask, runs generative editing on that crop and composites the result back through the user mask. With **Strict local** enabled, pixels outside the mask are copied from the source image rather than trusted to the generative model.

## Storage

There is no database. A session stores only the current image files and `chat.json` under `DATA_DIR`; chat history is the project history.

## Local launch

```bash
python -m venv .venv
source .venv/bin/activate
# install the CUDA build of PyTorch matching your driver first, e.g.
#   pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
cp .env.example .env
python run.py
```

Open `http://127.0.0.1:8000`. Model weights are downloaded from Hugging Face on first use.

## Google Colab

See `COLAB.md`. A T4-safe starter config is included as `.env.colab`.

## API

- `GET /api/health` — GPU info, provider list with labels/step ranges/effective quantization, defaults.
- `POST /api/sessions` → `{session_id}`
- `POST /api/sessions/{sid}/upload` (multipart `image`)
- `POST /api/sessions/{sid}/edit` (multipart: `prompt`, `provider`, `mode`, `strict_local`, `seed`, `steps`, `save_gpu`, `mask`, `sketch`, `references[]`)
- `POST /api/sessions/{sid}/generate` (JSON: `prompt`, `width`, `height`, `seed`, `provider`, `steps`, `save_gpu`)
- `GET /api/jobs/{job_id}` — `status`, `stage`, result `image_url`, `provider`, `steps`, `seed`, `save_gpu`

## Tests

```bash
PYTHONPATH=. pytest -q
```

Tests cover the multipart upload → `/media/...` roundtrip, quantization fallback logic, fp8 casting, and the three Save GPU strategies (with a fake pipeline, no weights needed).
