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
2. Choose aspect ratio and target long-side size.
3. Enter a prompt and click **Generate image**.
4. The generated image becomes the current canvas and the UI switches to editing.

## Models

- `Qwen/Qwen-Image-Edit-2511`: high-quality editing backend (Apache-2.0).
- `black-forest-labs/FLUX.2-klein-4B`: text-to-image + image editing + references (Apache-2.0). It powers generation from scratch.

The runtime automatically chooses FP16 on pre-Ampere CUDA GPUs such as T4 and BF16 on Ampere/Ada/Hopper.

## Strict local editing

When a mask exists, local mode crops a padded ROI around the mask, runs generative editing on that crop and composites the result back through the user mask. With **Strict local** enabled, pixels outside the mask are copied from the source image rather than trusted to the generative model.

## Storage

There is no database. A session stores only the current image files and `chat.json` under `DATA_DIR`; chat history is the project history.

## Local launch

```bash
python -m venv .venv
source .venv/bin/activate
# install the appropriate CUDA build of PyTorch first if needed
pip install -r requirements.txt
cp .env.example .env
python run.py
```

Open `http://127.0.0.1:8000`.

## Google Colab

See `COLAB.md`. A T4-safe starter config is included as `.env.colab`.

## API smoke tests

```bash
PYTHONPATH=. pytest -q
```

The tests include an actual multipart upload → `/media/...` roundtrip, so the browser upload path is covered without loading model weights.
