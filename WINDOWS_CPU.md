# Windows CPU mode (32 GB RAM)

Recommended: Python 3.12 x64. Python 3.14 may work, but 3.12 currently has broader ML-wheel compatibility.

```bat
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install -U pip
pip install -r requirements.txt
copy .env.windows_cpu .env
python run.py
```

Open http://127.0.0.1:8000.

Start with **FLUX** and 512px. FLUX.2 klein uses 4 inference steps but CPU generation is still slow.

## Experimental Qwen-Image-2.1 CPU mode
Qwen-Image-2.1 is ~16 GB with `QWEN21_QUANTIZATION=fp8` (the default in `.env.windows_cpu`), so it fits in 32 GB RAM, but CPU inference of a 7B DiT at 20+ steps takes many minutes per image. Do not use `QWEN21_QUANTIZATION=none` (~31 GB) on a 32 GB machine.

If it is too slow, use FLUX locally and Qwen-Image-2.1 on a Colab GPU; the UI/backend remains the same.
