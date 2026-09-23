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

## Experimental Qwen CPU mode
Qwen-Image-Edit-2511 is much larger. Install current bitsandbytes CPU support:

```bat
pip install -r requirements-cpu-qwen.txt
```

Keep `QWEN_QUANTIZATION=nf4`. Do not use `QWEN_QUANTIZATION=none` on a 32 GB machine.

If Qwen quantized CPU loading fails on your specific PyTorch/Python build, use FLUX locally and Qwen on Colab GPU; the UI/backend remains the same.
