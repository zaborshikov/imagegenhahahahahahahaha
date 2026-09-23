# Google Colab launch

Enable **Runtime → Change runtime type → T4 GPU** (or a stronger GPU).

Upload `iterative_image_editor_mvp_v2.zip`, then run:

```bash
!unzip -o iterative_image_editor_mvp_v2.zip -d /content/
%cd /content/iterative_image_editor
!cp .env.colab .env
!pip install -U pip
!pip install -r requirements.txt
```

Start the server **in the background**:

```python
import subprocess, time

server_log = open('/content/editor-server.log', 'w')
server = subprocess.Popen(
    ['python', 'run.py'],
    cwd='/content/iterative_image_editor',
    stdout=server_log,
    stderr=subprocess.STDOUT,
)
time.sleep(3)
print('server pid:', server.pid)
!tail -30 /content/editor-server.log
```

Sanity check:

```bash
!curl -I http://127.0.0.1:8000/
!curl http://127.0.0.1:8000/api/health
```

Install Cloudflare tunnel:

```bash
!wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O /usr/local/bin/cloudflared
!chmod +x /usr/local/bin/cloudflared
```

Start it in a separate cell and keep that cell running:

```bash
!cloudflared tunnel --url http://127.0.0.1:8000
```

Open the printed `https://....trycloudflare.com` URL.

## T4 note

The app detects the GPU. T4 uses `float16`; Ampere/Ada/Hopper GPUs use `bfloat16`. Start with FLUX.2 klein 4B. Qwen-Image-Edit-2511 is much heavier and its first load can exceed a small Colab runtime's CPU/GPU memory even with 4-bit quantization/offload.
