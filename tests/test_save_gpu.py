"""Save-GPU strategies and per-request steps, exercised with a fake pipeline (no weights)."""

from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from app.providers.base import EditInput, RunOptions
from app.providers.diffusers_base import DiffusersProvider


class FakePipe:
    """Mimics the subset of a diffusers pipeline used by DiffusersProvider."""

    def __init__(self, components: dict | None = None):
        self.components = components or {}
        self._execution_device = torch.device("cpu")
        self.calls: list[dict] = []
        self.encode_calls = 0

    def encode_prompt(self, prompt, num_images_per_prompt=1, **kwargs):
        self.encode_calls += 1
        embeds = torch.full((1, 4, 3), float(len(prompt)))
        return embeds, None  # second output None, like Flux' text_ids may be / Qwen's mask

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if self.components.get("text_encoder", "present") is None:
            # In stage 2 encode_prompt must have been patched: calling the real one would need the encoder.
            embeds, _ = self.encode_prompt(kwargs["prompt"])
            assert embeds.shape == (1, 4, 3)
        steps = kwargs["num_inference_steps"]
        cb = kwargs.get("callback_on_step_end")
        for i in range(steps):
            if cb:
                cb(self, i, i, {})
        return SimpleNamespace(images=[Image.new("RGB", (32, 32), "red")])


class FakeProvider(DiffusersProvider):
    name = "fake"
    supports_generate = True
    model_id = "fake/model"
    quantization = "none"
    offload = "none"
    default_steps = 10
    min_steps = 2
    max_steps = 20

    def __init__(self):
        super().__init__()
        self.loads: list[dict] = []
        self.pipes: list[FakePipe] = []

    def _new_pipe(self, components):
        pipe = FakePipe(components)
        self.pipes.append(pipe)
        return pipe

    def _edit_kwargs(self, req, steps):
        return {"prompt": req.prompt, "image": [req.image], "num_inference_steps": steps}

    def _generate_kwargs(self, prompt, width, height, steps):
        return {"prompt": prompt, "width": width, "height": height, "num_inference_steps": steps}

    def _encode_prompt_kwargs(self, pipe, call_kwargs):
        return {"prompt": call_kwargs["prompt"]}


@pytest.fixture
def provider(monkeypatch):
    prov = FakeProvider()

    def fake_load_pipeline(model_id, *, quantization, offload, allow_cpu_full_precision, components=None):
        prov.loads.append({"offload": offload, "components": components or {}})
        return prov._new_pipe(components or {})

    monkeypatch.setattr("app.providers.diffusers_base.load_pipeline", fake_load_pipeline)
    return prov


def test_steps_default_and_clamping(provider):
    assert provider.resolve_steps(None) == 10
    assert provider.resolve_steps(1) == 2
    assert provider.resolve_steps(999) == 20
    assert provider.resolve_steps(7) == 7


def test_off_mode_keeps_pipeline_resident(provider):
    stages = []
    opts = RunOptions(steps=3, save_gpu="off", progress=stages.append)
    provider.generate("a cat", 64, 64, 1, opts)
    provider.generate("a dog", 64, 64, 1, opts)
    assert len(provider.loads) == 1
    assert provider.loads[0]["offload"] == "none"
    assert provider.pipes[0].calls[0]["num_inference_steps"] == 3
    assert "Denoising 3/3" in stages


def test_ram_mode_uses_model_offload_and_reloads_on_switch(provider):
    provider.generate("x", 64, 64, 1, RunOptions(save_gpu="off"))
    provider.generate("x", 64, 64, 1, RunOptions(save_gpu="ram"))
    provider.generate("x", 64, 64, 1, RunOptions(save_gpu="ram"))
    assert [l["offload"] for l in provider.loads] == ["none", "model"]


def test_disk_mode_stages_and_cleans_up(provider, tmp_path, monkeypatch):
    from app.providers import diffusers_base

    monkeypatch.setattr(diffusers_base.settings, "data_dir", tmp_path)
    stages = []
    req = EditInput(
        Image.new("RGB", (64, 64)), "make it blue", options=RunOptions(steps=4, save_gpu="disk", progress=stages.append)
    )
    out = provider.edit(req)
    assert out.size == (32, 32)

    # two partial loads: text stage without transformer/vae, image stage without text_encoder
    assert len(provider.loads) == 2
    assert provider.loads[0]["components"] == {"transformer": None, "vae": None}
    assert provider.loads[1]["components"] == {"text_encoder": None}
    # the real encoder ran exactly once, in stage 1
    assert provider.pipes[0].encode_calls == 1
    # stage 2 pipeline ran denoising with the original kwargs
    assert provider.pipes[1].calls[0]["num_inference_steps"] == 4
    # nothing resident afterwards and the temp file is gone
    assert provider.pipe is None
    assert list((tmp_path / "tmp").glob("*.safetensors")) == []
    assert "Encoding prompt" in stages and "Loading transformer + VAE" in stages


def test_disk_mode_removes_tmp_file_on_failure(provider, tmp_path, monkeypatch):
    from app.providers import diffusers_base

    monkeypatch.setattr(diffusers_base.settings, "data_dir", tmp_path)

    def boom(**kwargs):
        raise RuntimeError("cuda oom")

    real_new_pipe = provider._new_pipe

    def new_pipe(components):
        pipe = real_new_pipe(components)
        if components.get("text_encoder", "present") is None:
            pipe.__call__ = boom
            pipe.__class__ = type("Boom", (FakePipe,), {"__call__": lambda self, **kw: boom(**kw)})
        return pipe

    provider._new_pipe = new_pipe
    with pytest.raises(RuntimeError, match="cuda oom"):
        provider.generate("x", 64, 64, 1, RunOptions(save_gpu="disk"))
    assert list((tmp_path / "tmp").glob("*.safetensors")) == []
