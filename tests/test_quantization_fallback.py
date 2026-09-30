import pytest
import torch

from app.providers import loading


@pytest.fixture
def no_bitsandbytes(monkeypatch):
    monkeypatch.setattr(loading, "bitsandbytes_status", lambda: (False, "bitsandbytes is not installed"))


@pytest.fixture
def with_bitsandbytes(monkeypatch):
    monkeypatch.setattr(loading, "bitsandbytes_status", lambda: (True, None))


def test_none_is_passthrough(no_bitsandbytes):
    assert loading.resolve_quantization("none", model_id="m") == "none"
    assert loading.effective_quantization("none") == "none"


def test_fp8_does_not_need_bitsandbytes(no_bitsandbytes):
    assert loading.resolve_quantization("fp8", model_id="m") == "fp8"
    assert loading.effective_quantization("fp8") == "fp8"


def test_fp8_falls_back_without_float8(monkeypatch):
    monkeypatch.setattr(loading, "fp8_supported", lambda: False)
    assert loading.resolve_quantization("fp8", model_id="m") == "none"
    assert loading.effective_quantization("fp8") == "none"


def test_bnb_kept_when_available(with_bitsandbytes):
    assert loading.resolve_quantization("nf4", model_id="m") == "nf4"
    assert loading.effective_quantization("int8") == "int8"


def test_bnb_cuda_falls_back_to_full_precision(no_bitsandbytes, monkeypatch):
    monkeypatch.setattr(loading.torch.cuda, "is_available", lambda: True)
    assert loading.resolve_quantization("nf4", model_id="m") == "none"
    assert loading.effective_quantization("nf4") == "none"


def test_bnb_cpu_raises_actionable_error(no_bitsandbytes, monkeypatch):
    monkeypatch.setattr(loading.torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="bitsandbytes"):
        loading.resolve_quantization("nf4", model_id="m")


def test_apply_fp8_stores_weights_as_float8():
    model = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.LayerNorm(8))
    loading.apply_fp8(model, torch.float32, name="test")
    assert model[0].weight.dtype == torch.float8_e4m3fn
    # norms are skipped and stay in the compute dtype
    assert model[1].weight.dtype == torch.float32
    out = model(torch.randn(2, 8))
    assert out.dtype == torch.float32


def test_health_reports_providers_and_quantization():
    from fastapi.testclient import TestClient

    from app.main import app

    body = TestClient(app).get("/api/health").json()
    names = {p["name"] for p in body["providers"]}
    assert names == {"qwen21", "flux"}
    assert "qwen" not in names
    for p in body["providers"]:
        assert p["quantization"]["effective"] in {"none", "fp8", "nf4", "int8"}
        assert p["steps"]["min"] <= p["steps"]["default"] <= p["steps"]["max"]
    assert set(body["capabilities"]["generate"]) == {"qwen21", "flux"}
    assert body["save_gpu_modes"] == ["off", "ram", "disk"]
