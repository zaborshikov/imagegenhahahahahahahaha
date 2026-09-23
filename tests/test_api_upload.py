from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from app.main import app


def make_png(size=(96, 64)) -> bytes:
    image = Image.new("RGB", size, (12, 34, 56))
    buf = BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def test_session_upload_and_media_roundtrip():
    client = TestClient(app)
    created = client.post("/api/sessions")
    assert created.status_code == 200
    sid = created.json()["session_id"]

    uploaded = client.post(
        f"/api/sessions/{sid}/upload",
        files={"image": ("photo.png", make_png(), "image/png")},
    )
    assert uploaded.status_code == 200, uploaded.text
    payload = uploaded.json()
    assert payload["width"] == 96
    assert payload["height"] == 64
    assert payload["image_url"].startswith("/media/")

    media = client.get(payload["image_url"])
    assert media.status_code == 200
    assert media.headers["content-type"].startswith("image/")
    assert len(media.content) > 0
