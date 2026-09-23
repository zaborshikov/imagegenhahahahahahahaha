from PIL import Image, ImageDraw

from app.image_ops import bbox_from_mask, composite_local, ensure_min_crop, expand_box, normalize_mask


def test_bbox_and_expand():
    mask = Image.new("L", (100, 80), 0)
    ImageDraw.Draw(mask).rectangle((20, 10, 39, 29), fill=255)
    box = bbox_from_mask(mask)
    assert box is not None
    assert box.pil() == (20, 10, 40, 30)
    assert expand_box(box, 8, (100, 80)).pil() == (12, 2, 48, 38)


def test_strict_composite_keeps_outside():
    base = Image.new("RGB", (60, 60), "white")
    edited = Image.new("RGB", (30, 30), "black")
    mask = Image.new("L", (60, 60), 0)
    ImageDraw.Draw(mask).rectangle((20, 20, 39, 39), fill=255)
    box = expand_box(bbox_from_mask(mask), 5, base.size)
    out = composite_local(base, edited, mask, box, feather=0)
    assert out.getpixel((5, 5)) == (255, 255, 255)
    assert out.getpixel((30, 30)) == (0, 0, 0)


def test_ensure_min_crop_stays_in_bounds():
    from app.image_ops import Box
    box = ensure_min_crop(Box(0, 0, 20, 20), (100, 80), 64)
    assert box.left >= 0 and box.top >= 0
    assert box.right <= 100 and box.bottom <= 80
    assert box.width == 64 and box.height == 64
