import numpy as np
import pytest

from pixel_patrol_base.core.record import record_from
from pixel_patrol_aqqua.plugins.processors.scale_bar_processor import ScaleBarProcessor


@pytest.fixture
def proc():
    return ScaleBarProcessor()


def _chunk(proc, arr, dim_order):
    record = record_from(arr, {"dim_order": dim_order.upper()})
    return proc.run_chunk(record)


def _image_with_bar(h=120, w=160, bar_value=255, bar_frac=0.3, thickness=3, pos_frac=0.92, seed=0):
    rng = np.random.default_rng(seed)
    img = rng.integers(60, 180, size=(h, w)).astype(np.uint8)
    bar_len = int(w * bar_frac)
    start = (w - bar_len) // 2
    img[int(h * pos_frac):int(h * pos_frac) + thickness, start:start + bar_len] = bar_value
    return img


def test_detects_bar(proc):
    # 2D image and a CYX stack where only one channel has a bar - both must fire.
    img = _image_with_bar()
    assert _chunk(proc, img, "YX")["scale_bar_score"] >= 0.2
    stack = np.stack([np.random.default_rng(2).integers(60, 180, img.shape).astype(np.uint8), img])
    assert _chunk(proc, stack, "CYX")["scale_bar_score"] >= 0.2


@pytest.mark.parametrize("data", [
    np.random.default_rng(1).integers(60, 180, (120, 160)).astype(np.uint8),
    np.tile(np.linspace(0, 255, 160, dtype=np.float32), (120, 1)),
    np.full((120, 160), 128, dtype=np.uint8),
], ids=["noise", "horizontal_gradient", "flat"])
def test_no_false_positive_on_simple_patterns(proc, data):
    assert _chunk(proc, data, "YX")["scale_bar_score"] == pytest.approx(0.0, abs=1e-6)


def test_ignores_bar_outside_border(proc):
    assert _chunk(proc, _image_with_bar(pos_frac=0.5), "YX")["scale_bar_score"] == pytest.approx(0.0, abs=1e-6)


def test_no_false_positive_on_frame_filling_object(proc):
    # Regression: a disc nearly filling the frame previously tripped the detector.
    h, w = 160, 160
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dist = np.sqrt((yy - h / 2) ** 2 + (xx - w / 2) ** 2)
    data = np.where(dist <= min(h, w) * 0.46, 60.0, 180.0).astype(np.uint8)
    assert _chunk(proc, data, "YX")["scale_bar_score"] == pytest.approx(0.0, abs=1e-6)


def test_small_image_returns_nan(proc):
    assert np.isnan(_chunk(proc, np.ones((4, 4), dtype=np.float32), "YX").get("scale_bar_score", np.nan))


def test_uses_full_image_width_not_chunk_width(proc):
    # 15/40 = 37.5% of this chunk but 15/400 = 3.75% of the real image - below minimum bar fraction.
    h, tile_w, full_w = 120, 40, 400
    tile = np.random.default_rng(0).integers(60, 180, (h, tile_w)).astype(np.uint8)
    tile[110:113, 5:20] = 255
    row = proc.run_chunk(record_from(tile, {"dim_order": "YX", "full_shape": (h, full_w)}))
    assert row["scale_bar_score"] == pytest.approx(0.0, abs=1e-6)
