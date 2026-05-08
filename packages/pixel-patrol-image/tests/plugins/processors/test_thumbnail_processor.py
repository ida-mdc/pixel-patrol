import warnings

import numpy as np
import dask.array as da
import pytest
from PIL import Image

from pixel_patrol_base.core.record import record_from
from pixel_patrol_base.config import SPRITE_SIZE
from pixel_patrol_image.plugins.processors.thumbnail_processor import ThumbnailProcessor


def _decode_thumbnail(thumbnail: bytes) -> Image.Image:
    arr = np.frombuffer(thumbnail, dtype=np.uint8).reshape(SPRITE_SIZE, SPRITE_SIZE, 4)
    return Image.fromarray(arr, mode="RGBA")


@pytest.fixture
def proc() -> ThumbnailProcessor:
    return ThumbnailProcessor()


def _run(proc, data: np.ndarray, dim_order: str, meta: dict | None = None) -> dict:
    m = {"dim_order": dim_order, **(meta or {})}
    return proc.run(record_from(da.from_array(data), m))


def _thumbnail(proc, data, dim_order, meta=None) -> bytes:
    return _run(proc, data, dim_order, meta)["thumbnail"]


# ---------------------------------------------------------------------------
# Processor contract
# ---------------------------------------------------------------------------

def test_global_only_flag(proc):
    assert proc.GLOBAL_ONLY is True


def test_run_returns_dict(proc):
    data = np.random.randint(0, 256, (32, 32), dtype=np.uint8)
    result = _run(proc, data, "YX")
    assert isinstance(result, dict)
    for k in ("thumbnail", "thumbnail_norm_min", "thumbnail_norm_max", "thumbnail_dtype"):
        assert k in result, f"Missing key: {k}"


# ---------------------------------------------------------------------------
# Output format
# ---------------------------------------------------------------------------

def test_output_format(proc):
    data = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
    result = _run(proc, data, "YX")
    thumb = result["thumbnail"]
    assert isinstance(thumb, bytes)
    assert len(thumb) == SPRITE_SIZE * SPRITE_SIZE * 4
    img = _decode_thumbnail(thumb)
    assert img.size == (SPRITE_SIZE, SPRITE_SIZE)
    assert img.mode == "RGBA"


def test_output_includes_norm_fields(proc):
    data = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
    result = _run(proc, data, "YX")
    assert isinstance(result["thumbnail_norm_min"], float)
    assert isinstance(result["thumbnail_norm_max"], float)
    assert isinstance(result["thumbnail_dtype"], str)


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def test_grayscale_normalization_min_to_zero_max_to_255(proc):
    data = np.zeros((64, 64), dtype=np.float32)
    data[:, 32:] = 1.0
    thumb = _thumbnail(proc, data, "YX")
    arr = np.array(_decode_thumbnail(thumb))[:, :, 0]
    assert arr[32, 16] == 0
    assert arr[32, 48] == 255


def test_normalization_zero_floor_when_min_positive(proc):
    data = np.zeros((64, 64), dtype=np.float32)
    data[:, 32:] = 100.0
    thumb = _thumbnail(proc, data, "YX")
    arr = np.array(_decode_thumbnail(thumb))[:, :, 0]
    assert arr[32, 16] == 0
    assert arr[32, 48] == 255


def test_normalization_negative_values(proc):
    data = np.zeros((64, 64), dtype=np.float32)
    data[:, :32] = -50.0
    data[:, 32:] =  50.0
    thumb = _thumbnail(proc, data, "YX")
    arr = np.array(_decode_thumbnail(thumb))[:, :, 0]
    assert arr[32, 16] == 0
    assert arr[32, 48] == 255


def test_normalization_ignores_nan_for_range(proc):
    data = np.full((64, 64), np.nan, dtype=np.float32)
    data[:, 32:] = 0.0
    data[:, 48:] = 1.0
    thumb = _thumbnail(proc, data, "YX")
    arr = np.array(_decode_thumbnail(thumb))[:, :, 0]
    assert arr[32, 16] == 0
    assert arr[32, 48] == 255


def test_all_nan_image_yields_black_thumbnail(proc):
    data = np.full((32, 32), np.nan, dtype=np.float32)
    result = _run(proc, data, "YX")
    assert result["thumbnail_norm_min"] == 0.0
    assert result["thumbnail_norm_max"] == 0.0
    arr = np.array(_decode_thumbnail(result["thumbnail"]))
    assert arr[:, :, 0].max() == 0


def test_uint8_grayscale_is_normalized(proc):
    data = np.full((50, 50), 128, dtype=np.uint8)
    thumb = _thumbnail(proc, data, "YX")
    arr = np.array(_decode_thumbnail(thumb))[:, :, 0]
    assert arr.max() == 255


def test_boolean_image_values(proc):
    data = np.zeros((64, 64), dtype=bool)
    data[:, 32:] = True
    thumb = _thumbnail(proc, data, "YX")
    arr = np.array(_decode_thumbnail(thumb))[:, :, 0]
    assert arr[32, 16] == 0
    assert arr[32, 48] == 255


def test_grayscale_thumbnail_has_equal_rgb_channels(proc):
    data = np.random.randint(0, 256, (20, 20), dtype=np.uint8)
    thumb = _thumbnail(proc, data, "YX")
    arr = np.array(_decode_thumbnail(thumb))
    np.testing.assert_array_equal(arr[:, :, 0], arr[:, :, 1])
    np.testing.assert_array_equal(arr[:, :, 0], arr[:, :, 2])


def test_empty_image(proc):
    data = np.array([[]], dtype=np.uint8)
    result = proc.run(record_from(da.from_array(data, chunks=(1, 1)), {"dim_order": "YX"}))
    assert result["thumbnail"] is None


def test_single_pixel(proc):
    data = np.array([[42]], dtype=np.uint8)
    thumb = _thumbnail(proc, data, "YX")
    assert len(thumb) == SPRITE_SIZE * SPRITE_SIZE * 4


# ---------------------------------------------------------------------------
# Dimension reduction
# ---------------------------------------------------------------------------

def test_z_dimension_center_slice(proc):
    nz = 5
    data = np.zeros((nz, 20, 20), dtype=np.uint8)
    data[nz // 2] = 200
    thumb = _thumbnail(proc, data, "ZYX")
    arr = np.array(_decode_thumbnail(thumb))[:, :, 0]
    assert arr[SPRITE_SIZE // 2, SPRITE_SIZE // 2] == 255


def test_channel_dimension_mean(proc):
    data = np.zeros((2, 20, 20), dtype=np.uint8)
    data[1] = 200
    thumb = _thumbnail(proc, data, "CYX")
    arr = np.array(_decode_thumbnail(thumb))[:, :, 0]
    assert arr[SPRITE_SIZE // 2, SPRITE_SIZE // 2] == 255


def test_s_dimension_mean_when_non_rgb(proc):
    data = np.zeros((2, 20, 20), dtype=np.float32)
    data[1] = 1.0
    thumb = _thumbnail(proc, data, "SYX")
    arr = np.array(_decode_thumbnail(thumb))[:, :, 0]
    assert arr[SPRITE_SIZE // 2, SPRITE_SIZE // 2] == 255


def test_multiple_dimensions_center_slice_t_z_mean_c(proc):
    nt, nc, nz = 3, 2, 5
    data = np.zeros((nt, nc, nz, 20, 20), dtype=np.uint8)
    data[nt // 2, :, nz // 2] = 200
    thumb = _thumbnail(proc, data, "TCZYX")
    arr = np.array(_decode_thumbnail(thumb))[:, :, 0]
    assert arr[SPRITE_SIZE // 2, SPRITE_SIZE // 2] == 255


# ---------------------------------------------------------------------------
# RGB / colour thumbnails
# ---------------------------------------------------------------------------

def test_rgb_via_s_dimension_normalized(proc):
    data = np.full((64, 64, 3), [100, 150, 200], dtype=np.uint8)
    thumb = _thumbnail(proc, data, "YXS")
    arr = np.array(_decode_thumbnail(thumb))
    np.testing.assert_array_equal(arr[SPRITE_SIZE // 2, SPRITE_SIZE // 2, :3], [127, 191, 255])


def test_rgba_via_s_dimension_drops_alpha(proc):
    data = np.full((64, 64, 4), [100, 150, 200, 128], dtype=np.uint8)
    thumb = _thumbnail(proc, data, "YXS")
    arr = np.array(_decode_thumbnail(thumb))
    np.testing.assert_array_equal(arr[SPRITE_SIZE // 2, SPRITE_SIZE // 2, :3], [127, 191, 255])


def test_rgb_via_c_dimension_normalized(proc):
    data = np.zeros((3, 64, 64), dtype=np.uint8)
    data[0] = 100
    data[1] = 150
    data[2] = 200
    thumb = _thumbnail(proc, data, "CYX", meta={"channel_names": ["R", "G", "B"]})
    arr = np.array(_decode_thumbnail(thumb))
    np.testing.assert_array_equal(arr[SPRITE_SIZE // 2, SPRITE_SIZE // 2, :3], [127, 191, 255])


def test_non_rgb_c_dimension_produces_grayscale(proc):
    data = np.random.randint(0, 256, (3, 20, 20), dtype=np.uint8)
    thumb = _thumbnail(proc, data, "CYX", meta={"channel_names": ["DAPI", "GFP", "mCherry"]})
    arr = np.array(_decode_thumbnail(thumb))
    np.testing.assert_array_equal(arr[:, :, 0], arr[:, :, 1])
    np.testing.assert_array_equal(arr[:, :, 0], arr[:, :, 2])


def test_non_uint8_s_dimension_renders_as_color(proc):
    data = np.zeros((20, 20, 3), dtype=np.float32)
    data[:, :, 0] = 1.0
    data[:, :, 1] = 0.5
    data[:, :, 2] = 0.0
    thumb = _thumbnail(proc, data, "YXS")
    arr = np.array(_decode_thumbnail(thumb))
    assert not np.array_equal(arr[:, :, 0], arr[:, :, 1])


def test_uint16_rgb_via_s_dimension(proc):
    data = np.full((64, 64, 3), [100, 150, 200], dtype=np.uint16)
    thumb = _thumbnail(proc, data, "YXS")
    arr = np.array(_decode_thumbnail(thumb))
    np.testing.assert_array_equal(arr[SPRITE_SIZE // 2, SPRITE_SIZE // 2, :3], [127, 191, 255])


# ---------------------------------------------------------------------------
# Aspect ratio / padding
# ---------------------------------------------------------------------------

def test_square_image_fills_thumbnail(proc):
    data = np.full((64, 64), 200, dtype=np.uint8)
    thumb = _thumbnail(proc, data, "YX")
    alpha = np.array(_decode_thumbnail(thumb))[:, :, 3]
    assert alpha[0, SPRITE_SIZE // 2] == 255
    assert alpha[-1, SPRITE_SIZE // 2] == 255
    assert alpha[SPRITE_SIZE // 2, 0] == 255
    assert alpha[SPRITE_SIZE // 2, -1] == 255


def test_wide_image_has_letterbox_rows(proc):
    data = np.full((1, 256), 200, dtype=np.uint8)
    thumb = _thumbnail(proc, data, "YX")
    alpha = np.array(_decode_thumbnail(thumb))[:, :, 3]
    assert alpha[0, SPRITE_SIZE // 2] == 0


def test_tall_image_has_pillarbox_cols(proc):
    data = np.full((256, 1), 200, dtype=np.uint8)
    thumb = _thumbnail(proc, data, "YX")
    alpha = np.array(_decode_thumbnail(thumb))[:, :, 3]
    assert alpha[SPRITE_SIZE // 2, 0] == 0


# ---------------------------------------------------------------------------
# Norm metadata
# ---------------------------------------------------------------------------

def test_norm_min_is_zero_when_data_min_positive(proc):
    data = np.full((20, 20), 50.0, dtype=np.float32)
    result = _run(proc, data, "YX")
    assert result["thumbnail_norm_min"] == 0.0


def test_norm_max_equals_data_max(proc):
    data = np.full((20, 20), 42.0, dtype=np.float32)
    result = _run(proc, data, "YX")
    assert result["thumbnail_norm_max"] == pytest.approx(42.0)


def test_norm_dtype_reflects_input(proc):
    data = np.zeros((10, 10), dtype=np.uint16)
    result = _run(proc, data, "YX")
    assert result["thumbnail_dtype"] == "uint16"


# ---------------------------------------------------------------------------
# NaN handling
# ---------------------------------------------------------------------------

def test_nan_handling(proc):
    data = np.array([[0, 1, 2, np.nan]], dtype=np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _run(proc, data, "YX")
