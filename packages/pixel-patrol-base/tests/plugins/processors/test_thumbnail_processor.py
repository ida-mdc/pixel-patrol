import warnings

import numpy as np
import pytest
from PIL import Image

from pixel_patrol_base.config import SPRITE_SIZE
from pixel_patrol_base.core.record import record_from
from pixel_patrol_base.plugins.processors.thumbnail_processor import ThumbnailProcessor


def _decode_thumbnail(thumbnail: bytes) -> Image.Image:
    arr = np.frombuffer(thumbnail, dtype=np.uint8).reshape(SPRITE_SIZE, SPRITE_SIZE, 4)
    return Image.fromarray(arr, mode="RGBA")


@pytest.fixture
def proc() -> ThumbnailProcessor:
    return ThumbnailProcessor()


def _run(proc, data: np.ndarray, dim_order: str) -> dict:
    record = record_from(data, {"dim_order": dim_order.upper()})
    dims = list(dim_order.upper())
    row = proc.run_chunk(record)
    # stamp spatial position metadata (normally done by the pipeline)
    if row and "__thumbnail_patch__" in row:
        for i, d in enumerate(dims):
            row[f"dim_{d.lower()}"] = 0
            row[f"{d}_size"] = data.shape[i]
    rows = [row]
    result = {}
    for name in proc.OUTPUT_SCHEMA:
        fn = proc.get_aggregation(name)
        if fn:
            val = fn(rows, ())
            if val is not None:
                result[name] = val
    return result


def _run_chunk_with_origin(proc, data: np.ndarray, origin: list, dim_order: str) -> dict:
    """Run a single chunk with explicit origin — simulates pipeline coordinate stamping."""
    record = record_from(data, {"dim_order": dim_order.upper()})
    dims = list(dim_order.upper())
    row = proc.run_chunk(record)
    if row and "__thumbnail_patch__" in row:
        for i, d in enumerate(dims):
            row[f"dim_{d.lower()}"] = origin[i]
            row[f"{d}_size"] = data.shape[i]
    return row


def _thumbnail(proc, data, dim_order) -> bytes:
    return _run(proc, data, dim_order)["thumbnail"]


def _aggregate(proc, rows) -> dict:
    """Simulate pipeline aggregation: call get_aggregation for every output column."""
    result = {}
    for name in proc.OUTPUT_SCHEMA:
        fn = proc.get_aggregation(name)
        if fn:
            val = fn(rows, ())
            if val is not None:
                result[name] = val
    return result


# ---------------------------------------------------------------------------
# Processor contract
# ---------------------------------------------------------------------------

def test_chunk_kind(proc):
    from pixel_patrol_base.core.contracts import ChunkKind
    assert proc.CHUNK_KIND is ChunkKind.MEMORY


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
    result = _run(proc, data, "YX")
    assert result == {}


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


def test_channel_dimension_2ch_kept_as_color(proc):
    """2 channels kept as R and G; channel 1 (G) is the bright one."""
    data = np.zeros((2, 20, 20), dtype=np.uint8)
    data[1] = 200
    thumb = _thumbnail(proc, data, "CYX")
    arr = np.array(_decode_thumbnail(thumb))
    # channel 0 → R = 0, channel 1 → G = 255
    assert arr[SPRITE_SIZE // 2, SPRITE_SIZE // 2, 1] == 255


def test_s_dimension_2ch_kept_as_color(proc):
    """S with 2 channels kept as color (not averaged)."""
    data = np.zeros((2, 20, 20), dtype=np.float32)
    data[1] = 1.0
    thumb = _thumbnail(proc, data, "SYX")
    arr = np.array(_decode_thumbnail(thumb))
    # S channels map to R, G: channel 1 (G) is bright
    assert arr[SPRITE_SIZE // 2, SPRITE_SIZE // 2, 1] == 255


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


def test_c_dimension_with_3_channels_produces_color(proc):
    data = np.zeros((3, 64, 64), dtype=np.uint8)
    data[0] = 100; data[1] = 150; data[2] = 200
    thumb = _thumbnail(proc, data, "CYX")
    arr = np.array(_decode_thumbnail(thumb))
    # With 3 channels kept as colour, RGB channels are not equal
    assert not np.array_equal(arr[:, :, 0], arr[:, :, 1])


def test_c_dimension_mean_when_more_than_4_channels(proc):
    data = np.random.randint(0, 256, (5, 20, 20), dtype=np.uint8)
    thumb = _thumbnail(proc, data, "CYX")
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


def test_wide_image_letterboxed(proc):
    """A 1×256 image fills the full canvas width and is letterboxed top/bottom."""
    data = np.full((1, 256), 200, dtype=np.uint8)
    thumb = _thumbnail(proc, data, "YX")
    alpha = np.array(_decode_thumbnail(thumb))[:, :, 3]
    opaque_rows = np.where(alpha.any(axis=1))[0]
    # exactly one row of content, all columns opaque in that row
    assert len(opaque_rows) == 1
    assert alpha[opaque_rows[0], :].all()
    # top and bottom edges are transparent (letterbox)
    assert alpha[0, SPRITE_SIZE // 2] == 0
    assert alpha[-1, SPRITE_SIZE // 2] == 0


def test_tall_image_pillarboxed(proc):
    """A 256×1 image fills the full canvas height and is pillarboxed left/right."""
    data = np.full((256, 1), 200, dtype=np.uint8)
    thumb = _thumbnail(proc, data, "YX")
    alpha = np.array(_decode_thumbnail(thumb))[:, :, 3]
    opaque_cols = np.where(alpha.any(axis=0))[0]
    # exactly one column of content, all rows opaque in that column
    assert len(opaque_cols) == 1
    assert alpha[:, opaque_cols[0]].all()
    # left and right edges are transparent (pillarbox)
    assert alpha[SPRITE_SIZE // 2, 0] == 0
    assert alpha[SPRITE_SIZE // 2, -1] == 0


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


# ---------------------------------------------------------------------------
# Multi-chunk assembly
# ---------------------------------------------------------------------------

def test_multi_z_chunks_assembled_via_aggregation(proc):
    """Multiple Z slices (shape 1×Y×X each) assembled into one thumbnail."""
    rows = []
    for z in range(3):
        data = np.full((1, 20, 20), z * 100, dtype=np.uint8)
        rows.append(_run_chunk_with_origin(proc, data, [z, 0, 0], "ZYX"))
    result = _aggregate(proc, rows)
    assert "thumbnail" in result
    assert len(result["thumbnail"]) == SPRITE_SIZE * SPRITE_SIZE * 4
    arr = np.frombuffer(result["thumbnail"], dtype=np.uint8).reshape(SPRITE_SIZE, SPRITE_SIZE, 4)
    assert arr[:, :, 3].min() == 255  # full canvas covered — all patches placed


def test_multi_xy_chunks_cover_full_canvas(proc):
    """Four XY tiles assembled: all four quadrants have alpha=255."""
    rows = []
    for y_off, x_off, val in [(0, 0, 50), (0, 20, 100), (20, 0, 150), (20, 20, 200)]:
        data = np.full((20, 20), val, dtype=np.uint8)
        rows.append(_run_chunk_with_origin(proc, data, [y_off, x_off], "YX"))
    result = _aggregate(proc, rows)
    assert "thumbnail" in result
    arr = np.frombuffer(result["thumbnail"], dtype=np.uint8).reshape(SPRITE_SIZE, SPRITE_SIZE, 4)
    half = SPRITE_SIZE // 2
    assert arr[:half, :half,   3].min() == 255, "top-left quadrant empty"
    assert arr[:half, half:,   3].min() == 255, "top-right quadrant empty"
    assert arr[half:, :half,   3].min() == 255, "bottom-left quadrant empty"
    assert arr[half:, half:,   3].min() == 255, "bottom-right quadrant empty"


def test_numpy_array_input(proc):
    """Processor must work when record.data is a plain numpy array (e.g. LMDB loader path)."""
    data = np.random.randint(0, 256, (20, 20, 3), dtype=np.uint8)
    result = _run(proc, data, "YXS")
    assert result["thumbnail"] is not None
    assert len(result["thumbnail"]) == SPRITE_SIZE * SPRITE_SIZE * 4


def test_helper_columns_not_aggregated(proc):
    """get_aggregation returns None for internal __*__ patch columns."""
    data = np.ones((20, 20), dtype=np.uint8) * 128
    record = record_from(data, {"dim_order": "YX"})
    row = proc.run_chunk(record)
    for key in row:
        if key.startswith("__"):
            assert proc.get_aggregation(key) is None, f"helper column has aggregation: {key}"
