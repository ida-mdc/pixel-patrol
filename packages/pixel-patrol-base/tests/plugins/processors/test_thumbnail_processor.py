import warnings

import numpy as np
import dask.array as da
import pytest
from PIL import Image

from pixel_patrol_base.core.record import record_from
from pixel_patrol_base.plugins.processors.thumbnail_processor import ThumbnailProcessor
from pixel_patrol_base.config import SPRITE_SIZE


def _decode_thumbnail(thumbnail: bytes) -> Image.Image:
    """Decode a raw RGBA thumbnail (SPRITE_SIZE × SPRITE_SIZE × 4) to a PIL Image."""
    arr = np.frombuffer(thumbnail, dtype=np.uint8).reshape(SPRITE_SIZE, SPRITE_SIZE, 4)
    return Image.fromarray(arr, mode="RGBA")


class TestThumbnailProcessor:
    """Test suite for ThumbnailProcessor."""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _run(self, data: np.ndarray, dim_order: str, meta: dict | None = None):
        m = {"dim_order": dim_order, **(meta or {})}
        record = record_from(da.from_array(data), m)
        rows = ThumbnailProcessor().run(record)
        assert rows[0]["obs_level"] == 0
        return rows[0]

    def _thumbnail(self, data, dim_order, meta=None) -> bytes:
        return self._run(data, dim_order, meta)["thumbnail"]

    # ------------------------------------------------------------------
    # Output format
    # ------------------------------------------------------------------

    def test_output_format(self):
        data = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
        result = self._run(data, "YX")
        thumb = result["thumbnail"]
        assert isinstance(thumb, bytes)
        assert len(thumb) == SPRITE_SIZE * SPRITE_SIZE * 4  # raw RGBA, fixed size
        img = _decode_thumbnail(thumb)
        assert img.size == (SPRITE_SIZE, SPRITE_SIZE)
        assert img.mode == "RGBA"

    def test_output_includes_norm_fields(self):
        data = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
        result = self._run(data, "YX")
        assert "thumbnail_norm_min" in result
        assert "thumbnail_norm_max" in result
        assert "thumbnail_dtype"    in result
        assert isinstance(result["thumbnail_norm_min"], float)
        assert isinstance(result["thumbnail_norm_max"], float)
        assert isinstance(result["thumbnail_dtype"], str)

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    def test_grayscale_normalization_min_to_zero_max_to_255(self):
        """Min pixel maps to 0 and max pixel maps to 255 (float32 input)."""
        data = np.zeros((64, 64), dtype=np.float32)
        data[:, 32:] = 1.0
        thumb = self._thumbnail(data, "YX")
        arr = np.array(_decode_thumbnail(thumb))[:, :, 0]  # R channel
        assert arr[32, 16] == 0    # min → 0
        assert arr[32, 48] == 255  # max → 255

    def test_normalization_zero_floor_when_min_positive(self):
        """When all values > 0, lower bound is 0 (not the data min), so zero maps to 0."""
        data = np.zeros((64, 64), dtype=np.float32)
        data[:, 32:] = 100.0  # left half = 0, right half = 100
        # lower = min(0, 0) = 0, upper = 100
        # 0 → 0, 100 → 255
        thumb = self._thumbnail(data, "YX")
        arr = np.array(_decode_thumbnail(thumb))[:, :, 0]
        assert arr[32, 16] == 0    # zero stays zero
        assert arr[32, 48] == 255  # max → 255

    def test_normalization_negative_values(self):
        """When data has negative values, lower bound = data_min (< 0)."""
        data = np.zeros((64, 64), dtype=np.float32)
        data[:, :32]  = -50.0  # left half
        data[:, 32:]  =  50.0  # right half
        # lower = -50, upper = 50 → range 100 → 0 maps to 127.5, ±50 maps to 0/255
        thumb = self._thumbnail(data, "YX")
        arr = np.array(_decode_thumbnail(thumb))[:, :, 0]
        assert arr[32, 16] == 0    # min (-50) → 0
        assert arr[32, 48] == 255  # max (+50) → 255

    def test_normalization_ignores_nan_for_range(self):
        """NaN pixels must not force the whole thumbnail to NaN / invalid uint8."""
        data = np.full((64, 64), np.nan, dtype=np.float32)
        data[:, 32:] = 0.0
        data[:, 48:] = 1.0
        thumb = self._thumbnail(data, "YX")
        arr = np.array(_decode_thumbnail(thumb))[:, :, 0]
        assert arr[32, 16] == 0    # min (0) → 0
        assert arr[32, 48] == 255  # max (1) → 255

    def test_all_nan_image_yields_black_thumbnail(self):
        data = np.full((32, 32), np.nan, dtype=np.float32)
        result = self._run(data, "YX")
        assert result["thumbnail_norm_min"] == 0.0
        assert result["thumbnail_norm_max"] == 0.0
        arr = np.array(_decode_thumbnail(result["thumbnail"]))
        assert arr[:, :, 0].max() == 0  # RGB black (content region still opaque)

    def test_uint8_grayscale_is_normalized(self):
        """uint8 grayscale is also normalized (not passed through as-is)."""
        data = np.full((50, 50), 128, dtype=np.uint8)
        # lower = min(128, 0) = 0, upper = 128 → constant 128 / 128 * 255 = 255
        thumb = self._thumbnail(data, "YX")
        arr = np.array(_decode_thumbnail(thumb))[:, :, 0]
        assert arr.max() == 255

    def test_boolean_image_values(self):
        """Boolean: False→0, True→255 after normalization."""
        data = np.zeros((64, 64), dtype=bool)
        data[:, 32:] = True
        thumb = self._thumbnail(data, "YX")
        arr = np.array(_decode_thumbnail(thumb))[:, :, 0]
        assert arr[32, 16] == 0    # False → 0
        assert arr[32, 48] == 255  # True → 255

    def test_grayscale_thumbnail_has_equal_rgb_channels(self):
        """Grayscale images are stored as RGB with R == G == B."""
        data = np.random.randint(0, 256, (20, 20), dtype=np.uint8)
        thumb = self._thumbnail(data, "YX")
        arr = np.array(_decode_thumbnail(thumb))
        np.testing.assert_array_equal(arr[:, :, 0], arr[:, :, 1])
        np.testing.assert_array_equal(arr[:, :, 0], arr[:, :, 2])

    def test_empty_image(self):
        data = np.array([[]], dtype=np.uint8)
        dask_data = da.from_array(data, chunks=(1, 1))
        record = record_from(dask_data, {"dim_order": "YX"})
        result = ThumbnailProcessor().run(record)
        assert result["thumbnail"] is None

    def test_single_pixel(self):
        data = np.array([[42]], dtype=np.uint8)
        thumb = self._thumbnail(data, "YX")
        assert len(thumb) == SPRITE_SIZE * SPRITE_SIZE * 4

    # ------------------------------------------------------------------
    # Dimension reduction
    # ------------------------------------------------------------------

    def test_z_dimension_center_slice(self):
        """Z: center slice (index nz//2) is selected, not first or last."""
        nz = 5
        data = np.zeros((nz, 20, 20), dtype=np.uint8)
        data[nz // 2] = 200  # only center z-slice is bright
        # constant 200 (all > 0) → lower=0, upper=200 → normalized to 255
        thumb = self._thumbnail(data, "ZYX")
        arr = np.array(_decode_thumbnail(thumb))[:, :, 0]
        assert arr[SPRITE_SIZE // 2, SPRITE_SIZE // 2] == 255

    def test_channel_dimension_mean(self):
        """Non-RGB C channels are collapsed by mean, not center slice."""
        # ch0 = 0, ch1 = 200 → mean = 100 (nonzero constant) → lower=0, upper=100 → 255
        # center slice would pick ch0 = 0 → lower=0, upper=0 → 0
        data = np.zeros((2, 20, 20), dtype=np.uint8)
        data[1] = 200
        thumb = self._thumbnail(data, "CYX")
        arr = np.array(_decode_thumbnail(thumb))[:, :, 0]
        assert arr[SPRITE_SIZE // 2, SPRITE_SIZE // 2] == 255

    def test_s_dimension_mean_when_non_rgb(self):
        """Non-uint8 S channels (no rgb capability) are meaned and normalized."""
        data = np.zeros((2, 20, 20), dtype=np.float32)
        data[1] = 1.0  # mean = 0.5, lower=0, upper=0.5 → normalized to 255
        thumb = self._thumbnail(data, "SYX")
        arr = np.array(_decode_thumbnail(thumb))[:, :, 0]
        assert arr[SPRITE_SIZE // 2, SPRITE_SIZE // 2] == 255

    def test_multiple_dimensions_center_slice_t_z_mean_c(self):
        """TCZYX: center T/Z slices and mean over C produce correct pixel values."""
        nt, nc, nz = 3, 2, 5
        data = np.zeros((nt, nc, nz, 20, 20), dtype=np.uint8)
        data[nt // 2, :, nz // 2] = 200  # signal only at center T and Z, all C
        thumb = self._thumbnail(data, "TCZYX")
        arr = np.array(_decode_thumbnail(thumb))[:, :, 0]
        assert arr[SPRITE_SIZE // 2, SPRITE_SIZE // 2] == 255

    # ------------------------------------------------------------------
    # RGB / color thumbnails
    # ------------------------------------------------------------------

    def test_rgb_via_s_dimension_normalized(self):
        """S=3 uint8 → rgb:S capability → normalized like any other image (lower=0, upper=max)."""
        # [100, 150, 200] with lower=0, upper=200 → [127, 191, 255]
        data = np.full((64, 64, 3), [100, 150, 200], dtype=np.uint8)
        thumb = self._thumbnail(data, "YXS")
        arr = np.array(_decode_thumbnail(thumb))
        np.testing.assert_array_equal(arr[SPRITE_SIZE // 2, SPRITE_SIZE // 2, :3], [127, 191, 255])

    def test_rgba_via_s_dimension_drops_alpha(self):
        """S=4 (RGBA): alpha channel dropped, RGB values normalized."""
        # [100, 150, 200] with lower=0, upper=200 → [127, 191, 255]
        data = np.full((64, 64, 4), [100, 150, 200, 128], dtype=np.uint8)
        thumb = self._thumbnail(data, "YXS")
        arr = np.array(_decode_thumbnail(thumb))
        np.testing.assert_array_equal(arr[SPRITE_SIZE // 2, SPRITE_SIZE // 2, :3], [127, 191, 255])

    def test_rgb_via_c_dimension_normalized(self):
        """C with R/G/B channel names → rgb:C → normalized like any other image (lower=0, upper=max)."""
        # [100, 150, 200] with lower=0, upper=200 → [127, 191, 255]
        data = np.zeros((3, 64, 64), dtype=np.uint8)
        data[0] = 100; data[1] = 150; data[2] = 200
        thumb = self._thumbnail(data, "CYX", meta={"channel_names": ["R", "G", "B"]})
        arr = np.array(_decode_thumbnail(thumb))
        np.testing.assert_array_equal(arr[SPRITE_SIZE // 2, SPRITE_SIZE // 2, :3], [127, 191, 255])

    def test_non_rgb_c_dimension_produces_grayscale(self):
        """C with non-RGB names → no rgb capability → R == G == B (grayscale stored as RGB)."""
        data = np.random.randint(0, 256, (3, 20, 20), dtype=np.uint8)
        thumb = self._thumbnail(data, "CYX", meta={"channel_names": ["DAPI", "GFP", "mCherry"]})
        arr = np.array(_decode_thumbnail(thumb))
        np.testing.assert_array_equal(arr[:, :, 0], arr[:, :, 1])
        np.testing.assert_array_equal(arr[:, :, 0], arr[:, :, 2])

    def test_non_uint8_s_dimension_renders_as_color(self):
        """S=3 float32 → rgb:S capability → rendered as colour (R/G/B channels differ)."""
        data = np.zeros((20, 20, 3), dtype=np.float32)
        data[:, :, 0] = 1.0   # R = max
        data[:, :, 1] = 0.5   # G = mid
        data[:, :, 2] = 0.0   # B = 0
        thumb = self._thumbnail(data, "YXS")
        arr = np.array(_decode_thumbnail(thumb))
        # Channels must differ — not grayscale
        assert not np.array_equal(arr[:, :, 0], arr[:, :, 1])

    def test_uint16_rgb_via_s_dimension(self):
        """S=3 uint16 (e.g. Sentinel-2 reflectances) → rgb:S → correct colour thumbnail."""
        # [100, 150, 200] with lower=0, upper=200 → [127, 191, 255]
        data = np.full((64, 64, 3), [100, 150, 200], dtype=np.uint16)
        thumb = self._thumbnail(data, "YXS")
        arr = np.array(_decode_thumbnail(thumb))
        np.testing.assert_array_equal(arr[SPRITE_SIZE // 2, SPRITE_SIZE // 2, :3], [127, 191, 255])

    # ------------------------------------------------------------------
    # Aspect ratio / padding
    # ------------------------------------------------------------------

    def test_square_image_fills_thumbnail(self):
        """Square image: content touches all four borders — alpha=255 everywhere."""
        data = np.full((64, 64), 200, dtype=np.uint8)
        thumb = self._thumbnail(data, "YX")
        alpha = np.array(_decode_thumbnail(thumb))[:, :, 3]
        assert alpha[0, SPRITE_SIZE // 2] == 255
        assert alpha[-1, SPRITE_SIZE // 2] == 255
        assert alpha[SPRITE_SIZE // 2, 0] == 255
        assert alpha[SPRITE_SIZE // 2, -1] == 255

    def test_wide_image_has_letterbox_rows(self):
        """Very wide image → transparent (alpha=0) padding rows top and bottom."""
        data = np.full((1, 256), 200, dtype=np.uint8)
        thumb = self._thumbnail(data, "YX")
        alpha = np.array(_decode_thumbnail(thumb))[:, :, 3]
        assert alpha[0, SPRITE_SIZE // 2] == 0

    def test_tall_image_has_pillarbox_cols(self):
        """Very tall image → transparent (alpha=0) padding cols left and right."""
        data = np.full((256, 1), 200, dtype=np.uint8)
        thumb = self._thumbnail(data, "YX")
        alpha = np.array(_decode_thumbnail(thumb))[:, :, 3]
        assert alpha[SPRITE_SIZE // 2, 0] == 0

    # ------------------------------------------------------------------
    # Norm metadata
    # ------------------------------------------------------------------

    def test_norm_min_is_zero_when_data_min_positive(self):
        """norm_min == 0 when all data values are positive."""
        data = np.full((20, 20), 50.0, dtype=np.float32)
        result = self._run(data, "YX")
        assert result["thumbnail_norm_min"] == 0.0

    def test_norm_max_equals_data_max(self):
        data = np.full((20, 20), 42.0, dtype=np.float32)
        result = self._run(data, "YX")
        assert result["thumbnail_norm_max"] == pytest.approx(42.0)

    def test_norm_dtype_reflects_input(self):
        data = np.zeros((10, 10), dtype=np.uint16)
        result = self._run(data, "YX")
        assert result["thumbnail_dtype"] == "uint16"

    # ------------------------------------------------------------------
    # NaN handling
    # ------------------------------------------------------------------
    def test_nan_handling(self):
        data = np.array([[0, 1, 2, np.nan]], dtype=np.float32)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self._run(data, "YX")
