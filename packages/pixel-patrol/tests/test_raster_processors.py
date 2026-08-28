"""Tests for raster processors - run_chunk and get_aggregation."""

import numpy as np
import pytest

from pixel_patrol_base.config import HISTOGRAM_BINS
from pixel_patrol_base.core.record import record_from
from pixel_patrol_base.plugins.processors.raster_image_processor import (
    QualityMetricsProcessor,
)
from pixel_patrol_base.plugins.processors.raster_processor import (
    BasicMetricsProcessor,
    HistogramProcessor,
)


@pytest.fixture
def proc():
    return BasicMetricsProcessor()


@pytest.fixture
def hist_proc():
    return HistogramProcessor()


@pytest.fixture
def quality_proc():
    return QualityMetricsProcessor()


def _chunk(proc, np_arr, dim_order_str, origin=None):
    """Run a single chunk through a processor, replicating pipeline coordinate-stamping."""
    dim_order_upper = dim_order_str.upper()
    origin = origin or [0] * len(dim_order_upper)
    record = record_from(np_arr, {"dim_order": dim_order_upper})
    row = proc.run_chunk(record)
    # Stamp coordinates and shape info the same way the pipeline does.
    row.update({f"dim_{d.lower()}": origin[i] for i, d in enumerate(dim_order_upper)})
    row["num_pixels"] = int(np.prod(np_arr.shape))
    row.update({f"size_{d}": np_arr.shape[i] for i, d in enumerate(dim_order_upper)})
    return row


def _blurred(sharp):
    """Box-blur a 2-D float32 array 8 times to produce a heavily blurred version."""
    b = sharp.copy()
    for _ in range(8):
        b[1:-1, 1:-1] = (
            b[:-2, :-2] + b[:-2, 1:-1] + b[:-2, 2:] +
            b[1:-1, :-2] + b[1:-1, 1:-1] + b[1:-1, 2:] +
            b[2:, :-2]  + b[2:, 1:-1]  + b[2:, 2:]
        ) / 9.0
    return b


# ---------------------------------------------------------------------------
# Output keys
# ---------------------------------------------------------------------------

def test_basic_processor_keys(proc):
    row = _chunk(proc, np.arange(16, dtype=np.uint8).reshape(4, 4), "YX")
    for k in ("mean_intensity", "std_intensity", "min_intensity", "max_intensity", "finite_pixel_count"):
        assert k in row, f"Missing key: {k}"


def test_histogram_processor_keys(hist_proc):
    row = _chunk(hist_proc, np.arange(16, dtype=np.uint8).reshape(4, 4), "YX")
    for k in ("histogram_counts", "histogram_min", "histogram_max", "histogram_nan_count"):
        assert k in row, f"Missing key: {k}"


def test_quality_processor_keys(quality_proc):
    row = _chunk(quality_proc, np.arange(16, dtype=np.uint8).reshape(4, 4), "YX")
    for k in ("laplacian_variance", "spectral_slope", "dark_clipping_fraction", "bright_clipping_fraction"):
        assert k in row, f"Missing key: {k}"
    for removed in ("ringing_index", "jpeg_block_ratio", "estimated_noise_std",
                    "min_value_pixel_fraction", "fraction_at_image_max", "noise_mad"):
        assert removed not in row, f"Removed metric still present: {removed}"


# ---------------------------------------------------------------------------
# Metric values
# ---------------------------------------------------------------------------

def test_mean_std_min_max(proc):
    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
    row = _chunk(proc, data, "YX")
    assert row["mean_intensity"] == pytest.approx(5.0, rel=1e-5)
    assert row["min_intensity"]  == pytest.approx(1.0, rel=1e-5)
    assert row["max_intensity"]  == pytest.approx(9.0, rel=1e-5)
    assert row["std_intensity"]  == pytest.approx(2.5819888, rel=1e-5)


def test_nan_excluded(proc):
    data = np.array([[0, 1, 2, 3, 4, np.nan]], dtype=np.float32)
    row = _chunk(proc, data, "YX")
    assert row["mean_intensity"]     == pytest.approx(2.0, rel=1e-5)
    assert row["finite_pixel_count"] == 5


def test_multi_dim_chunk_reduces_over_all_dims(proc):
    """run_chunk reduces over all dims including leading ones."""
    data = np.array(
        [[[1.0, 2.0], [3.0, 4.0]], [[np.nan, np.nan], [np.nan, np.nan]]],
        dtype=np.float32,
    )
    row = _chunk(proc, data, "TYX")
    assert row["mean_intensity"] == pytest.approx(2.5, rel=1e-5)
    assert row["min_intensity"]  == pytest.approx(1.0, rel=1e-5)


def test_lowercase_dim_order(quality_proc):
    data = np.random.default_rng(0).integers(10, 200, (8, 8), dtype=np.uint8).astype(np.float32)
    row = _chunk(quality_proc, data, "yx")
    assert np.isfinite(row["laplacian_variance"])


# ---------------------------------------------------------------------------
# Origin / coordinates
# ---------------------------------------------------------------------------

def test_origin_embedded_in_row(proc):
    row = _chunk(proc, np.ones((8, 8), dtype=np.float32), "YX", origin=[256, 512])
    assert row["dim_y"] == 256
    assert row["dim_x"] == 512


def test_nonspatial_origin(proc):
    row = _chunk(proc, np.ones((1, 8, 8), dtype=np.float32), "TYX", origin=[3, 0, 0])
    assert row["dim_t"] == 3


# ---------------------------------------------------------------------------
# Histograms
# ---------------------------------------------------------------------------

def test_histogram_uint8_bins(hist_proc):
    data = np.array([[0, 128, 255], [50, 100, 200]], dtype=np.uint8)
    row = _chunk(hist_proc, data, "YX")
    counts = row["histogram_counts"]
    assert len(counts) == HISTOGRAM_BINS
    assert int(counts.sum()) == 6
    assert row["histogram_min"] == 0.0
    assert row["histogram_max"] == float(HISTOGRAM_BINS)
    for v in (0, 50, 100, 128, 200, 255):
        assert counts[v] == 1


def test_histogram_uint8_narrow_span_anchors_to_dtype(hist_proc):
    data = np.array([[10, 15, 20], [12, 18, 11]], dtype=np.uint8)
    row = _chunk(hist_proc, data, "YX")
    assert row["histogram_min"] == 0.0
    assert row["histogram_max"] == float(HISTOGRAM_BINS)
    for v in (10, 11, 12, 15, 18, 20):
        assert row["histogram_counts"][v] == 1


def test_histogram_uint16_narrow_span_stays_tight(hist_proc):
    data = np.array([[100, 110, 120], [105, 115, 101]], dtype=np.uint16)
    row = _chunk(hist_proc, data, "YX")
    assert row["histogram_min"] == 100.0
    assert row["histogram_max"] == 100.0 + HISTOGRAM_BINS


def test_histogram_int_wide_span_keeps_true_range(hist_proc):
    data = np.arange(HISTOGRAM_BINS * 4, dtype=np.uint16).reshape(2, -1)
    row = _chunk(hist_proc, data, "YX")
    assert row["histogram_min"] == 0.0
    assert row["histogram_max"] == float(HISTOGRAM_BINS * 4 - 1)


def test_histogram_nan_count(hist_proc):
    data = np.array([[0.0, 1.0, 2.0, 1.0], [np.nan, np.nan, np.nan, 255.0]], dtype=np.float32)
    assert _chunk(hist_proc, data, "YX")["histogram_nan_count"] == 3


# ---------------------------------------------------------------------------
# laplacian_variance
# ---------------------------------------------------------------------------

def test_laplacian_variance_finite(quality_proc):
    data = np.linspace(0, 1, 40 * 40, dtype=np.float32).reshape(40, 40)
    assert np.isfinite(_chunk(quality_proc, data, "YX")["laplacian_variance"])


def test_laplacian_variance_sharper_scores_higher(quality_proc):
    rng = np.random.default_rng(42)
    sharp = rng.integers(0, 256, (32, 32), dtype=np.uint8).astype(np.float32)
    row_sharp   = _chunk(quality_proc, sharp,          "YX")
    row_blurred = _chunk(quality_proc, _blurred(sharp), "YX")
    assert row_sharp["laplacian_variance"] > row_blurred["laplacian_variance"]


def test_laplacian_variance_small_image_returns_nan(quality_proc):
    row = _chunk(quality_proc, np.ones((2, 2), dtype=np.float32), "YX")
    assert np.isnan(row.get("laplacian_variance", np.nan))


def test_laplacian_variance_flat_image_scores_zero(quality_proc):
    data = np.full((8, 8), 255, dtype=np.uint8)
    assert _chunk(quality_proc, data, "YX")["laplacian_variance"] == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# spectral_slope
# ---------------------------------------------------------------------------

def test_spectral_slope_finite(quality_proc):
    data = np.random.default_rng(0).integers(10, 200, (64, 64), dtype=np.uint8).astype(np.float32)
    assert np.isfinite(_chunk(quality_proc, data, "YX")["spectral_slope"])


def test_spectral_slope_small_image_returns_nan(quality_proc):
    row = _chunk(quality_proc, np.ones((16, 16), dtype=np.float32), "YX")
    assert np.isnan(row.get("spectral_slope", np.nan))


def test_spectral_slope_blur_steepens(quality_proc):
    # Blur concentrates power at low frequencies -> more negative slope.
    rng = np.random.default_rng(3)
    sharp = rng.integers(0, 256, (64, 64), dtype=np.uint8).astype(np.float32)
    row_s = _chunk(quality_proc, sharp,          "YX")
    row_b = _chunk(quality_proc, _blurred(sharp), "YX")
    assert row_b["spectral_slope"] < row_s["spectral_slope"]


def test_spectral_slope_negative_for_natural_content(quality_proc):
    # Natural images have 1/f^alpha spectra with alpha > 0, so slope is negative.
    rng = np.random.default_rng(5)
    data = rng.integers(0, 256, (64, 64), dtype=np.uint8).astype(np.float32)
    # Even random noise has slope close to 0 (white), so blur it slightly to
    # get a realistic image-like spectrum that's clearly negative.
    blurred = _blurred(data)
    assert _chunk(quality_proc, blurred, "YX")["spectral_slope"] < 0


# ---------------------------------------------------------------------------
# dark_clipping_fraction
# ---------------------------------------------------------------------------

def test_dark_clipping_fraction_counts_zero_pixels(quality_proc):
    data = np.array([[0, 0, 255, 255], [0, 255, 255, 255]], dtype=np.uint8)
    assert _chunk(quality_proc, data, "YX")["dark_clipping_fraction"] == pytest.approx(3 / 8, rel=1e-5)


def test_dark_clipping_fraction_signed_int(quality_proc):
    # int16 min is -32768; pixels at that value should be counted.
    data = np.zeros((8, 8), dtype=np.int16)
    data[0, 0] = -32768
    data[0, 1] = -32768
    assert _chunk(quality_proc, data, "YX")["dark_clipping_fraction"] == pytest.approx(2 / 64, rel=1e-5)


def test_dark_clipping_fraction_nan_for_float(quality_proc):
    data = np.linspace(0, 1, 8 * 8, dtype=np.float32).reshape(8, 8)
    assert np.isnan(_chunk(quality_proc, data, "YX")["dark_clipping_fraction"])


# ---------------------------------------------------------------------------
# bright_clipping_fraction
# ---------------------------------------------------------------------------

def test_bright_clipping_fraction_uint8(quality_proc):
    data = np.array([[255, 255, 0, 0], [255, 0, 0, 0]], dtype=np.uint8)
    assert _chunk(quality_proc, data, "YX")["bright_clipping_fraction"] == pytest.approx(3 / 8, rel=1e-5)


def test_bright_clipping_fraction_12bit_in_uint16(quality_proc):
    # 12-bit data in uint16: image max is 4095, not 65535 (dtype ceiling).
    # bright_clipping_fraction uses dtype ceiling, so this should be 0.
    data = np.zeros((8, 8), dtype=np.uint16)
    data[0, 0] = 4095
    data[0, 1] = 4095
    assert _chunk(quality_proc, data, "YX")["bright_clipping_fraction"] == pytest.approx(0.0, rel=1e-5)


def test_bright_clipping_fraction_uint16_at_ceiling(quality_proc):
    data = np.zeros((8, 8), dtype=np.uint16)
    data[0, 0] = 65535
    data[0, 1] = 65535
    assert _chunk(quality_proc, data, "YX")["bright_clipping_fraction"] == pytest.approx(2 / 64, rel=1e-5)


def test_bright_clipping_fraction_nan_for_float(quality_proc):
    data = np.linspace(0, 1, 8 * 8, dtype=np.float32).reshape(8, 8)
    assert np.isnan(_chunk(quality_proc, data, "YX")["bright_clipping_fraction"])


# ---------------------------------------------------------------------------
# get_aggregation
# ---------------------------------------------------------------------------

def test_get_aggregation_callable_for_own_columns(proc):
    for name in proc.OUTPUT_SCHEMA:
        assert callable(proc.get_aggregation(name)), f"expected callable for {name}"


def test_get_aggregation_none_for_unknown(proc):
    assert proc.get_aggregation("no_such_column") is None


def test_get_aggregation_mean_is_correct(proc):
    row1 = _chunk(proc, np.full((4, 4), 2.0, dtype=np.float32), "YX")
    row2 = _chunk(proc, np.full((4, 4), 4.0, dtype=np.float32), "YX")
    fn = proc.get_aggregation("mean_intensity")
    assert fn([row1, row2], ()) == pytest.approx(3.0, rel=1e-4)


def test_get_aggregation_histogram_callable(hist_proc):
    assert callable(hist_proc.get_aggregation("histogram_counts"))
