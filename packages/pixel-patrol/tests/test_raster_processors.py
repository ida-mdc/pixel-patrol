"""Tests for raster processors — run_chunk and get_aggregation."""

import numpy as np
import pytest

from pixel_patrol_base.config import HISTOGRAM_BINS
from pixel_patrol_base.core.record import record_from
from pixel_patrol_image.plugins.processors.raster_image_processor import (
    CompressionMetricsProcessor,
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
    row.update({f"{d}_size": np_arr.shape[i] for i, d in enumerate(dim_order_upper)})
    return row


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
    for k in ("michelson_contrast", "mscn_variance", "texture_heterogeneity", "laplacian_variance"):
        assert k in row, f"Missing key: {k}"


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
    assert np.isfinite(row["michelson_contrast"])
    assert np.isfinite(row["mscn_variance"])
    assert np.isfinite(row["texture_heterogeneity"])


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
    assert len(row["histogram_counts"]) == HISTOGRAM_BINS
    assert int(row["histogram_counts"].sum()) == 6
    assert row["histogram_min"] == 0.0
    assert row["histogram_max"] == 255.0


def test_histogram_nan_count(hist_proc):
    data = np.array([[0.0, 1.0, 2.0, 1.0], [np.nan, np.nan, np.nan, 255.0]], dtype=np.float32)
    assert _chunk(hist_proc, data, "YX")["histogram_nan_count"] == 3


# ---------------------------------------------------------------------------
# Compression metrics
# ---------------------------------------------------------------------------

def test_compression_absent_from_basic(proc):
    data = np.linspace(0, 1, 64 * 64, dtype=np.float32).reshape(64, 64)
    row = _chunk(proc, data, "YX")
    assert "blocking_index" not in row
    assert "ringing_index" not in row


def test_compression_finite_when_enabled():
    data = np.linspace(0, 1, 64 * 64, dtype=np.float32).reshape(64, 64)
    row = _chunk(CompressionMetricsProcessor(), data, "YX")
    assert np.isfinite(row["blocking_index"])
    assert np.isfinite(row["ringing_index"])


# ---------------------------------------------------------------------------
# Quality metrics
# ---------------------------------------------------------------------------

def test_quality_metrics_finite(quality_proc):
    data = np.linspace(0, 1, 40 * 40, dtype=np.float32).reshape(40, 40)
    row = _chunk(quality_proc, data, "YX")
    assert np.isfinite(row["michelson_contrast"])
    assert np.isfinite(row["mscn_variance"])
    assert np.isfinite(row["texture_heterogeneity"])
    assert np.isfinite(row["laplacian_variance"])


def test_laplacian_variance_sharper_image_scores_higher(quality_proc):
    rng = np.random.default_rng(42)
    sharp = rng.integers(0, 256, (32, 32), dtype=np.uint8).astype(np.float32)
    # Blur by repeated box-averaging — reduces second-derivative energy.
    blurred = sharp.copy()
    for _ in range(8):
        blurred[1:-1, 1:-1] = (
            blurred[:-2, :-2] + blurred[:-2, 1:-1] + blurred[:-2, 2:] +
            blurred[1:-1, :-2] + blurred[1:-1, 1:-1] + blurred[1:-1, 2:] +
            blurred[2:, :-2]  + blurred[2:, 1:-1]  + blurred[2:, 2:]
        ) / 9.0
    row_sharp   = _chunk(quality_proc, sharp,   "YX")
    row_blurred = _chunk(quality_proc, blurred, "YX")
    assert row_sharp["laplacian_variance"] > row_blurred["laplacian_variance"]


def test_laplacian_variance_small_image_returns_nan(quality_proc):
    row = _chunk(quality_proc, np.ones((2, 2), dtype=np.float32), "YX")
    assert np.isnan(row.get("laplacian_variance", np.nan))


def test_michelson_contrast_high_frequency_scores_higher(quality_proc):
    # Checkerboard has local range = 1 in every 3×3 window; a smooth gradient
    # has tiny local range per window, so the ratio to global std is much smaller.
    checker = (np.indices((32, 32), dtype=np.float32).sum(axis=0) % 2)
    smooth  = np.linspace(0, 1, 32 * 32, dtype=np.float32).reshape(32, 32)
    row_checker = _chunk(quality_proc, checker, "YX")
    row_smooth  = _chunk(quality_proc, smooth,  "YX")
    assert row_checker["michelson_contrast"] > row_smooth["michelson_contrast"]


def test_mscn_variance_noise_scores_higher_than_gradient(quality_proc):
    # For a linear gradient each pixel equals its 3×3 local mean exactly, so
    # every MSCN coefficient is zero and variance is exactly zero.
    gradient = np.linspace(0, 255, 32 * 32, dtype=np.float32).reshape(32, 32)
    noise    = np.random.default_rng(0).integers(0, 256, (32, 32), dtype=np.uint8).astype(np.float32)
    row_gradient = _chunk(quality_proc, gradient, "YX")
    row_noise    = _chunk(quality_proc, noise,    "YX")
    assert row_noise["mscn_variance"] > row_gradient["mscn_variance"]


def test_texture_heterogeneity_patchy_scores_higher(quality_proc):
    # Patchy image: flat top half (local std = 0) + noisy bottom half (local std > 0)
    # → wide spread of local stds → high CoV. Uniform noise → consistent local stds → low CoV.
    rng  = np.random.default_rng(1)
    noise = rng.integers(0, 256, (32, 32), dtype=np.uint8).astype(np.float32)
    patchy = noise.copy()
    patchy[:16, :] = 0.0
    row_patchy  = _chunk(quality_proc, patchy, "YX")
    row_uniform = _chunk(quality_proc, noise,  "YX")
    assert row_patchy["texture_heterogeneity"] > row_uniform["texture_heterogeneity"]


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


def test_get_aggregation_skips_histogram_in_spatial_grouping(hist_proc):
    """Histograms are non-spatial: get_aggregation returns None for spatial groupings."""
    row = _chunk(hist_proc, np.ones((8, 8), dtype=np.float32), "YX")
    fn = hist_proc.get_aggregation("histogram_counts")
    assert fn([row], ("dim_y", "dim_x")) is None
