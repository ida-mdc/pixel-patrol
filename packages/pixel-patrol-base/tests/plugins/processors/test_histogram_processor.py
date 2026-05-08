import dask.array as da
import numpy as np
import pytest

from pixel_patrol_base.core.record import record_from
from pixel_patrol_base.plugins.processors.histogram_processor import HistogramProcessor
from pixel_patrol_base.config import HISTOGRAM_BINS


def _row(rows, **dims):
    obs = len(dims)
    for r in rows:
        if r["obs_level"] == obs and all(r.get(f"dim_{k}") == v for k, v in dims.items()):
            return r
    raise KeyError(f"No row with obs_level={obs} dims={dims}")


class TestHistogramProcessor:
    """Test suite for HistogramProcessor to verify correct calculation of histograms."""

    def test_histogram_simple_2d_image(self):
        """Test histogram calculation on a simple 2D image."""
        data = np.array([[0, 128, 255], [50, 100, 200]], dtype=np.uint8)
        dask_data = da.from_array(data, chunks=(2, 3))
        record = record_from(dask_data, {"dim_order": "YX"})
        processor = HistogramProcessor()

        result = _row(processor.run(record))
        assert "histogram_counts" in result and "histogram_min" in result and "histogram_max" in result
        assert len(result["histogram_counts"]) == HISTOGRAM_BINS
        assert isinstance(result["histogram_counts"], np.ndarray)
        assert result["histogram_min"] == 0.0
        assert result["histogram_max"] == 255.0
        assert result["histogram_counts"][0] == 1
        assert result["histogram_counts"][50] == 1
        assert result["histogram_counts"][100] == 1
        assert result["histogram_counts"][128] == 1
        assert result["histogram_counts"][200] == 1
        assert result["histogram_counts"][255] == 1
        assert sum(result["histogram_counts"]) == 6

    def test_histogram_constant_image(self):
        """Test histogram on a constant image."""
        data = np.full((10, 10), 42, dtype=np.uint8)
        dask_data = da.from_array(data, chunks=(5, 5))
        record = record_from(dask_data, {"dim_order": "YX"})
        processor = HistogramProcessor()

        result = _row(processor.run(record))
        assert result["histogram_counts"][42] == 100
        for i in range(HISTOGRAM_BINS):
            if i != 42:
                assert result["histogram_counts"][i] == 0
        assert result["histogram_min"] == 0.0
        assert result["histogram_max"] == 255.0

    def test_histogram_uint8_full_range(self):
        """Test histogram on uint8 image uses full 0-255 range."""
        data = np.array([[100, 150], [120, 180]], dtype=np.uint8)
        dask_data = da.from_array(data, chunks=(2, 2))
        record = record_from(dask_data, {"dim_order": "YX"})

        result = _row(HistogramProcessor().run(record))
        assert result["histogram_min"] == 0.0
        assert result["histogram_max"] == 255.0
        assert len(result["histogram_counts"]) == HISTOGRAM_BINS

    def test_histogram_with_time_dimension(self):
        """Test histogram on image with time dimension."""
        data = np.array([
            [[0, 128, 255], [50, 100, 200], [25, 75, 225]],  # t0
            [[10, 140, 245], [60, 110, 210], [35, 85, 235]]  # t1
        ], dtype=np.uint8)
        dask_data = da.from_array(data, chunks=(1, 3, 3))
        record = record_from(dask_data, {"dim_order": "TYX"})

        rows = HistogramProcessor().run(record)
        g = _row(rows)
        assert "histogram_counts" in g and "histogram_min" in g and "histogram_max" in g

        r0 = _row(rows, t=0)
        r1 = _row(rows, t=1)
        assert len(r0["histogram_counts"]) == HISTOGRAM_BINS
        assert len(r1["histogram_counts"]) == HISTOGRAM_BINS
        assert sum(r0["histogram_counts"]) == 9
        assert sum(r1["histogram_counts"]) == 9

    def test_histogram_with_channel_dimension(self):
        """Test histogram on image with channel dimension."""
        data = np.array([
            [[0, 255], [128, 64]],    # Channel 0
            [[50, 200], [100, 150]]   # Channel 1
        ], dtype=np.uint8)
        dask_data = da.from_array(data, chunks=(1, 2, 2))
        record = record_from(dask_data, {"dim_order": "CYX"})

        rows = HistogramProcessor().run(record)
        assert "histogram_counts" in _row(rows)
        assert sum(_row(rows, c=0)["histogram_counts"]) == 4
        assert sum(_row(rows, c=1)["histogram_counts"]) == 4

    def test_histogram_with_multiple_dimensions(self):
        """Test histogram on image with T, C, Z dimensions."""
        data = np.random.randint(0, HISTOGRAM_BINS, size=(2, 2, 2, 2, 2), dtype=np.uint8)
        dask_data = da.from_array(data, chunks=(1, 1, 1, 2, 2))
        record = record_from(dask_data, {"dim_order": "TCZYX"})

        rows = HistogramProcessor().run(record)
        assert "histogram_counts" in _row(rows)
        assert any(r.get("dim_t") is not None for r in rows)
        assert any(r.get("dim_c") is not None for r in rows)
        assert any(r.get("dim_z") is not None for r in rows)
        for r in rows:
            if "histogram_counts" in r:
                assert len(r["histogram_counts"]) == HISTOGRAM_BINS
                assert isinstance(r["histogram_counts"], np.ndarray)

    def test_histogram_empty_image(self):
        """Empty array: zero-filled 256-bin histogram, nan_count=0."""
        data = np.array([[]], dtype=np.uint8)
        dask_data = da.from_array(data, chunks=(1, 1))
        record = record_from(dask_data, {"dim_order": "YX"})

        result = _row(HistogramProcessor().run(record))
        assert len(result["histogram_counts"]) == HISTOGRAM_BINS
        assert sum(result["histogram_counts"]) == 0
        assert result["histogram_min"] == 0.0
        assert result["histogram_max"] == 0.0
        assert result["histogram_nan_count"] == 0

    def test_histogram_float_image(self):
        """Test histogram on float image (non-uint8)."""
        data = np.array([[0.0, 0.5, 1.0], [0.25, 0.75, 0.9]], dtype=np.float32)
        dask_data = da.from_array(data, chunks=(2, 3))
        record = record_from(dask_data, {"dim_order": "YX"})

        result = _row(HistogramProcessor().run(record))
        assert "histogram_counts" in result
        assert len(result["histogram_counts"]) == HISTOGRAM_BINS
        assert result["histogram_min"] == pytest.approx(0.0, rel=1e-5)
        assert result["histogram_max"] == pytest.approx(1.0, rel=1e-5)
        assert sum(result["histogram_counts"]) == 6

    def test_histogram_integer_non_uint8(self):
        """Test histogram on integer image that's not uint8."""
        data = np.array([[0, 1000, 2000], [500, 1500, 3000]], dtype=np.int16)
        dask_data = da.from_array(data, chunks=(2, 3))
        record = record_from(dask_data, {"dim_order": "YX"})

        result = _row(HistogramProcessor().run(record))
        assert "histogram_counts" in result
        assert len(result["histogram_counts"]) == HISTOGRAM_BINS
        assert result["histogram_min"] == pytest.approx(0.0, rel=1e-5)
        assert result["histogram_max"] == pytest.approx(3000.0, rel=1e-5)

    def test_histogram_with_nan_dask(self):
        """Partial NaN: valid pixels binned correctly, NaNs tallied in nan_count."""
        data = np.array([[0., 1., 2., 1.], [np.nan, np.nan, float("nan"), 255.]], dtype=np.float32)
        dask_data = da.from_array(data, chunks=(2, 3))
        record = record_from(dask_data, {"dim_order": "YX"})

        result = _row(HistogramProcessor().run(record))
        assert len(result["histogram_counts"]) == HISTOGRAM_BINS
        assert result["histogram_min"] == pytest.approx(0., rel=1e-5)
        assert result["histogram_max"] == pytest.approx(255., rel=1e-5)
        assert list(result["histogram_counts"][0:3]) == [1, 2, 1]
        assert result["histogram_counts"][255] == 1
        assert result["histogram_nan_count"] == 3

    def test_histogram_all_nan_image(self):
        """All-NaN array: zero bin counts, nan_count == total pixels."""
        data = np.full((10, 10), np.nan, dtype=np.float32)
        dask_data = da.from_array(data, chunks=(5, 5))
        record = record_from(dask_data, {"dim_order": "YX"})

        result = _row(HistogramProcessor().run(record))
        assert len(result["histogram_counts"]) == HISTOGRAM_BINS
        assert sum(result["histogram_counts"]) == 0
        assert np.isfinite(result["histogram_min"])
        assert np.isfinite(result["histogram_max"])
        assert result["histogram_nan_count"] == 100

