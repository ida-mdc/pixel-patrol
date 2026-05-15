import pytest
import numpy as np
import dask.array as da
from pixel_patrol_base.core.record import record_from
from pixel_patrol_base.plugins.processors.basic_stats_processor import BasicStatsProcessor


def _row(rows, **dims):
    """Find the row matching the given dim_* values (obs_level = len(dims))."""
    obs = len(dims)
    for r in rows:
        if r["obs_level"] == obs and all(r.get(f"dim_{k}") == v for k, v in dims.items()):
            return r
    raise KeyError(f"No row with obs_level={obs} dims={dims}")


class TestBasicStatsProcessor:
    """Test suite for BasicStatsProcessor to verify correct calculation of basic statistics."""

    def test_basic_stats_simple_2d_image(self):
        """Test basic stats calculation on a simple 2D image."""
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
        dask_data = da.from_array(data, chunks=(3, 3))
        record = record_from(dask_data, {"dim_order": "YX"})
        processor = BasicStatsProcessor()

        rows = processor.run(record)
        g = _row(rows)

        assert "mean_intensity" in g
        assert "std_intensity" in g
        assert "min_intensity" in g
        assert "max_intensity" in g
        assert g["mean_intensity"] == pytest.approx(5.0, rel=1e-5)
        assert g["min_intensity"] == pytest.approx(1.0, rel=1e-5)
        assert g["max_intensity"] == pytest.approx(9.0, rel=1e-5)
        assert g["std_intensity"] == pytest.approx(2.5819888, rel=1e-5)

    def test_basic_stats_constant_image(self):
        """Test basic stats on a constant image."""
        data = np.full((10, 10), 42.0, dtype=np.float32)
        dask_data = da.from_array(data, chunks=(5, 5))
        record = record_from(dask_data, {"dim_order": "YX"})
        processor = BasicStatsProcessor()

        g = _row(processor.run(record))
        assert g["mean_intensity"] == pytest.approx(42.0, rel=1e-5)
        assert g["min_intensity"] == pytest.approx(42.0, rel=1e-5)
        assert g["max_intensity"] == pytest.approx(42.0, rel=1e-5)
        assert g["std_intensity"] == pytest.approx(0.0, rel=1e-5)

    def test_basic_stats_uint8_image(self):
        """Test basic stats on uint8 image."""
        data = np.array([[0, 128, 255], [50, 100, 200]], dtype=np.uint8)
        dask_data = da.from_array(data, chunks=(2, 3))
        record = record_from(dask_data, {"dim_order": "YX"})
        processor = BasicStatsProcessor()

        g = _row(processor.run(record))
        assert float(g["mean_intensity"]) == pytest.approx(122.16666666, rel=1e-5)
        assert g["min_intensity"] == pytest.approx(0.0, rel=1e-5)
        assert g["max_intensity"] == pytest.approx(255.0, rel=1e-5)

    def test_basic_stats_with_time_dimension(self):
        """Test basic stats on image with time dimension (aggregated)."""
        data = np.array([[[1,2,3],[4,5,6],[7,8,9],
                          [10,11,12],[13,14,15],[16,17,18],
                          [19,20,21],[22,23,24],[25,26,27]],
                         [[28, 29, 30], [31, 32, 33], [34, 35, 36],
                          [37, 38, 39], [40, 41, 42], [43, 44, 45],
                          [46, 47, 48], [49, 50, 51], [52, 53, 54]]
                         ]).astype(np.float32)
        dask_data = da.from_array(data)
        record = record_from(dask_data, {"dim_order": "TYX"})
        processor = BasicStatsProcessor()

        rows = processor.run(record)
        assert _row(rows)["mean_intensity"] == pytest.approx(27.5, rel=1e-4)
        assert _row(rows, t=0)["mean_intensity"] == pytest.approx(14, rel=1e-4)
        assert _row(rows, t=1)["mean_intensity"] == pytest.approx(41, rel=1e-4)
        assert not any(r.get("dim_t") == 2 for r in rows)

    def test_basic_stats_with_channel_dimension(self):
        """Test basic stats on image with channel dimension."""
        data = np.array([
            [[1, 2], [3, 4]],   # Channel 0
            [[10, 20], [30, 40]] # Channel 1
        ], dtype=np.float32)
        dask_data = da.from_array(data, chunks=(1, 2, 2))
        record = record_from(dask_data, {"dim_order": "CYX"})
        processor = BasicStatsProcessor()

        rows = processor.run(record)
        assert "mean_intensity" in _row(rows)
        assert _row(rows, c=0)["mean_intensity"] == pytest.approx(2.5, rel=1e-5)
        assert _row(rows, c=1)["mean_intensity"] == pytest.approx(25.0, rel=1e-5)

    def test_basic_stats_with_multiple_dimensions(self):
        """Test basic stats on image with T, C, Z dimensions."""
        data = np.random.rand(2, 2, 2, 3, 3).astype(np.float32) * 100
        dask_data = da.from_array(data, chunks=(1, 1, 1, 3, 3))
        record = record_from(dask_data, {"dim_order": "TCZYX"})
        processor = BasicStatsProcessor()

        rows = processor.run(record)
        for i in [0, 1]:
            assert _row(rows, t=i)["mean_intensity"] is not None
            assert _row(rows, z=i)["mean_intensity"] is not None
            assert _row(rows, c=i)["mean_intensity"] is not None
            for j in [0, 1]:
                assert _row(rows, t=i, z=j)["mean_intensity"] is not None
                assert _row(rows, t=i, c=j)["mean_intensity"] is not None
                for k in [0, 1]:
                    assert _row(rows, t=i, c=j, z=k)["mean_intensity"] is not None

    def test_basic_stats_empty_image(self):
        """Test basic stats on empty image."""
        data = np.array([[]], dtype=np.float32)
        dask_data = da.from_array(data, chunks=(1, 1))
        record = record_from(dask_data, {"dim_order": "YX"})
        processor = BasicStatsProcessor()

        g = _row(processor.run(record))
        assert "mean_intensity" in g
        assert np.isnan(g["mean_intensity"])

    def test_basic_stats_single_pixel(self):
        """Test basic stats on single pixel image."""
        data = np.array([[42.0]], dtype=np.float32)
        dask_data = da.from_array(data, chunks=(1, 1))
        record = record_from(dask_data, {"dim_order": "YX"})
        processor = BasicStatsProcessor()

        g = _row(processor.run(record))
        assert g["mean_intensity"] == pytest.approx(42.0, rel=1e-5)
        assert g["min_intensity"] == pytest.approx(42.0, rel=1e-5)
        assert g["max_intensity"] == pytest.approx(42.0, rel=1e-5)
        assert g["std_intensity"] == pytest.approx(0.0, rel=1e-5)

    def test_basic_stats_image_with_nan(self):
        """Test basic stats on image with NaN values — NaNs are ignored in all statistics."""
        data = np.array([[0, 1, 2, 3, 4, np.nan]], dtype=np.float32)
        dask_data = da.from_array(data, chunks=(1, 1))
        record = record_from(dask_data, {"dim_order": "YX"})
        processor = BasicStatsProcessor()

        g = _row(processor.run(record))
        assert g["mean_intensity"] == pytest.approx(2, rel=1e-5)
        assert g["min_intensity"] == pytest.approx(0, rel=1e-5)
        assert g["max_intensity"] == pytest.approx(4, rel=1e-5)
        assert g["std_intensity"] == pytest.approx(2**0.5, rel=1e-5)

    def test_basic_stats_nan_slice_does_not_poison_aggregate(self):
        """If one time slice is entirely NaN, the aggregate must reflect only the valid slice."""
        data = np.array([
            [[1., 2.], [3., 4.]],                     # t0: mean=2.5
            [[np.nan, np.nan], [np.nan, np.nan]],      # t1: all NaN
        ], dtype=np.float32)
        dask_data = da.from_array(data, chunks=(1, 2, 2))
        record = record_from(dask_data, {"dim_order": "TYX"})

        rows = BasicStatsProcessor().run(record)
        assert _row(rows, t=0)["mean_intensity"] == pytest.approx(2.5, rel=1e-5)
        assert np.isnan(_row(rows, t=1)["mean_intensity"])
        assert _row(rows)["mean_intensity"] == pytest.approx(2.5, rel=1e-5)
        assert _row(rows)["min_intensity"] == pytest.approx(1.0, rel=1e-5)
        assert _row(rows)["max_intensity"] == pytest.approx(4.0, rel=1e-5)