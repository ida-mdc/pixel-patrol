import pytest
import numpy as np
import dask.array as da
from pixel_patrol_base.core.record import record_from
from pixel_patrol_base.plugins.processors.basic_stats_processor import BasicStatsProcessor


class TestBasicStatsProcessor:
    """Test suite for BasicStatsProcessor to verify correct calculation of basic statistics."""

    def test_basic_stats_simple_2d_image(self):
        """Test basic stats calculation on a simple 2D image."""
        # Create a simple 2D image with known values
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
        dask_data = da.from_array(data, chunks=(3, 3))
        
        record = record_from(dask_data, {"dim_order": "YX"})
        processor = BasicStatsProcessor()
        
        result = processor.run(record)
        
        # Verify all expected keys are present
        assert "mean_intensity" in result
        assert "std_intensity" in result
        assert "min_intensity" in result
        assert "max_intensity" in result
        
        # Verify calculations
        assert result["mean_intensity"] == pytest.approx(5.0, rel=1e-5)
        assert result["min_intensity"] == pytest.approx(1.0, rel=1e-5)
        assert result["max_intensity"] == pytest.approx(9.0, rel=1e-5)
        assert result["std_intensity"] == pytest.approx(2.5819888, rel=1e-5)

    def test_basic_stats_constant_image(self):
        """Test basic stats on a constant image."""
        data = np.full((10, 10), 42.0, dtype=np.float32)
        dask_data = da.from_array(data, chunks=(5, 5))
        
        record = record_from(dask_data, {"dim_order": "YX"})
        processor = BasicStatsProcessor()
        
        result = processor.run(record)
        
        assert result["mean_intensity"] == pytest.approx(42.0, rel=1e-5)
        assert result["min_intensity"] == pytest.approx(42.0, rel=1e-5)
        assert result["max_intensity"] == pytest.approx(42.0, rel=1e-5)
        assert result["std_intensity"] == pytest.approx(0.0, rel=1e-5)

    def test_basic_stats_uint8_image(self):
        """Test basic stats on uint8 image."""
        data = np.array([[0, 128, 255], [50, 100, 200]], dtype=np.uint8)
        dask_data = da.from_array(data, chunks=(2, 3))
        
        record = record_from(dask_data, {"dim_order": "YX"})
        processor = BasicStatsProcessor()
        
        result = processor.run(record)
        
        assert float(result["mean_intensity"]) == pytest.approx(122.16666666, rel=1e-5)
        assert result["min_intensity"] == pytest.approx(0.0, rel=1e-5)
        assert result["max_intensity"] == pytest.approx(255.0, rel=1e-5)

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
        
        result = processor.run(record)
        
        # Should have aggregated stats across time
        assert "mean_intensity" in result
        assert "std_intensity" in result
        assert "min_intensity" in result
        assert "max_intensity" in result
        
        # Also should have per-time-slice stats
        assert "mean_intensity_t0" in result
        assert "mean_intensity_t1" in result
        assert "mean_intensity_t2" not in result

        assert result["mean_intensity"] == pytest.approx(27.5, rel=1e-4)
        assert result["mean_intensity_t0"] == pytest.approx(14, rel=1e-4)
        assert result["mean_intensity_t1"] == pytest.approx(41, rel=1e-4)

    def test_basic_stats_with_channel_dimension(self):
        """Test basic stats on image with channel dimension."""
        # Create a CYX image: 2 channels, 5x5 spatial
        data = np.array([
            [[1, 2], [3, 4]],  # Channel 0
            [[10, 20], [30, 40]]  # Channel 1
        ], dtype=np.float32)
        dask_data = da.from_array(data, chunks=(1, 2, 2))
        
        record = record_from(dask_data, {"dim_order": "CYX"})
        processor = BasicStatsProcessor()
        
        result = processor.run(record)
        
        # Should have aggregated stats
        assert "mean_intensity" in result
        
        # Should have per-channel stats
        assert "mean_intensity_c0" in result
        assert "mean_intensity_c1" in result
        
        # Verify per-channel means
        assert result["mean_intensity_c0"] == pytest.approx(2.5, rel=1e-5)
        assert result["mean_intensity_c1"] == pytest.approx(25.0, rel=1e-5)

    def test_basic_stats_with_multiple_dimensions(self):
        """Test basic stats on image with T, C, Z dimensions."""
        # Create a TCZYX image: 2 time, 2 channels, 2 z-slices, 3x3 spatial
        data = np.random.rand(2, 2, 2, 3, 3).astype(np.float32) * 100
        dask_data = da.from_array(data, chunks=(1, 1, 1, 3, 3))
        
        record = record_from(dask_data, {"dim_order": "TCZYX"})
        processor = BasicStatsProcessor()
        
        result = processor.run(record)
        
        for i in [0,1]:
            assert f"mean_intensity_t{i}" in result
            assert f"mean_intensity_z{i}" in result
            assert f"mean_intensity_c{i}" in result

            for j in [0,1]:
                assert f"mean_intensity_t{i}_z{j}" in result
                assert f"mean_intensity_t{i}_c{j}" in result

                for k in [0,1]:
                    assert f"mean_intensity_t{i}_c{j}_z{k}" in result

    def test_basic_stats_empty_image(self):
        """Test basic stats on empty image."""
        data = np.array([[]], dtype=np.float32)
        dask_data = da.from_array(data, chunks=(1, 1))
        
        record = record_from(dask_data, {"dim_order": "YX"})
        processor = BasicStatsProcessor()
        
        result = processor.run(record)
        
        # Should handle empty arrays gracefully
        assert "mean_intensity" in result
        assert np.isnan(result["mean_intensity"])

    def test_basic_stats_single_pixel(self):
        """Test basic stats on single pixel image."""
        data = np.array([[42.0]], dtype=np.float32)
        dask_data = da.from_array(data, chunks=(1, 1))
        
        record = record_from(dask_data, {"dim_order": "YX"})
        processor = BasicStatsProcessor()
        
        result = processor.run(record)
        
        assert result["mean_intensity"] == pytest.approx(42.0, rel=1e-5)
        assert result["min_intensity"] == pytest.approx(42.0, rel=1e-5)
        assert result["max_intensity"] == pytest.approx(42.0, rel=1e-5)
        assert result["std_intensity"] == pytest.approx(0.0, rel=1e-5)

