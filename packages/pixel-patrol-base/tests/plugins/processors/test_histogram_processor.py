import dask.array as da
import numpy as np
import pytest

from pixel_patrol_base.core.record import record_from
from pixel_patrol_base.plugins.processors.histogram_processor import HistogramProcessor


class TestHistogramProcessor:
    """Test suite for HistogramProcessor to verify correct calculation of histograms."""

    def test_histogram_simple_2d_image(self):
        """Test histogram calculation on a simple 2D image."""
        # Create a simple 2D image with values 0-255
        data = np.array([[0, 128, 255], [50, 100, 200]], dtype=np.uint8)
        dask_data = da.from_array(data, chunks=(2, 3))
        
        record = record_from(dask_data, {"dim_order": "YX"})
        processor = HistogramProcessor()
        
        result = processor.run(record)
        
        # Verify main histogram keys are present
        assert "histogram_counts" in result
        assert "histogram_min" in result
        assert "histogram_max" in result
        
        # Verify histogram_counts is a list of 256 elements
        assert len(result["histogram_counts"]) == 256
        assert isinstance(result["histogram_counts"], list)
        
        # Verify min and max
        assert result["histogram_min"] == 0.0
        assert result["histogram_max"] == 255.0
        
        # Verify that specific bins have counts
        # Value 0 appears once
        assert result["histogram_counts"][0] == 1
        # Value 50 appears once
        assert result["histogram_counts"][50] == 1
        # Value 100 appears once
        assert result["histogram_counts"][100] == 1
        # Value 128 appears once
        assert result["histogram_counts"][128] == 1
        # Value 200 appears once
        assert result["histogram_counts"][200] == 1
        # Value 255 appears once
        assert result["histogram_counts"][255] == 1
        
        # Verify total counts match number of pixels
        assert sum(result["histogram_counts"]) == 6

    def test_histogram_constant_image(self):
        """Test histogram on a constant image."""
        data = np.full((10, 10), 42, dtype=np.uint8)
        dask_data = da.from_array(data, chunks=(5, 5))
        
        record = record_from(dask_data, {"dim_order": "YX"})
        processor = HistogramProcessor()
        
        result = processor.run(record)
        
        # All pixels have value 42
        assert result["histogram_counts"][42] == 100
        # All other bins should be 0
        for i in range(256):
            if i != 42:
                assert result["histogram_counts"][i] == 0
        
        assert result["histogram_min"] == 0.0
        assert result["histogram_max"] == 255.0

    def test_histogram_uint8_full_range(self):
        """Test histogram on uint8 image uses full 0-255 range."""
        # Create image with values only in middle range
        data = np.array([[100, 150], [120, 180]], dtype=np.uint8)
        dask_data = da.from_array(data, chunks=(2, 2))
        
        record = record_from(dask_data, {"dim_order": "YX"})
        processor = HistogramProcessor()
        
        result = processor.run(record)
        
        # Even though actual min is 100 and max is 180,
        # for uint8, histogram should use full 0-255 range
        assert result["histogram_min"] == 0.0
        assert result["histogram_max"] == 255.0
        assert len(result["histogram_counts"]) == 256

    def test_histogram_with_time_dimension(self):
        """Test histogram on image with time dimension."""
        # Create a TYX image: 2 time points, 3x3 spatial
        data = np.array([
            [[0, 128, 255], [50, 100, 200], [25, 75, 225]],  # t0
            [[10, 140, 245], [60, 110, 210], [35, 85, 235]]  # t1
        ], dtype=np.uint8)
        dask_data = da.from_array(data, chunks=(1, 3, 3))
        
        record = record_from(dask_data, {"dim_order": "TYX"})
        processor = HistogramProcessor()
        
        result = processor.run(record)
        
        # Should have full image histogram
        assert "histogram_counts" in result
        assert "histogram_min" in result
        assert "histogram_max" in result
        
        # Should have per-time-slice histograms
        assert "histogram_counts_t0" in result
        assert "histogram_min_t0" in result
        assert "histogram_max_t0" in result
        assert "histogram_counts_t1" in result
        assert "histogram_min_t1" in result
        assert "histogram_max_t1" in result
        
        # Verify per-time histogram counts
        assert len(result["histogram_counts_t0"]) == 256
        assert len(result["histogram_counts_t1"]) == 256
        
        # Each time slice has 9 pixels
        assert sum(result["histogram_counts_t0"]) == 9
        assert sum(result["histogram_counts_t1"]) == 9

    def test_histogram_with_channel_dimension(self):
        """Test histogram on image with channel dimension."""
        # Create a CYX image: 2 channels, 2x2 spatial
        data = np.array([
            [[0, 255], [128, 64]],  # Channel 0
            [[50, 200], [100, 150]]  # Channel 1
        ], dtype=np.uint8)
        dask_data = da.from_array(data, chunks=(1, 2, 2))
        
        record = record_from(dask_data, {"dim_order": "CYX"})
        processor = HistogramProcessor()
        
        result = processor.run(record)
        
        # Should have full image histogram
        assert "histogram_counts" in result
        
        # Should have per-channel histograms
        assert "histogram_counts_c0" in result
        assert "histogram_min_c0" in result
        assert "histogram_max_c0" in result
        assert "histogram_counts_c1" in result
        assert "histogram_min_c1" in result
        assert "histogram_max_c1" in result
        
        # Each channel has 4 pixels
        assert sum(result["histogram_counts_c0"]) == 4
        assert sum(result["histogram_counts_c1"]) == 4

    def test_histogram_with_multiple_dimensions(self):
        """Test histogram on image with T, C, Z dimensions."""
        # Create a TCZYX image: 2 time, 2 channels, 2 z-slices, 2x2 spatial
        data = np.random.randint(0, 256, size=(2, 2, 2, 2, 2), dtype=np.uint8)
        dask_data = da.from_array(data, chunks=(1, 1, 1, 2, 2))
        
        record = record_from(dask_data, {"dim_order": "TCZYX"})
        processor = HistogramProcessor()
        
        result = processor.run(record)
        
        # Should have full image histogram
        assert "histogram_counts" in result
        
        # Should have histograms for various dimension combinations
        # Check that we have histograms for at least some dimension combinations
        assert any("histogram_counts_t" in key for key in result.keys())
        assert any("histogram_counts_c" in key for key in result.keys())
        assert any("histogram_counts_z" in key for key in result.keys())
        
        # Verify all histogram_counts lists have 256 elements
        for key, value in result.items():
            if key.startswith("histogram_counts"):
                assert len(value) == 256
                assert isinstance(value, list)

    def test_histogram_empty_image(self):
        """Test histogram on empty image."""
        data = np.array([[]], dtype=np.uint8)
        dask_data = da.from_array(data, chunks=(1, 1))
        
        record = record_from(dask_data, {"dim_order": "YX"})
        processor = HistogramProcessor()
        
        result = processor.run(record)
        
        # Should handle empty arrays gracefully
        assert "histogram_counts" in result
        assert len(result["histogram_counts"]) == 256
        # All counts should be zero
        assert all(count == 0 for count in result["histogram_counts"])
        assert result["histogram_min"] == 0.0
        assert result["histogram_max"] == 255.0

    def test_histogram_float_image(self):
        """Test histogram on float image (non-uint8)."""
        # Create float image with values 0.0 to 1.0
        data = np.array([[0.0, 0.5, 1.0], [0.25, 0.75, 0.9]], dtype=np.float32)
        dask_data = da.from_array(data, chunks=(2, 3))
        
        record = record_from(dask_data, {"dim_order": "YX"})
        processor = HistogramProcessor()
        
        result = processor.run(record)
        
        # Should still produce 256-bin histogram
        assert "histogram_counts" in result
        assert len(result["histogram_counts"]) == 256
        
        # Min and max should reflect actual data range
        assert result["histogram_min"] == pytest.approx(0.0, rel=1e-5)
        assert result["histogram_max"] == pytest.approx(1.0, rel=1e-5)
        
        # Total counts should match number of pixels
        assert sum(result["histogram_counts"]) == 6

    def test_histogram_integer_non_uint8(self):
        """Test histogram on integer image that's not uint8."""
        # Create int16 image
        data = np.array([[0, 1000, 2000], [500, 1500, 3000]], dtype=np.int16)
        dask_data = da.from_array(data, chunks=(2, 3))
        
        record = record_from(dask_data, {"dim_order": "YX"})
        processor = HistogramProcessor()
        
        result = processor.run(record)
        
        # Should produce 256-bin histogram
        assert "histogram_counts" in result
        assert len(result["histogram_counts"]) == 256
        
        # Min and max should reflect actual data range
        assert result["histogram_min"] == pytest.approx(0.0, rel=1e-5)
        # For integer types, max_adj = max + 1, but histogram_max should be the actual max
        assert result["histogram_max"] == pytest.approx(3000.0, rel=1e-5)

