import pytest
import numpy as np
import dask.array as da
import cv2
from pixel_patrol_base.core.record import record_from
from pixel_patrol_image.plugins.processors.quality_metrics_processor import (
    QualityMetricsProcessor,
    _variance_of_laplacian_2d,
    _tenengrad_2d,
    _brenner_2d,
    _noise_estimation_2d,
    _check_blocking_records_2d,
    _check_ringing_records_2d,
)


class TestQualityMetricsProcessor:
    """Test suite for QualityMetricsProcessor to verify correct calculation of quality metrics."""

    # FIXME all numbers here need to be verified. For now, just recording them here to recognize if they change.

    def test_quality_metrics_simple_2d_image(self):
        """Test quality metrics calculation on a simple 2D image."""
        # Create a simple 2D image with some variation
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.uint8)
        dask_data = da.from_array(data, chunks=(3, 3))
        
        record = record_from(dask_data, {"dim_order": "YX"})
        processor = QualityMetricsProcessor()
        
        result = processor.run(record)
        
        # Verify all expected keys are present
        assert "laplacian_variance" in result
        assert "tenengrad" in result
        assert "brenner" in result
        assert "noise_std" in result
        assert "blocking_records" in result
        assert "ringing_records" in result
        
        # Verify all values are floats
        for key, value in result.items():
            assert isinstance(value, (float, np.floating)) or np.isnan(value)

        assert result["laplacian_variance"] == pytest.approx(26.66666, rel=1e-5)
        assert result["tenengrad"] == pytest.approx(9.92202377, rel=1e-5)
        assert result["brenner"] == pytest.approx(40.0, rel=1e-5)
        assert result["noise_std"] == pytest.approx(0.6666666, rel=1e-5)
        assert result["blocking_records"] == pytest.approx(0, rel=1e-5)
        assert result["ringing_records"] == pytest.approx(9.666666, rel=1e-5)

    def test_quality_metrics_constant_image(self):
        """Test quality metrics on a constant image."""
        data = np.full((10, 10), 128, dtype=np.uint8)
        dask_data = da.from_array(data, chunks=(5, 5))
        
        record = record_from(dask_data, {"dim_order": "YX"})
        processor = QualityMetricsProcessor()
        
        result = processor.run(record)
        
        assert result["laplacian_variance"] == pytest.approx(0.0, abs=1e-5)
        assert np.isnan(result["tenengrad"])
        assert result["brenner"] == pytest.approx(0.0, abs=1e-5)
        assert result["noise_std"] == pytest.approx(0.0, abs=1e-5)
        assert result["blocking_records"] == pytest.approx(0.0, abs=1e-5)
        assert result["ringing_records"] == pytest.approx(0.0, abs=1e-5)

    def test_quality_metrics_with_edges(self):
        """Test quality metrics on image with clear edges."""
        # Create image with a clear edge (step function)
        data = np.zeros((20, 20), dtype=np.uint8)
        data[:, 10:] = 255  # Right half is white
        dask_data = da.from_array(data, chunks=(10, 10))
        
        record = record_from(dask_data, {"dim_order": "YX"})
        processor = QualityMetricsProcessor()
        
        result = processor.run(record)

        assert result["laplacian_variance"] == 6502.5
        assert result["tenengrad"] == 102.0
        assert result["brenner"] == 7225.0
        assert result["noise_std"] == 0
        assert result["blocking_records"] == 0
        assert result["ringing_records"] == 16256.25

    def test_quality_metrics_empty_image(self):
        """Test quality metrics on empty image."""
        data = np.array([[]], dtype=np.uint8)
        dask_data = da.from_array(data, chunks=(1, 1))
        
        record = record_from(dask_data, {"dim_order": "YX"})
        processor = QualityMetricsProcessor()
        
        result = processor.run(record)
        
        # Should handle empty arrays gracefully
        # All metrics should be NaN for empty images
        for key in ["laplacian_variance", "tenengrad", "brenner", "noise_std", 
                    "blocking_records", "ringing_records"]:
            if key in result:
                assert np.isnan(result[key])
        
    def test_variance_of_laplacian_2d_simple_pattern(self):
        """Test laplacian variance on a simple known pattern."""
        # Create a simple checkerboard-like pattern
        image = np.array([[0, 255, 0], [255, 0, 255], [0, 255, 0]], dtype=np.float32)
        
        result = _variance_of_laplacian_2d(image)
        assert result == 1027555.5625

    def test_tenengrad_2d_function(self):
        """Test the _tenengrad_2d function directly."""
        # Create image with gradients
        image = np.zeros((10, 10), dtype=np.float32)
        image[:, :] = np.arange(10).reshape(1, -1)
        
        result = _tenengrad_2d(image)
        assert result == pytest.approx(6.4, rel=1e-5)

    def test_brenner_2d_function(self):
        """Test the _brenner_2d function directly."""
        # Create image with known pattern for predictable brenner value
        # Simple horizontal gradient: brenner uses 2-pixel spacing (img[:, 2:] - img[:, :-2])
        image = np.array([[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]], dtype=np.float32)
        
        result = _brenner_2d(image)

        assert result == pytest.approx(4.0, abs=0.01)

    def test_noise_estimation_2d_function(self):
        """Test the _noise_estimation_2d function directly."""
        # Create image with known noise pattern
        # Base image with added noise
        base = np.full((10, 10), 128.0, dtype=np.float32)
        noise = np.random.RandomState(42).randn(10, 10).astype(np.float32) * 10
        image = base + noise
        
        result = _noise_estimation_2d(image)

        assert result == pytest.approx(8.29397, rel=1e-5)

    def test_blocking_records_2d_function(self):
        """Test the _check_blocking_records_2d function directly."""
        # Create image with clear blocking artifacts (8x8 blocks with different values)
        image = np.zeros((16, 16), dtype=np.float32)
        image[0:8, 0:8] = 0.0    # Top-left block: 0
        image[0:8, 8:16] = 255.0  # Top-right block: 255
        image[8:16, 0:8] = 128.0  # Bottom-left block: 128
        image[8:16, 8:16] = 64.0  # Bottom-right block: 64
        
        result = _check_blocking_records_2d(image)

        assert result == 159.5

    def test_ringing_records_2d_function(self):
        """Test the _check_ringing_records_2d function directly."""
        # Create image with sharp edges (which can cause ringing)
        image = np.zeros((20, 20), dtype=np.float32)
        image[10:, :] = 255  # Sharp horizontal edge
        
        result = _check_ringing_records_2d(image)
        
        assert result == 16256.25

    def test_quality_metrics_boolean_image(self):
        """Test quality metrics on boolean image (should be handled)."""
        data = np.array([[True, False], [False, True]], dtype=bool)
        dask_data = da.from_array(data, chunks=(2, 2))
        
        record = record_from(dask_data, {"dim_order": "YX"})
        processor = QualityMetricsProcessor()
        
        result = processor.run(record)
        
        # Should handle boolean images (they get converted in the function)
        # All metrics should be present, though some may be NaN
        assert "laplacian_variance" in result
        assert "tenengrad" in result
        assert "brenner" in result

    def test_quality_metrics_with_multiple_dimensions(self):
        """Test quality metrics on image with T, C, Z dimensions."""
        # Create a TCZYX image: 2 time, 2 channels, 2 z-slices, 10x10 spatial
        data = np.random.randint(0, 256, size=(2, 2, 2, 10, 10), dtype=np.uint8)
        dask_data = da.from_array(data, chunks=(1, 1, 1, 5, 5))
        
        record = record_from(dask_data, {"dim_order": "TCZYX"})
        processor = QualityMetricsProcessor()
        
        result = processor.run(record)
        
        # Should have aggregated metrics
        assert "laplacian_variance" in result
        
        # Should have metrics for various dimension combinations
        assert any("laplacian_variance_t" in key for key in result.keys())
        assert any("laplacian_variance_c" in key for key in result.keys())
        assert any("laplacian_variance_z" in key for key in result.keys())

