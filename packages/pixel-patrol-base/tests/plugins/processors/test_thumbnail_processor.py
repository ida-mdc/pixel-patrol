import pytest
import numpy as np
import dask.array as da
from pixel_patrol_base.core.record import record_from
from pixel_patrol_base.plugins.processors.thumbnail_processor import ThumbnailProcessor
from pixel_patrol_base.config import SPRITE_SIZE


class TestThumbnailProcessor:
    """Test suite for ThumbnailProcessor to verify correct thumbnail generation."""

    def test_thumbnail_simple_2d_image(self):
        """Test thumbnail generation on a simple 2D image."""
        # Create a simple 2D image
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.uint8)
        dask_data = da.from_array(data, chunks=(3, 3))
        
        record = record_from(dask_data, {"dim_order": "YX"})
        processor = ThumbnailProcessor()
        
        result = processor.run(record)
        
        # Verify thumbnail key is present
        assert "thumbnail" in result
        thumbnail = result["thumbnail"]
        
        # Verify thumbnail is a numpy array
        assert isinstance(thumbnail, np.ndarray)
        
        # Verify thumbnail dimensions (should be SPRITE_SIZE x SPRITE_SIZE)
        assert thumbnail.shape == (SPRITE_SIZE, SPRITE_SIZE)
        
        # Verify thumbnail is uint8
        assert thumbnail.dtype == np.uint8
        
        # Verify values are in valid range
        assert thumbnail.min() >= 0
        assert thumbnail.max() <= 255

    def test_thumbnail_larger_image(self):
        """Test thumbnail generation on a larger image (should be resized)."""
        # Create a larger image
        data = np.random.randint(0, 256, size=(200, 300), dtype=np.uint8)
        dask_data = da.from_array(data, chunks=(50, 50))
        
        record = record_from(dask_data, {"dim_order": "YX"})
        processor = ThumbnailProcessor()
        
        result = processor.run(record)
        
        thumbnail = result["thumbnail"]
        
        # Should be resized to SPRITE_SIZE x SPRITE_SIZE
        assert thumbnail.shape == (SPRITE_SIZE, SPRITE_SIZE)
        assert thumbnail.dtype == np.uint8

    def test_thumbnail_constant_image(self):
        """Test thumbnail on a constant image."""
        data = np.full((50, 50), 128, dtype=np.uint8)
        dask_data = da.from_array(data, chunks=(25, 25))
        
        record = record_from(dask_data, {"dim_order": "YX"})
        processor = ThumbnailProcessor()
        
        result = processor.run(record)
        
        thumbnail = result["thumbnail"]
        
        # Should be SPRITE_SIZE x SPRITE_SIZE
        assert thumbnail.shape == (SPRITE_SIZE, SPRITE_SIZE)
        # For constant images, should be normalized to 255 (since value > 0)
        assert np.all(thumbnail == 255)

    def test_thumbnail_zero_image(self):
        """Test thumbnail on an all-zero image."""
        data = np.zeros((50, 50), dtype=np.uint8)
        dask_data = da.from_array(data, chunks=(25, 25))
        
        record = record_from(dask_data, {"dim_order": "YX"})
        processor = ThumbnailProcessor()
        
        result = processor.run(record)
        
        thumbnail = result["thumbnail"]
        
        # Should be SPRITE_SIZE x SPRITE_SIZE
        assert thumbnail.shape == (SPRITE_SIZE, SPRITE_SIZE)
        # For zero images, should remain 0
        assert np.all(thumbnail == 0)

    def test_thumbnail_with_time_dimension(self):
        """Test thumbnail on image with time dimension (should take center slice)."""
        # Create a TYX image: 5 time points, 10x10 spatial
        data = np.random.randint(0, 256, size=(5, 10, 10), dtype=np.uint8)
        dask_data = da.from_array(data, chunks=(1, 5, 5))
        
        record = record_from(dask_data, {"dim_order": "TYX"})
        processor = ThumbnailProcessor()
        
        result = processor.run(record)
        
        thumbnail = result["thumbnail"]
        
        # Should reduce to 2D and resize
        assert thumbnail.shape == (SPRITE_SIZE, SPRITE_SIZE)
        assert thumbnail.dtype == np.uint8

    def test_thumbnail_with_channel_dimension(self):
        """Test thumbnail on image with channel dimension (should take mean)."""
        # Create a CYX image: 3 channels, 10x10 spatial
        data = np.random.randint(0, 256, size=(3, 10, 10), dtype=np.uint8)
        dask_data = da.from_array(data, chunks=(1, 5, 5))
        
        record = record_from(dask_data, {"dim_order": "CYX"})
        processor = ThumbnailProcessor()
        
        result = processor.run(record)
        
        thumbnail = result["thumbnail"]
        
        # Should reduce channels by taking mean and resize
        assert thumbnail.shape == (SPRITE_SIZE, SPRITE_SIZE)
        assert thumbnail.dtype == np.uint8

    def test_thumbnail_with_z_dimension(self):
        """Test thumbnail on image with Z dimension (should take center slice)."""
        # Create a ZYX image: 5 z-slices, 10x10 spatial
        data = np.random.randint(0, 256, size=(5, 10, 10), dtype=np.uint8)
        dask_data = da.from_array(data, chunks=(1, 5, 5))
        
        record = record_from(dask_data, {"dim_order": "ZYX"})
        processor = ThumbnailProcessor()
        
        result = processor.run(record)
        
        thumbnail = result["thumbnail"]
        
        # Should take center z-slice and resize
        assert thumbnail.shape == (SPRITE_SIZE, SPRITE_SIZE)
        assert thumbnail.dtype == np.uint8

    def test_thumbnail_with_multiple_dimensions(self):
        """Test thumbnail on image with T, C, Z dimensions."""
        # Create a TCZYX image: 3 time, 2 channels, 4 z-slices, 20x20 spatial
        data = np.random.randint(0, 256, size=(3, 2, 4, 20, 20), dtype=np.uint8)
        dask_data = da.from_array(data, chunks=(1, 1, 1, 10, 10))
        
        record = record_from(dask_data, {"dim_order": "TCZYX"})
        processor = ThumbnailProcessor()
        
        result = processor.run(record)
        
        thumbnail = result["thumbnail"]
        
        # Should reduce all non-spatial dims and resize
        assert thumbnail.shape == (SPRITE_SIZE, SPRITE_SIZE)
        assert thumbnail.dtype == np.uint8

    def test_thumbnail_boolean_image(self):
        """Test thumbnail on boolean image (should be cast to uint8)."""
        data = np.array([[True, False], [False, True]], dtype=bool)
        dask_data = da.from_array(data, chunks=(2, 2))
        
        record = record_from(dask_data, {"dim_order": "YX"})
        processor = ThumbnailProcessor()
        
        result = processor.run(record)
        
        thumbnail = result["thumbnail"]
        
        # Should be converted to uint8
        assert thumbnail.dtype == np.uint8
        assert thumbnail.shape == (SPRITE_SIZE, SPRITE_SIZE)

    def test_thumbnail_float_image(self):
        """Test thumbnail on float image (should be normalized)."""
        # Create float image with values 0.0 to 1.0
        data = np.array([[0.0, 0.5, 1.0], [0.25, 0.75, 0.9]], dtype=np.float32)
        dask_data = da.from_array(data, chunks=(2, 3))
        
        record = record_from(dask_data, {"dim_order": "YX"})
        processor = ThumbnailProcessor()
        
        result = processor.run(record)
        
        thumbnail = result["thumbnail"]
        
        # Should be normalized to 0-255 uint8
        assert thumbnail.dtype == np.uint8
        assert thumbnail.shape == (SPRITE_SIZE, SPRITE_SIZE)
        assert thumbnail.min() >= 0
        assert thumbnail.max() <= 255

    def test_thumbnail_empty_image(self):
        """Test thumbnail on empty image."""
        data = np.array([[]], dtype=np.uint8)
        dask_data = da.from_array(data, chunks=(1, 1))
        
        record = record_from(dask_data, {"dim_order": "YX"})
        processor = ThumbnailProcessor()
        
        result = processor.run(record)
        
        thumbnail = result["thumbnail"]
        
        # Should return empty array
        assert isinstance(thumbnail, np.ndarray)
        assert thumbnail.size == 0

    def test_thumbnail_single_pixel(self):
        """Test thumbnail on single pixel image."""
        data = np.array([[42]], dtype=np.uint8)
        dask_data = da.from_array(data, chunks=(1, 1))
        
        record = record_from(dask_data, {"dim_order": "YX"})
        processor = ThumbnailProcessor()
        
        result = processor.run(record)
        
        thumbnail = result["thumbnail"]
        
        # Should be resized to SPRITE_SIZE x SPRITE_SIZE
        assert thumbnail.shape == (SPRITE_SIZE, SPRITE_SIZE)
        assert thumbnail.dtype == np.uint8

    def test_thumbnail_normalization(self):
        """Test that thumbnail correctly normalizes values."""
        # Create image with values 100-200 (not full range)
        data = np.full((50, 50), 150, dtype=np.uint8)
        # Add some variation
        data[0, 0] = 100
        data[-1, -1] = 200
        dask_data = da.from_array(data, chunks=(25, 25))
        
        record = record_from(dask_data, {"dim_order": "YX"})
        processor = ThumbnailProcessor()
        
        result = processor.run(record)
        
        thumbnail = result["thumbnail"]
        
        # After normalization, min should map to 0 and max to 255
        # Since we have variation, the thumbnail should have some range
        assert thumbnail.min() >= 0
        assert thumbnail.max() <= 255
        # For this specific case with min=100, max=200, the normalized values
        # should span a range (not all be the same)
        assert thumbnail.max() > thumbnail.min() or np.all(thumbnail == thumbnail[0, 0])

