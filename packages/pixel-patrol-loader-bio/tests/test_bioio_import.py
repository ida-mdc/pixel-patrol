from pathlib import Path
from typing import Dict, Any

import numpy as np
import pytest

from pixel_patrol_loader_bio.config import STANDARD_DIM_ORDER
from pixel_patrol_loader_bio.plugins.loaders.bioio_loader import BioIoLoader
from pixel_patrol_base.config import SPRITE_SIZE
from pixel_patrol_base.core.processing import load_and_process_records_from_file
from pixel_patrol_base.plugin_registry import discover_processor_plugins


@pytest.fixture(scope="module")
def loader():
    return BioIoLoader()

@pytest.fixture(scope="module")
def processors():
    return discover_processor_plugins()

@pytest.fixture(scope="module")
def standard_dim_order():
    return STANDARD_DIM_ORDER


def get_image_files_from_data_dir(test_data_dir: Path):
    """Helper to get all image files from the test_data_dir, excluding non-image files."""
    return [f for f in test_data_dir.rglob("*")
            if f.is_file() and f.name != "not_an_image.txt"]


def test_nonexistent_path_raises(tmp_path, loader, processors):
    missing = tmp_path / "nope.tiny_png"
    assert load_and_process_records_from_file(missing, loader=loader, processors=processors) == []


def test_unsupported_file(test_data_dir: Path, loader, processors):
    non_image_file = test_data_dir / "not_an_image.txt"
    result = load_and_process_records_from_file(non_image_file, loader=loader, processors=processors)
    assert result == [], f"Expected empty list for non-image file, got {result}"


def test_empty_or_corrupt_image(tmp_path, loader, processors):
    f = tmp_path / "zero.tif"
    f.write_bytes(b"")
    assert load_and_process_records_from_file(f, loader=loader, processors=processors) == []


def test_bioio_image_properties_per_file(
    image_file_path: Path,
    loader, processors,
    standard_dim_order: str,
    expected_image_data: Dict[str, Dict[str, Any]]
):
    """
    Detailed metadata checks for files that have expectations in `expected_image_data`,
    and general sanity checks for all other files.
    """
    file_name = image_file_path.name
    result = load_and_process_records_from_file(image_file_path, loader=loader, processors=processors)

    assert result is not None, f"Failed to get properties for {file_name}"
    assert result != [], f"Properties list is empty for {file_name}"

    actual_properties = result[0]
    # Always enforce core presence
    assert "shape" in actual_properties, f"Missing 'shape' for {file_name}"
    assert "dtype" in actual_properties, f"Missing 'dtype' for {file_name}"
    assert "ndim" in actual_properties, f"Missing 'ndim' for {file_name}"

    # If we have expectations for this file, check them specifically.
    expected = expected_image_data.get(file_name)
    if expected:
        if "dtype" in expected:
            assert expected["dtype"] in str(actual_properties["dtype"]), \
                f"Dtype mismatch for {file_name}: expected contains {expected['dtype']}, got {actual_properties['dtype']}"
        if "min_ndim" in expected:
            assert actual_properties["ndim"] >= expected["min_ndim"], \
                f"ndim too small for {file_name}: expected >= {expected['min_ndim']}, got {actual_properties['ndim']}"


def test_all_image_files_load_and_standardize(
    image_file_path: Path,
    loader, processors,
    standard_dim_order: str
):
    """
    Ensure all image files can be loaded by bioio, standardized, and a thumbnail generated.
    """
    file_name = image_file_path.name
    result = load_and_process_records_from_file(image_file_path, loader=loader, processors=processors)


    assert result is not None, f"Failed to get properties for {file_name}"
    assert result != [], f"Properties list is empty for {file_name}"

    # Check each record returned (single-record files will have exactly one)
    for i, properties in enumerate(result):
        record_label = f"{file_name}[{i}]" if len(result) > 1 else file_name

        assert "dim_order" in properties, f"Missing dim_order for {record_label}"
        assert "thumbnail" in properties, f"Missing thumbnail for {record_label}"
        assert isinstance(properties["thumbnail"], np.ndarray), f"Thumbnail not a numpy array for {record_label}"
        assert properties["thumbnail"].shape == (SPRITE_SIZE, SPRITE_SIZE), \
            f"Thumbnail size mismatch for {record_label}: Expected ({SPRITE_SIZE}, {SPRITE_SIZE}), Got {properties['thumbnail'].shape}"

        assert "shape" in properties, f"Missing shape for {record_label}"
        assert "dtype" in properties, f"Missing dtype for {record_label}"
        assert "ndim" in properties, f"Missing ndim for {record_label}"
