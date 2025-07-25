import pytest
from pathlib import Path
import numpy as np
from typing import Dict, Any

# Assuming these are accessible from your project root or via PYTHONPATH
from pixel_patrol.core.image_operations_and_metadata import get_all_image_properties, available_columns
from pixel_patrol.config import STANDARD_DIM_ORDER, SPRITE_SIZE


@pytest.fixture(scope="module")
def image_properties_to_check():
    """Define a set of all available image properties we expect to extract."""
    return available_columns()

@pytest.fixture(scope="module")
def standard_dim_order():
    """Provide the standard dimension order from your config."""
    return STANDARD_DIM_ORDER


def get_image_files_from_data_dir(test_data_dir: Path):
    """Helper to get all image files from the test_data_dir, excluding non-image files."""
    return [f for f in test_data_dir.rglob("*")
        if f.is_file() and f.name != "not_an_image.txt"]


def test_nonexistent_path_raises(tmp_path, image_properties_to_check):
    missing = tmp_path / "nope.png"
    assert get_all_image_properties(missing, required_columns=image_properties_to_check) == {}


def test_unsupported_file(test_data_dir: Path, image_properties_to_check: list):
    """
    Test that a non-image file returns an empty dictionary, indicating it's not processed as an image.
    """
    non_image_file = test_data_dir / "not_an_image.txt"
    properties = get_all_image_properties(non_image_file, required_columns=image_properties_to_check)
    assert properties == {}, f"Expected empty dict for non-image file, got {properties}"


def test_empty_or_corrupt_image(tmp_path, image_properties_to_check):
    f = tmp_path / "zero.tif"
    f.write_bytes(b"")
    assert get_all_image_properties(f, required_columns=image_properties_to_check) == {}


@pytest.mark.parametrize("image_file_path", get_image_files_from_data_dir(Path(__file__).parent / "data"))
def test_bioio_image_properties_per_file(
    image_file_path: Path,
    image_properties_to_check: list,
    standard_dim_order: str, # This fixture is still useful for direct comparison
    expected_image_data: Dict[str, Dict[str, Any]]
):
    """
    Tests detailed metadata extraction and dimension standardization for each specific image file.
    This test verifies the combined output of get_all_image_properties, which includes
    both BioImage derived properties and standardized array properties.
    """
    file_name = image_file_path.name
    expected_props = expected_image_data.get(file_name)

    assert expected_props is not None, f"No expected data defined for {file_name}. Add it to expected_image_data fixture."

    # Get properties using your function
    actual_properties = get_all_image_properties(image_file_path, required_columns=image_properties_to_check)

    assert actual_properties is not None, f"Failed to get properties for {file_name}"
    assert actual_properties != {}, f"Properties dictionary is empty for {file_name}"

    # Assert expected properties based on the combined output
    for prop, expected_value in expected_props.items():
        if prop == "shape":
            # Convert string representation of shape to tuple for comparison
            actual_shape_str = actual_properties.get("shape")
            assert actual_shape_str is not None, f"Missing 'shape' property for {file_name}"
            actual_shape_tuple = tuple(map(int, actual_shape_str.strip('()').split(',')))
            assert actual_shape_tuple == expected_value, \
                f"Shape mismatch for {file_name}: Expected {expected_value}, Got {actual_shape_tuple}"
        elif prop == "dtype":
            # Check if the actual dtype string contains the expected dtype string
            actual_dtype_str = actual_properties.get("dtype")
            assert actual_dtype_str is not None, f"Missing 'dtype' property for {file_name}"
            assert expected_value in actual_dtype_str, \
                f"Dtype mismatch for {file_name}: Expected '{expected_value}' in '{actual_dtype_str}'"
        else:
            assert actual_properties.get(prop) == expected_value, \
                f"Property '{prop}' mismatch for {file_name}: Expected {expected_value}, Got {actual_properties.get(prop)}"

    # Additional check for ndim property as it's critical for standardized output
    assert actual_properties["ndim"] == expected_props["ndim"], \
        f"ndim mismatch for {file_name}: Expected {expected_props['ndim']}, Got {actual_properties['ndim']}"


@pytest.mark.parametrize("image_file_path", get_image_files_from_data_dir(Path(__file__).parent / "data"))
def test_all_image_files_load_and_standardize(
    image_file_path: Path,
    image_properties_to_check: list,
    standard_dim_order: str
):
    """
    General test to ensure all image files can be loaded by bioio,
    have their dimension order standardized, and a thumbnail generated.
    This test focuses on the existence and basic correctness of core properties.
    """
    file_name = image_file_path.name
    properties = get_all_image_properties(image_file_path, required_columns=image_properties_to_check)

    assert properties is not None, f"Failed to get properties for {file_name}"
    assert properties != {}, f"Properties dictionary is empty for {file_name}"

    # Check for core properties expected from any successful image load and standardization
    assert "dim_order" in properties, f"Missing dim_order for {file_name}"
    assert properties["dim_order"] == standard_dim_order, \
        f"Dimension order not standardized for {file_name}: Expected {standard_dim_order}, Got {properties['dim_order']}"

    assert "thumbnail" in properties, f"Missing thumbnail for {file_name}"
    assert isinstance(properties["thumbnail"], np.ndarray), f"Thumbnail not a numpy array for {file_name}"
    assert properties["thumbnail"].shape == (SPRITE_SIZE, SPRITE_SIZE), \
        f"Thumbnail size mismatch for {file_name}: Expected ({SPRITE_SIZE}, {SPRITE_SIZE}), Got {properties['thumbnail'].shape}"

    assert "shape" in properties, f"Missing shape for {file_name}"
    assert "dtype" in properties, f"Missing dtype for {file_name}"
    assert "ndim" in properties, f"Missing ndim for {file_name}"
    assert properties["ndim"] == 6, f"ndim not 6 after standardization for {file_name}"

