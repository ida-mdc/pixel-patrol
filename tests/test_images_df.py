import polars as pl
from pathlib import Path
from typing import Dict
from datetime import datetime
import pytest

from pixel_patrol.core.image_operations_and_metadata import get_all_image_properties
from pixel_patrol.utils.utils import format_bytes_to_human_readable


COMMON_REQUIRED_METADATA_COLS = [
    "width", "height", "n_channels", "dim_order",
    "mean_intensity", "std_intensity", "min_intensity",
    "max_intensity", "num_pixels", "dtype", "shape", "ndim"
]

@pytest.fixture
def static_test_images(request) -> Dict[str, Path]:
    """
    Provides paths to pre-existing static test image files.
    Assumes `tests/data/test_image.png` and `tests/data/test_image.tif` exist.
    """
    # Use request.fspath to get the path of the current test file, then navigate
    # to the 'data' directory relative to the 'tests' directory.
    test_dir = Path(request.fspath).parent
    data_dir = test_dir / "data"

    # Make sure the files exist before returning their paths
    png_path = data_dir / "test_image.png"
    tif_path = data_dir / "test_image.tif" # Assuming this is your BioImage test file
    invalid_path = data_dir / "not_an_image.txt" # Assuming you have a non-image file for error testing

    if not png_path.exists():
        pytest.fail(f"Test image not found: {png_path}")
    if not tif_path.exists():
        pytest.fail(f"Test image not found: {tif_path}")
    if not invalid_path.exists():
        invalid_path.write_text("This is deliberately not an image file.")

    return {
        "png": png_path,
        "tif": tif_path,
        "invalid": invalid_path,
        "non_existent": data_dir / "this_file_does_not_exist.xyz" # For non-existent file tests
    }


def test_get_all_image_properties_png(static_test_images: Dict[str, Path]):
    pass


def test_get_all_image_properties_tif(static_test_images: Dict[str, Path]):
    pass


def test_get_all_image_properties_invalid_file(static_test_images: Dict[str, Path]):
    invalid_path = static_test_images["invalid"]
    metadata = get_all_image_properties(invalid_path, COMMON_REQUIRED_METADATA_COLS)

    assert isinstance(metadata, dict)
    assert not metadata # Should be empty


def test_get_all_image_properties_non_existent_file(static_test_images: Dict[str, Path]):
    non_existent_path = static_test_images["non_existent"]
    metadata = get_all_image_properties(non_existent_path, COMMON_REQUIRED_METADATA_COLS)

    assert isinstance(metadata, dict)
    assert not metadata # Should be empty


# TODO: should be / already in conftest?
@pytest.fixture
def mock_processing_paths_df(tmp_path):
    fixed_dt = datetime(2023, 1, 1, 12, 0, 0)

    imported_root_1 = tmp_path / "project_root"
    imported_root_1.mkdir()  # Ensure this directory exists for the test
    (imported_root_1 / "subdir_a").mkdir()
    imported_root_2 = tmp_path / "another_root"
    imported_root_2.mkdir()

    data = {
        "path": [
            str(tmp_path / "image1.jpg"),
            str(tmp_path / "subdir_a" / "image2.png"),
            str(tmp_path / "another_root" / "image3.jpeg"), # Adjust if another_root isn't under tmp_path
            str(tmp_path / "document.pdf"),
            str(tmp_path / "video.mp4"),
            str(tmp_path / "subdir_b" / "text.txt")
        ],
        "name": [
            "image1.jpg", "image2.png", "image3.jpeg",
            "document.pdf", "video.mp4", "text.txt"
        ],
        "type": [
            "file", "file", "file",
            "file", "file", "file"
        ],
        "parent": [
            str(tmp_path), str(tmp_path / "subdir_a"), str(tmp_path / "another_root"),
            str(tmp_path), str(tmp_path), str(tmp_path / "subdir_b")
        ],
        "depth": [1, 2, 1, 1, 1, 2],
        "size_bytes": [100, 200, 150, 500, 1000, 50],
        "modification_date": [fixed_dt, fixed_dt, fixed_dt, fixed_dt, fixed_dt, fixed_dt],
        "file_extension": ["jpg", "png", "jpeg", "pdf", "mp4", "txt"],
        # <--- THIS IS THE CRUCIAL PART ---
        "size_readable": [
            format_bytes_to_human_readable(100), # Use the actual function to ensure consistency
            format_bytes_to_human_readable(200),
            format_bytes_to_human_readable(150),
            format_bytes_to_human_readable(500),
            format_bytes_to_human_readable(1000),
            format_bytes_to_human_readable(50)
        ],
        "imported_path": [
            str(imported_root_1), # For image1.jpg
            str(imported_root_1), # For image2.png
            str(imported_root_2), # For image3.jpeg
            str(imported_root_1), # For document.pdf
            str(imported_root_1), # For video.mp4
            str(imported_root_1)  # For text.txt
        ]
    }
    return pl.DataFrame(data)


def test_build_images_df_empty_paths_df(mock_settings, mocker):
    pass

def test_build_images_df_no_image_files_after_filtering(
    mock_processing_paths_df: pl.DataFrame,
    mock_settings,
    mocker
):
    pass