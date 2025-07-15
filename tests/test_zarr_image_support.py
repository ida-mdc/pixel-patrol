from pathlib import Path

import numpy as np
import polars as pl
import pytest
import zarr
from bioio import BioImage
from zarr.storage import DirectoryStore as LocalStore

from pixel_patrol.core.image_operations_and_metadata import get_all_image_properties
from pixel_patrol.core.processing import build_paths_df

COMMON_REQUIRED_METADATA_COLS = [
    "dim_order",
    "dtype",
    "t_size", "c_size", "z_size", "y_size", "x_size",
    "mean_intensity", "std_intensity", "min_intensity", "max_intensity",
    "num_pixels", "shape", "ndim"
]



@pytest.fixture
def zarr_folder(tmp_path: Path) -> Path:
    """
    Create a minimal OME-Zarr folder with valid NGFF metadata using the modern LocalStore interface.
    Returns the .zarr folder path.
    """
    zarr_path = tmp_path / "project" / "test_image.zarr"
    zarr_path.parent.mkdir(parents=True, exist_ok=True)

    shape = (1, 2, 1, 10, 10)
    chunks = (1, 1, 1, 10, 10)
    dtype = "uint16"
    data = np.random.randint(0, 65535, size=shape, dtype=dtype)

    # Use LocalStore for compatibility with modern Zarr v3+
    store = LocalStore(str(zarr_path))
    root = zarr.group(store=store)

    arr = root.create_dataset(
        "0",
        shape=shape,
        chunks=chunks,
        dtype=dtype,
        overwrite=True
    )
    arr[:] = data

    # Add required NGFF metadata
    root.attrs.put({
        "multiscales": [{
            "version": "0.4",
            "datasets": [{"path": "0"}],

            "axes": [
                {"name": "t", "type": "time"},
                {"name": "c", "type": "channel"},
                {"name": "z", "type": "space"},
                {"name": "y", "type": "space"},
                {"name": "x", "type": "space"}
            ]
        }],
        "omero": {
            "channels": [
                {"label": "Channel 0"},
                {"label": "Channel 1"}
            ]
        }
    })

    return zarr_path

def test_zarr_path_recognition_as_image(zarr_folder: Path):
    """
    Test that a .zarr folder is correctly recognized and included in paths_df with type='file'.
    """
    parent_dir = zarr_folder.parent
    paths_df = build_paths_df([parent_dir])
    zarr_rows = paths_df.filter(pl.col("path") == str(zarr_folder))

    assert not zarr_rows.is_empty(), "Zarr folder not found in paths_df"
    assert zarr_rows[0, "type"] == "file", "Zarr folder should be recognized as type 'file'"
    assert zarr_rows[0, "file_extension"] == "zarr", "Zarr folder should have 'zarr' as file_extension"


def test_bioio_can_read_zarr(zarr_folder: Path):
    """
    Tests whether bioio.imread can read a valid OME-Zarr dataset.
    """
    try:
        image = BioImage(zarr_folder).data
    except Exception as e:
        pytest.fail(f"bioio.imread failed to read .zarr: {e}")

    assert image is not None
    assert hasattr(image, "shape")
    assert image.shape == (1, 2, 1, 10, 10)


def test_extract_image_metadata_from_zarr(zarr_folder: Path):
    """
    Test that extract_image_metadata can process a .zarr folder and returns valid metadata.
    """
    metadata = get_all_image_properties(zarr_folder, COMMON_REQUIRED_METADATA_COLS)

    assert isinstance(metadata, dict)

    assert metadata.get("dim_order") in ["TCZYXS", "TCZYX", "TCYX", "CZYX", "CXY", "TYX"]  # TODO: probably need to change so dim order is always TCZYXS
    assert metadata.get("dtype") == "uint16"
    assert metadata.get("t_size") == 1
    assert metadata.get("c_size") == 2
    assert metadata.get("z_size") == 1
    assert metadata.get("y_size") == 10
    assert metadata.get("x_size") == 10

    assert "num_pixels" in metadata and metadata["num_pixels"] == 1 * 2 * 1 * 10 * 10
    assert "shape" in metadata and metadata["shape"] in [(1, 2, 1, 10, 10, 1), str((1, 2, 1, 10, 10, 1))]
    assert "ndim" in metadata and metadata["ndim"] == 6