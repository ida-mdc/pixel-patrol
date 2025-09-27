from pathlib import Path

import numpy as np
import pytest
import zarr
from zarr.storage import LocalStore
import polars as pl

from pixel_patrol_loader_bio.plugins.loaders.bioio_loader import BioIoLoader
from pixel_patrol_loader_bio.plugins.loaders.zarr_loader import ZarrLoader
from pixel_patrol_base.core.processing import get_all_record_properties, build_records_df
from pixel_patrol_base.plugin_registry import discover_processor_plugins


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

    arr = root.create_array(
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
    paths_df = build_records_df([parent_dir], selected_extensions='all', loader=ZarrLoader())
    zarr_rows = paths_df.filter(pl.col("path") == str(zarr_folder))

    assert not zarr_rows.is_empty(), "Zarr folder not found in paths_df"
    assert zarr_rows[0, "type"] == "file", "Zarr folder should be recognized as type 'file'"
    assert zarr_rows[0, "file_extension"] == "zarr", "Zarr folder should have 'zarr' as file_extension"


# test decorator + signature (edit existing test)
@pytest.mark.parametrize("loader", [ZarrLoader(), BioIoLoader()])
def test_extract_metadata_from_zarr_using_bioio(zarr_folder: Path, loader):
    """
    Test that extract_image_metadata can process a .zarr folder and returns valid metadata.
    """
    metadata = get_all_record_properties(zarr_folder, loader=loader, processors=discover_processor_plugins())

    assert isinstance(metadata, dict)

    assert metadata.get("dim_order") in ["TCZYXS", "TCZYX", "TCYX", "CZYX", "CXY", "TYX"]  # TODO: probably need to change so dim order is always TCZYXS
    assert metadata.get("dtype") == "uint16"
    assert metadata.get("T_size") == 1
    assert metadata.get("C_size") == 2
    assert metadata.get("Z_size") == 1
    assert metadata.get("Y_size") == 10
    assert metadata.get("X_size") == 10

    assert "num_pixels" in metadata and metadata["num_pixels"] == 1 * 2 * 1 * 10 * 10
    assert "shape" in metadata and metadata["shape"]  == [1, 2, 1, 10, 10]
    assert "ndim" in metadata and metadata["ndim"] == 5