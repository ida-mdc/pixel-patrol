from pathlib import Path

import numpy as np
import polars as pl
import pytest
import zarr
from zarr.storage import LocalStore

from pixel_patrol_loader_bio.plugins.loaders.bioio_loader import BioIoLoader
from pixel_patrol_loader_bio.plugins.loaders.zarr_loader import ZarrLoader
from pixel_patrol_base.core.processing import build_records_df
from pixel_patrol_base.core.processing_config import ProcessingConfig
from pixel_patrol_base.plugin_registry import discover_processor_plugins


@pytest.fixture
def zarr_folder(tmp_path: Path) -> Path:
    zarr_path = tmp_path / "project" / "test_image.zarr"
    zarr_path.parent.mkdir(parents=True, exist_ok=True)

    shape = (1, 2, 1, 10, 10)
    chunks = (1, 1, 1, 10, 10)
    dtype = "uint16"
    data = np.random.randint(0, 65535, size=shape, dtype=dtype)

    store = LocalStore(str(zarr_path))
    root = zarr.group(store=store)

    arr = root.create_array("0", shape=shape, chunks=chunks, dtype=dtype, overwrite=True)
    arr[:] = data

    root.attrs.put({
        "multiscales": [{
            "version": "0.4",
            "datasets": [{"path": "0"}],
            "axes": [
                {"name": "t", "type": "time"},
                {"name": "c", "type": "channel"},
                {"name": "z", "type": "space"},
                {"name": "y", "type": "space"},
                {"name": "x", "type": "space"},
            ],
        }],
        "omero": {"channels": [{"label": "Channel 0"}, {"label": "Channel 1"}]},
    })

    return zarr_path


def test_zarr_path_recognition_as_file(zarr_folder: Path):
    parent_dir = zarr_folder.parent
    loader = ZarrLoader()
    df, _ = build_records_df(
        bases=[parent_dir],
        loader=loader,
        processors=discover_processor_plugins(),
    )
    assert df is not None
    zarr_rows = df.filter(pl.col("path") == str(zarr_folder))
    assert not zarr_rows.is_empty(), "Zarr folder not found in result"
    assert zarr_rows[0, "file_extension"] == "zarr"


@pytest.mark.parametrize("loader", [ZarrLoader(), BioIoLoader()])
def test_extract_metadata_from_zarr(zarr_folder: Path, loader):
    processors = discover_processor_plugins()
    df, _ = build_records_df(
        bases=[zarr_folder.parent],
        loader=loader,
        processors=processors,
    )
    assert df is not None
    rows = df.filter(pl.col("obs_level") == 0).to_dicts()
    assert len(rows) >= 1

    metadata = rows[0]
    assert isinstance(metadata, dict)

    # Both loaders: zarr is recognized and intensity metrics are computed
    assert metadata.get("file_extension") == "zarr"
    assert metadata.get("num_pixels") == 1 * 2 * 1 * 10 * 10
    assert metadata.get("min_intensity") is not None
    assert metadata.get("max_intensity") is not None
