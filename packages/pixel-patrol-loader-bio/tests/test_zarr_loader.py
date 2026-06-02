from pathlib import Path

import numpy as np
import pytest
import zarr
from zarr.storage import LocalStore

from pixel_patrol_loader_bio.plugins.loaders.zarr_loader import ZarrLoader


@pytest.fixture
def loader():
    return ZarrLoader()


@pytest.fixture
def zarr_folder(tmp_path: Path) -> Path:
    zarr_path = tmp_path / "test_image.zarr"
    store = LocalStore(str(zarr_path))
    root = zarr.group(store=store)
    data = np.random.randint(0, 65535, size=(2, 10, 10), dtype="uint16")
    arr = root.create_array("0", shape=data.shape, chunks=data.shape, dtype="uint16", overwrite=True)
    arr[:] = data
    root.attrs.put({
        "multiscales": [{
            "version": "0.4",
            "datasets": [{"path": "0"}],
            "axes": [
                {"name": "c", "type": "channel"},
                {"name": "y", "type": "space"},
                {"name": "x", "type": "space"},
            ],
        }],
    })
    return zarr_path


def test_name_and_extensions(loader):
    assert loader.NAME == "zarr"
    assert "zarr" in loader.SUPPORTED_EXTENSIONS
    assert "ome.zarr" in loader.SUPPORTED_EXTENSIONS
    assert "zarr" in loader.FOLDER_EXTENSIONS
    assert loader.CONTAINER_EXTENSIONS == set()


def test_load(zarr_folder: Path, loader):
    rec = loader.load(zarr_folder)
    assert rec.dim_order == "CYX"
    assert tuple(rec.data.shape) == (2, 10, 10)
    assert rec.meta["C_size"] == 2
    assert rec.meta["Y_size"] == 10
    assert rec.meta["X_size"] == 10


def test_read_header(zarr_folder: Path, loader):
    info = loader.read_header(zarr_folder)
    assert info.shape == (2, 10, 10)
    assert info.n_images == 1
    assert tuple(info.dim_order) == ("C", "Y", "X")


def test_is_folder_supported(zarr_folder: Path, loader):
    assert loader.is_folder_supported(zarr_folder) is True
    assert loader.is_folder_supported(zarr_folder.parent) is False
