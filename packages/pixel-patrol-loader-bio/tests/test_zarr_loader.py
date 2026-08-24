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
    assert rec.meta["size_C"] == 2
    assert rec.meta["size_Y"] == 10
    assert rec.meta["size_X"] == 10


def test_read_header(zarr_folder: Path, loader):
    info = loader.read_header(zarr_folder)
    assert info.shape == (2, 10, 10)
    assert info.n_images == 1
    assert info.dim_order == "CYX"


def test_is_folder_supported(zarr_folder: Path, loader):
    assert loader.is_folder_supported(zarr_folder) is True
    assert loader.is_folder_supported(zarr_folder.parent) is False


def test_load_invalid_path_raises(tmp_path: Path, loader):
    with pytest.raises(Exception):
        loader.load(tmp_path / "nonexistent.zarr")


def test_n_images_always_one(zarr_folder: Path, loader):
    info = loader.read_header(zarr_folder)
    assert info.n_images == 1


@pytest.fixture
def bioformats2raw_zarr(tmp_path: Path) -> Path:
    """bioformats2raw layout 3: root has {'bioformats2raw.layout': 3}, actual OME-NGFF is under 0/."""
    zarr_path = tmp_path / "test.ome.zarr"
    store = LocalStore(str(zarr_path))
    root = zarr.group(store=store)
    root.attrs.put({"bioformats2raw.layout": 3})

    sub = root.require_group("0")
    data = np.zeros((1, 2, 8, 16, 16), dtype="uint16")
    arr = sub.create_array("0", shape=data.shape, chunks=data.shape, dtype="uint16", overwrite=True)
    arr[:] = data
    sub.attrs.put({
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
    })
    return zarr_path


def test_load_bioformats2raw_layout3(bioformats2raw_zarr: Path, loader):
    rec = loader.load(bioformats2raw_zarr)
    assert rec.dim_order == "TCZYX"
    assert tuple(rec.data.shape) == (1, 2, 8, 16, 16)


def test_read_header_bioformats2raw_layout3(bioformats2raw_zarr: Path, loader):
    info = loader.read_header(bioformats2raw_zarr)
    assert info.shape == (1, 2, 8, 16, 16)
    assert info.dim_order == "TCZYX"
