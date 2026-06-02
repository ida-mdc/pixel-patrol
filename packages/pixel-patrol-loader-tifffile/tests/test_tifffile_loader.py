"""Tests for :class:`TifffileLoader`."""

from pathlib import Path

import numpy as np
import pytest
import tifffile

from pixel_patrol_loader_tifffile.plugins.loaders.tifffile_loader import TifffileLoader


@pytest.fixture
def loader():
    return TifffileLoader()


def test_name_and_extensions(loader):
    assert loader.NAME == "tifffile"
    assert "ome.tif" in loader.SUPPORTED_EXTENSIONS
    assert "tif" in loader.CONTAINER_EXTENSIONS
    assert "tiff" in loader.CONTAINER_EXTENSIONS


def test_load_cyx_imagej_roundtrip(tmp_path: Path, loader):
    path = tmp_path / "cyx.tif"
    rng = np.random.default_rng(0)
    im = rng.integers(0, 65535, size=(3, 24, 32), dtype=np.uint16)
    tifffile.imwrite(path, im, imagej=True, metadata={"axes": "CYX"})

    rec = loader.load(path)
    assert rec.dim_order == "CYX"
    assert tuple(rec.data.shape) == (3, 24, 32)
    assert rec.meta["C_size"] == 3
    assert rec.meta["Y_size"] == 24
    assert rec.meta["X_size"] == 32
    np.testing.assert_array_almost_equal(im, rec.data.compute())


def test_read_header(tmp_path: Path, loader):
    path = tmp_path / "cyx.tif"
    im = np.zeros((3, 24, 32), dtype=np.uint16)
    tifffile.imwrite(path, im, imagej=True, metadata={"axes": "CYX"})

    info = loader.read_header(path)
    assert info.shape == (3, 24, 32)
    assert info.n_images == 1
    assert tuple(info.dim_order) == ("C", "Y", "X")


def test_read_header_multi_series(tmp_path: Path, loader):
    path = tmp_path / "multi.tif"
    a = np.zeros((2, 4, 4), dtype=np.uint8)
    b = np.ones((2, 4, 4), dtype=np.uint8)
    with tifffile.TiffWriter(path) as tw:
        tw.write(a, metadata={"axes": "CYX"})
        tw.write(b, metadata={"axes": "CYX"})

    info = loader.read_header(path)
    assert info.n_images == 2


def test_load_multi_series_load_range(tmp_path: Path, loader):
    path = tmp_path / "multi.tif"
    a = np.zeros((2, 4, 4), dtype=np.uint8)
    b = np.ones((2, 4, 4), dtype=np.uint8)
    with tifffile.TiffWriter(path) as tw:
        tw.write(a, metadata={"axes": "CYX"})
        tw.write(b, metadata={"axes": "CYX"})

    results = dict(loader.load_range(path, 0, 2))
    assert set(results.keys()) == {"0", "1"}
    np.testing.assert_array_equal(results["0"].data.compute(), a)
    np.testing.assert_array_equal(results["1"].data.compute(), b)


def test_dask_chunks_reasonable(tmp_path: Path, loader):
    path = tmp_path / "chunked.tif"
    im = np.zeros((2, 64, 64), dtype=np.uint16)
    tifffile.imwrite(path, im, tile=(64, 64), imagej=True, metadata={"axes": "CYX"})
    rec = loader.load(path)
    ch = rec.data.chunks
    assert len(ch) == 3
    assert all(c is not None for c in ch)
