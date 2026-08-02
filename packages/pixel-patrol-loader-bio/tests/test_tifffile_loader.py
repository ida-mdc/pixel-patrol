"""Tests for :class:`TifffileLoader`."""

import logging
from pathlib import Path

import numpy as np
import pytest
import tifffile

from pixel_patrol_loader_bio.plugins.loaders.tifffile_loader import TifffileLoader


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
    assert rec.meta["size_C"] == 3
    assert rec.meta["size_Y"] == 24
    assert rec.meta["size_X"] == 32
    np.testing.assert_array_almost_equal(im, rec.data.compute())


def test_read_header(tmp_path: Path, loader):
    path = tmp_path / "cyx.tif"
    im = np.zeros((3, 24, 32), dtype=np.uint16)
    tifffile.imwrite(path, im, imagej=True, metadata={"axes": "CYX"})

    info = loader.read_header(path)
    assert info.shape == (3, 24, 32)
    assert info.n_images == 1
    assert info.dim_order == "CYX"


def test_read_header_multi_series(tmp_path: Path, loader):
    path = tmp_path / "multi.tif"
    a = np.zeros((2, 4, 4), dtype=np.uint8)
    b = np.ones((2, 4, 4), dtype=np.uint8)
    with tifffile.TiffWriter(path) as tw:
        tw.write(a, metadata={"axes": "CYX"})
        tw.write(b, metadata={"axes": "CYX"})

    info = loader.read_header(path)
    assert info.n_images == 2


def test_read_header_multi_series_varying_size_uses_largest(tmp_path: Path, loader):
    path = tmp_path / "multi_varying.tif"
    small = np.zeros((2, 4, 4), dtype=np.uint8)
    big = np.zeros((2, 16, 16), dtype=np.uint8)
    with tifffile.TiffWriter(path) as tw:
        tw.write(small, metadata={"axes": "CYX"})
        tw.write(big, metadata={"axes": "CYX"})

    info = loader.read_header(path)
    assert info.shape == (2, 16, 16)
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


def test_load_range_one_bad_series_does_not_lose_its_siblings(tmp_path: Path, loader, monkeypatch):
    path = tmp_path / "multi3.tif"
    a = np.zeros((2, 4, 4), dtype=np.uint8)
    b = np.ones((2, 4, 4), dtype=np.uint8)
    c = np.full((2, 4, 4), 2, dtype=np.uint8)
    with tifffile.TiffWriter(path) as tw:
        tw.write(a, metadata={"axes": "CYX"})
        tw.write(b, metadata={"axes": "CYX"})
        tw.write(c, metadata={"axes": "CYX"})

    real_build_record = TifffileLoader._build_record

    def flaky_build_record(tf, series_index):
        if series_index == 1:
            raise RuntimeError("simulated decode failure")
        return real_build_record(tf, series_index)

    monkeypatch.setattr(TifffileLoader, "_build_record", staticmethod(flaky_build_record))

    results = dict(loader.load_range(path, 0, 3))
    assert set(results.keys()) == {"0", "1", "2"}
    assert results["1"] is None
    np.testing.assert_array_equal(results["0"].data.compute(), a)
    np.testing.assert_array_equal(results["2"].data.compute(), c)


def test_dask_chunks_reasonable(tmp_path: Path, loader):
    path = tmp_path / "chunked.tif"
    im = np.zeros((2, 64, 64), dtype=np.uint16)
    tifffile.imwrite(path, im, tile=(64, 64), imagej=True, metadata={"axes": "CYX"})
    rec = loader.load(path)
    ch = rec.data.chunks
    assert len(ch) == 3
    assert all(c is not None for c in ch)


def test_load_invalid_file_raises(tmp_path: Path, loader):
    path = tmp_path / "garbage.tif"
    path.write_bytes(b"not a tiff file")
    with pytest.raises(Exception):
        loader.load(path)


def test_load_2d_no_axes_metadata(tmp_path: Path, loader):
    path = tmp_path / "plain.tif"
    im = np.zeros((32, 32), dtype=np.uint8)
    import tifffile as tf
    tf.imwrite(path, im)
    rec = loader.load(path)
    assert rec.data.ndim == 2
    assert "Y" in rec.dim_order or len(rec.dim_order) == 2


def test_load_pyramidal_ome_tiff_is_lazy_and_chunked(tmp_path: Path, loader, caplog):
    """Regression test: da.from_zarr fails for zarr arrays extracted from a Group
    (multiscale OME-TIFF store), silently falling back to series.asarray() which
    loads the entire array into memory.  da.from_array must be used instead."""
    path = tmp_path / "pyramid.ome.tif"
    rng = np.random.default_rng(42)
    n_channels, tile = 4, 16
    im = rng.integers(0, 255, (n_channels, 64, 64), dtype=np.uint16)

    with tifffile.TiffWriter(path, bigtiff=True) as tif:
        opts = dict(photometric="minisblack", metadata={"axes": "CYX"})
        tif.write(im, subifds=1, tile=(tile, tile), **opts)
        tif.write(im[:, ::2, ::2], subfiletype=1, tile=(tile, tile), **opts)

    # Confirm the TIFF has multiple resolution levels (triggers the multiscale
    # zarr store, which is the path that previously caused the failure).
    with tifffile.TiffFile(path) as tf:
        assert len(tf.series[0].levels) > 1, "fixture must produce a multiscale series"

    with caplog.at_level(logging.WARNING):
        rec = loader.load(path)

    # The zarr/aszarr path must have succeeded directly; the exception fallback
    # (series.asarray() + da.from_array) logs this warning when it fires.
    assert "aszarr/Zarr failed" not in caplog.text, "must not silently fall back to eager load"

    # Result must be a lazy dask array, not an in-memory numpy array.
    import dask.array as da
    assert isinstance(rec.data, da.Array), "load() must return a lazy dask array"

    # Data must round-trip correctly.
    np.testing.assert_array_equal(rec.data.compute(), im)
