"""Tests for :class:`TifffileLoader`."""

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


def test_load_pyramidal_ome_tiff_is_lazy_and_chunked(tmp_path: Path, loader):
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

    rec = loader.load(path)

    # Result must be a lazy dask array, not an in-memory numpy array.
    import dask.array as da
    assert isinstance(rec.data, da.Array), "load() must return a lazy dask array"

    # Chunks should reflect tiling: one chunk per channel, one tile per spatial chunk.
    # A single-chunk array (shape == chunk shape) means the fallback fired and the
    # entire image was loaded into memory.
    assert rec.data.chunks[0] == (1,) * n_channels, "channel dim must be chunked per page"
    assert rec.data.chunks[1][0] == tile, "Y dim must be chunked at tile size"
    assert rec.data.chunks[2][0] == tile, "X dim must be chunked at tile size"

    # Data must round-trip correctly.
    np.testing.assert_array_equal(rec.data.compute(), im)
