"""Tests for reading the storage codec of a file from its header."""

import struct
from pathlib import Path
from types import SimpleNamespace

import h5py
import numcodecs
import numpy as np
import pytest
import tifffile
import zarr
from zarr.codecs import BloscCodec

from pixel_patrol_loader_bio.plugins.loaders import _compression as compression
from pixel_patrol_loader_bio.plugins.loaders._compression import compression_of
from pixel_patrol_loader_bio.plugins.loaders.bioio_loader import BioIoLoader
from pixel_patrol_loader_bio.plugins.loaders.tifffile_loader import TifffileLoader
from pixel_patrol_loader_bio.plugins.loaders.zarr_loader import ZarrLoader


def czi_bytes(codes):
    directory_at = 128
    header = b"ZISRAWFILE".ljust(32, b"\0") + b"\0" * (8 + 8 + 16 + 16 + 4) + struct.pack("<q", directory_at)
    body = b"ZISRAWDIRECTORY".ljust(32, b"\0") + struct.pack("<i", len(codes)) + b"\0" * 124
    for code in codes:
        dims = b"".join(struct.pack("<4siifi", name, 0, 1024, 0, 1024) for name in (b"X", b"Y"))
        body += b"DV" + struct.pack("<iqii", 3, 0, 0, code) + b"\0" * 6 + struct.pack("<i", 2) + dims
    return header.ljust(directory_at, b"\0") + body


def write_zarr(root: Path, zarr_format: int, **array_kwargs):
    group = zarr.open_group(root, mode="w", zarr_format=zarr_format)
    group.create_array("0", shape=(1, 64, 64), chunks=(1, 32, 32), dtype="uint8", **array_kwargs)


@pytest.mark.parametrize("kwargs, expected", [
    ({}, "none"),
    ({"compression": "zlib"}, "deflate"),
    ({"compression": "lzw"}, "lzw"),
    ({"compression": "zlib", "predictor": True}, "deflate + horizontal predictor"),
])
def test_tiff_codec(tmp_path, kwargs, expected):
    path = tmp_path / "image.tif"
    tifffile.imwrite(path, np.zeros((64, 96), dtype=np.uint8), **kwargs)
    assert compression_of(path) == expected


def test_czi_reports_codecs_by_frequency(tmp_path):
    path = tmp_path / "scan.czi"
    path.write_bytes(czi_bytes([4, 4, 4, 0]))
    assert compression_of(path) == "JPEG XR + none"


def test_zarr_v3_codec_chain(tmp_path):
    root = tmp_path / "scan.ome.zarr"
    write_zarr(root, 3, compressors=[BloscCodec(cname="zstd", clevel=3, shuffle="bitshuffle")])
    assert compression_of(root) == "blosc(zstd, bitshuffle) level 3"


def test_zarr_v2_compressor(tmp_path):
    root = tmp_path / "old.zarr"
    write_zarr(root, 2, compressors=numcodecs.Zstd())
    assert compression_of(root) == "zstd"


def test_zarr_without_compressor(tmp_path):
    root = tmp_path / "raw.zarr"
    write_zarr(root, 2, compressors=None)
    assert compression_of(root) == "none"


@pytest.mark.parametrize("name, expected", [("a.png", "deflate"), ("a.JPG", "JPEG"), ("a.jpeg", "JPEG"), ("a.bmp", "none"), ("a.lif", "none")])
def test_formats_with_one_codec(tmp_path, name, expected):
    assert compression_of(tmp_path / name) == expected


@pytest.mark.parametrize("attribute, expected", [("lossless", "lossless"), ("lossy", "lossy"), (None, "none")])
def test_nd2_compression_type(monkeypatch, tmp_path, attribute, expected):
    class FakeND2File:
        attributes = SimpleNamespace(compressionType=attribute)
        def __init__(self, path): pass
        def __enter__(self): return self
        def __exit__(self, *exc): return False
    monkeypatch.setattr(compression, "nd2", SimpleNamespace(ND2File=FakeND2File))
    assert compression_of(tmp_path / "a.nd2") == expected


@pytest.mark.parametrize("kwargs, expected", [
    ({}, "none"),
    ({"compression": "gzip", "compression_opts": 4}, "gzip level 4"),
    ({"compression": "lzf", "shuffle": True}, "lzf + shuffle"),
])
def test_hdf5_dataset_compression(tmp_path, kwargs, expected):
    with h5py.File(tmp_path / "a.h5", "w") as f:
        dataset = f.create_dataset("img", data=np.zeros((8, 8), dtype=np.uint8), chunks=(4, 4) if kwargs else None, **kwargs)
        assert compression.hdf5_compression(dataset) == expected


def test_unknown_or_broken_files_have_no_codec(tmp_path):
    (tmp_path / "notes.txt").write_text("hello")
    (tmp_path / "broken.czi").write_bytes(b"not a czi")
    assert compression_of(tmp_path / "notes.txt") is None
    assert compression_of(tmp_path / "broken.czi") is None
    assert compression_of(tmp_path / "missing.tif") is None


def test_loaders_put_the_codec_into_the_record_after_dtype(tmp_path):
    path = tmp_path / "image.tif"
    tifffile.imwrite(path, np.zeros((16, 16), dtype=np.uint8), compression="zlib")
    for loader in (TifffileLoader(), BioIoLoader()):
        meta = dict(loader.load(path).meta)
        assert meta["compression"] == "deflate"
        keys = list(meta)
        assert keys.index("compression") == keys.index("dtype") + 1
