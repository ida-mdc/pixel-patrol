"""Shared fixtures for pixel-patrol-aqqua tests."""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Dict, Any

import blosc2
import lmdb
import numpy as np
import pytest


_RGB_CHANNELS = [
    {"name": "red",   "kind": "brightfield"},
    {"name": "green", "kind": "brightfield"},
    {"name": "blue",  "kind": "brightfield"},
]
_GRAY_CHANNELS = [{"name": "grayscale", "kind": "brightfield"}]


def _write_lmdb(path: Path, images: List[Tuple[np.ndarray, Dict[str, Any]]]) -> None:
    """Write (array, meta) pairs into an LMDB image_data sub-database.

    Keys are sequential 8-byte little-endian integers starting from 0.
    Each value is a blosc2 cframe with the array and its metadata.
    """
    env = lmdb.open(str(path), map_size=50 * 1024 * 1024, max_dbs=2)
    db = env.open_db(key=b"image_data", integerkey=True, create=True)
    with env.begin(db=db, write=True) as txn:
        for idx, (arr, meta) in enumerate(images):
            b2arr = blosc2.asarray(arr, meta=meta)
            cframe = b2arr.to_cframe()
            key = idx.to_bytes(8, byteorder="little")
            txn.put(key, cframe)
    env.close()


@pytest.fixture()
def rgb_lmdb(tmp_path: Path) -> Path:
    """LMDB with two RGB images in CYX layout (3, H, W), uint16."""
    lmdb_path = tmp_path / "test.lmdb"
    rng = np.random.default_rng(42)
    images = [
        (
            rng.integers(0, 65535, (3, 48, 84), dtype=np.uint16),
            {"image-uuid": "uuid-0", "class": "diatom", "image-channels": _RGB_CHANNELS},
        ),
        (
            rng.integers(0, 65535, (3, 48, 84), dtype=np.uint16),
            {"image-uuid": "uuid-1", "class": "copepod", "image-channels": _RGB_CHANNELS},
        ),
    ]
    _write_lmdb(lmdb_path, images)
    return lmdb_path


@pytest.fixture()
def grayscale_lmdb(tmp_path: Path) -> Path:
    """LMDB with two grayscale images in CYX layout (1, H, W), uint16."""
    lmdb_path = tmp_path / "gray.lmdb"
    rng = np.random.default_rng(7)
    images = [
        (
            rng.integers(0, 65535, (1, 32, 64), dtype=np.uint16),
            {"image-uuid": "gray-0", "image-channels": _GRAY_CHANNELS},
        ),
        (
            rng.integers(0, 65535, (1, 32, 64), dtype=np.uint16),
            {"image-uuid": "gray-1", "image-channels": _GRAY_CHANNELS},
        ),
    ]
    _write_lmdb(lmdb_path, images)
    return lmdb_path


@pytest.fixture()
def varying_size_lmdb(tmp_path: Path) -> Path:
    """LMDB with 2D (YX) images of different sizes - no channel dim."""
    lmdb_path = tmp_path / "varying.lmdb"
    rng = np.random.default_rng(3)
    images = [
        (rng.integers(0, 255, (16, 16), dtype=np.uint8), {"image-uuid": "small-0"}),
        (rng.integers(0, 255, (64, 64), dtype=np.uint8), {"image-uuid": "big-1"}),
    ]
    _write_lmdb(lmdb_path, images)
    return lmdb_path


@pytest.fixture()
def five_image_lmdb(tmp_path: Path) -> Path:
    """LMDB with five grayscale CYX images, uuid-0..uuid-4 in order."""
    lmdb_path = tmp_path / "five.lmdb"
    rng = np.random.default_rng(11)
    images = [
        (
            rng.integers(0, 65535, (1, 8, 8), dtype=np.uint16),
            {"image-uuid": f"uuid-{i}", "image-channels": _GRAY_CHANNELS},
        )
        for i in range(5)
    ]
    _write_lmdb(lmdb_path, images)
    return lmdb_path


@pytest.fixture()
def rgb_lmdb_with_meta_parquet(tmp_path: Path) -> Path:
    """LMDB with numeric ID suffix + matching parquet sidecar."""
    import polars as pl

    lmdb_path = tmp_path / "sample-images-42.lmdb"
    rng = np.random.default_rng(42)
    images = [
        (
            rng.integers(0, 65535, (3, 48, 84), dtype=np.uint16),
            {"image-uuid": "uuid-0", "image-channels": _RGB_CHANNELS},
        ),
        (
            rng.integers(0, 65535, (3, 48, 84), dtype=np.uint16),
            {"image-uuid": "uuid-1", "image-channels": _RGB_CHANNELS},
        ),
    ]
    _write_lmdb(lmdb_path, images)

    parquet_path = tmp_path / "sample-meta-42.parquet"
    pl.DataFrame({
        "image-uuid": ["uuid-0", "uuid-1"],
        "image-region": ["NECS", "NECS"],
        "image-altitude-meters": [0, 0],
    }).write_parquet(parquet_path)

    return lmdb_path


@pytest.fixture()
def lmdb_with_toml(tmp_path: Path) -> Path:
    """LMDB with a sibling image-set.toml providing dataset metadata and defaults."""
    lmdb_path = tmp_path / "dataset.lmdb"
    rng = np.random.default_rng(99)
    images = [
        (
            rng.integers(0, 65535, (1, 32, 64), dtype=np.uint16),
            {"image-uuid": "toml-0", "image-channels": _GRAY_CHANNELS},
        ),
    ]
    _write_lmdb(lmdb_path, images)

    toml_content = """\
[meta]
image-set-name = "Test Dataset"
image-set-sensor = "TestSensor"
image-set-sensor-family = "TestFamily"
image-set-spectral-resolution = "grayscale"
image-set-uuid = "test-uuid-1234"
image-set-region = "North Sea"
image-set-ifdo-version = "v2.1.0"

[[meta.image-set-channel]]
name = "grayscale"
kind = "brightfield"

[image-defaults]
image-pixel-magnitude = 0.5
image-latitude = 54.0
image-longitude = 8.0
image-altitude-meters = -10
"""
    (tmp_path / "image-set.toml").write_text(toml_content)
    return lmdb_path
