"""Shared fixtures for pixel-patrol-aqqua tests."""
from __future__ import annotations

import struct
from pathlib import Path
from typing import List, Tuple, Dict, Any

import blosc2
import lmdb
import numpy as np
import pytest


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
    """Single LMDB with two small RGB (HxWxC) images."""
    lmdb_path = tmp_path / "test.lmdb"
    rng = np.random.default_rng(42)
    images = [
        (
            rng.integers(0, 255, (48, 84, 3), dtype=np.uint8),
            {"image-uuid": "uuid-0", "class": "diatom"},
        ),
        (
            rng.integers(0, 255, (48, 84, 3), dtype=np.uint8),
            {"image-uuid": "uuid-1", "class": "copepod"},
        ),
    ]
    _write_lmdb(lmdb_path, images)
    return lmdb_path


@pytest.fixture()
def grayscale_lmdb(tmp_path: Path) -> Path:
    """Single LMDB with two grayscale (HxW) images."""
    lmdb_path = tmp_path / "gray.lmdb"
    rng = np.random.default_rng(7)
    images = [
        (rng.integers(0, 255, (32, 64), dtype=np.uint8), {"image-uuid": "gray-0"}),
        (rng.integers(0, 255, (32, 64), dtype=np.uint8), {"image-uuid": "gray-1"}),
    ]
    _write_lmdb(lmdb_path, images)
    return lmdb_path
