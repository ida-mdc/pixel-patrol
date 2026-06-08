"""Tests for LmdbLoader - read_header, load, load_range, and metadata extraction."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pixel_patrol_base.core.contracts import FileInfo
from pixel_patrol_aqqua.plugins.loaders.lmdb_loader import LmdbLoader


# ---------------------------------------------------------------------------
# is_folder_supported
# ---------------------------------------------------------------------------


def test_is_folder_supported_valid(rgb_lmdb: Path) -> None:
    assert LmdbLoader.is_folder_supported(rgb_lmdb)


def test_is_folder_supported_rejects_plain_dir(tmp_path: Path) -> None:
    assert not LmdbLoader.is_folder_supported(tmp_path)


def test_is_folder_supported_rejects_file(tmp_path: Path) -> None:
    f = tmp_path / "file.txt"
    f.touch()
    assert not LmdbLoader.is_folder_supported(f)


# ---------------------------------------------------------------------------
# read_header
# ---------------------------------------------------------------------------


def test_read_header_returns_file_info(rgb_lmdb: Path) -> None:
    info = LmdbLoader.read_header(rgb_lmdb)
    assert isinstance(info, FileInfo)


def test_read_header_n_images(rgb_lmdb: Path) -> None:
    info = LmdbLoader.read_header(rgb_lmdb)
    assert info.n_images == 2


def test_read_header_shape_rgb(rgb_lmdb: Path) -> None:
    info = LmdbLoader.read_header(rgb_lmdb)
    assert info.shape == (48, 84, 3)


def test_read_header_dtype(rgb_lmdb: Path) -> None:
    info = LmdbLoader.read_header(rgb_lmdb)
    assert info.dtype == np.dtype("uint8")


def test_read_header_dim_order_rgb(rgb_lmdb: Path) -> None:
    info = LmdbLoader.read_header(rgb_lmdb)
    assert "".join(info.dim_order) == "YXS"


def test_read_header_grayscale_dim_order(grayscale_lmdb: Path) -> None:
    info = LmdbLoader.read_header(grayscale_lmdb)
    assert "".join(info.dim_order) == "YX"


# ---------------------------------------------------------------------------
# load (first image only)
# ---------------------------------------------------------------------------


def test_load_returns_record(rgb_lmdb: Path) -> None:
    from pixel_patrol_base.core.record import Record
    record = LmdbLoader.load(rgb_lmdb)
    assert isinstance(record, Record)


def test_load_data_is_numpy(rgb_lmdb: Path) -> None:
    record = LmdbLoader.load(rgb_lmdb)
    data = np.asarray(record.data)
    assert isinstance(data, np.ndarray)


def test_load_shape(rgb_lmdb: Path) -> None:
    record = LmdbLoader.load(rgb_lmdb)
    assert np.asarray(record.data).shape == (48, 84, 3)


def test_load_dim_order(rgb_lmdb: Path) -> None:
    record = LmdbLoader.load(rgb_lmdb)
    assert record.dim_order == "YXS"


def test_load_meta_contains_uuid(rgb_lmdb: Path) -> None:
    record = LmdbLoader.load(rgb_lmdb)
    assert "image-uuid" in record.meta
    assert record.meta["image-uuid"] == "uuid-0"


def test_load_meta_contains_dim_sizes(rgb_lmdb: Path) -> None:
    record = LmdbLoader.load(rgb_lmdb)
    assert record.meta.get("Y_size") == 48
    assert record.meta.get("X_size") == 84


# ---------------------------------------------------------------------------
# load_range
# ---------------------------------------------------------------------------


def test_load_range_yields_two_records(rgb_lmdb: Path) -> None:
    results = list(LmdbLoader.load_range(rgb_lmdb, start=0, stop=2))
    assert len(results) == 2


def test_load_range_child_ids_are_strings(rgb_lmdb: Path) -> None:
    for child_id, _ in LmdbLoader.load_range(rgb_lmdb, start=0, stop=2):
        assert isinstance(child_id, str)


def test_load_range_child_id_unique(rgb_lmdb: Path) -> None:
    ids = [child_id for child_id, _ in LmdbLoader.load_range(rgb_lmdb, start=0, stop=2)]
    assert len(set(ids)) == len(ids)


def test_load_range_partial(rgb_lmdb: Path) -> None:
    results = list(LmdbLoader.load_range(rgb_lmdb, start=1, stop=2))
    assert len(results) == 1


def test_load_range_meta_uuid_present(rgb_lmdb: Path) -> None:
    uuids = [rec.meta.get("image-uuid") for _, rec in LmdbLoader.load_range(rgb_lmdb, start=0, stop=2)]
    assert uuids == ["uuid-0", "uuid-1"]


def test_load_range_all_records_have_correct_shape(rgb_lmdb: Path) -> None:
    for _, record in LmdbLoader.load_range(rgb_lmdb, start=0, stop=2):
        assert np.asarray(record.data).shape == (48, 84, 3)


# ---------------------------------------------------------------------------
# Loader class attributes (conformance)
# ---------------------------------------------------------------------------


def test_loader_name() -> None:
    assert LmdbLoader.NAME == "aqqua_lmdb"


def test_supported_extensions_contains_lmdb() -> None:
    assert "lmdb" in LmdbLoader.SUPPORTED_EXTENSIONS


def test_folder_extensions_subset_of_supported() -> None:
    assert LmdbLoader.FOLDER_EXTENSIONS <= LmdbLoader.SUPPORTED_EXTENSIONS
