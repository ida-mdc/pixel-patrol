"""Tests for LmdbLoader - read_header, load, load_range, and metadata extraction."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pixel_patrol_base.core.contracts import FileInfo
from pixel_patrol_aqqua.plugins.loaders.lmdb_loader import LmdbLoader


@pytest.fixture
def loader():
    return LmdbLoader()


# ---------------------------------------------------------------------------
# is_folder_supported
# ---------------------------------------------------------------------------


def test_is_folder_supported_valid(rgb_lmdb: Path, loader) -> None:
    assert loader.is_folder_supported(rgb_lmdb)


def test_is_folder_supported_rejects_plain_dir(tmp_path: Path, loader) -> None:
    assert not loader.is_folder_supported(tmp_path)


def test_is_folder_supported_rejects_file(tmp_path: Path, loader) -> None:
    f = tmp_path / "file.txt"
    f.touch()
    assert not loader.is_folder_supported(f)


# ---------------------------------------------------------------------------
# read_header
# ---------------------------------------------------------------------------


def test_read_header_returns_file_info(rgb_lmdb: Path, loader) -> None:
    info = loader.read_header(rgb_lmdb)
    assert isinstance(info, FileInfo)


def test_read_header_n_images(rgb_lmdb: Path, loader) -> None:
    info = loader.read_header(rgb_lmdb)
    assert info.n_images == 2


def test_read_header_shape_rgb(rgb_lmdb: Path, loader) -> None:
    info = loader.read_header(rgb_lmdb)
    assert info.shape == (48, 84, 3)


def test_read_header_dtype(rgb_lmdb: Path, loader) -> None:
    info = loader.read_header(rgb_lmdb)
    assert info.dtype == np.dtype("uint8")


def test_read_header_dim_order_rgb(rgb_lmdb: Path, loader) -> None:
    info = loader.read_header(rgb_lmdb)
    assert info.dim_order == "YXS"


def test_read_header_grayscale_dim_order(grayscale_lmdb: Path, loader) -> None:
    info = loader.read_header(grayscale_lmdb)
    assert info.dim_order == "YX"


def test_read_header_shape_is_largest_sampled(varying_size_lmdb: Path, loader) -> None:
    info = loader.read_header(varying_size_lmdb)
    assert info.shape == (64, 64)


# ---------------------------------------------------------------------------
# load (first image only)
# ---------------------------------------------------------------------------


def test_load_returns_record(rgb_lmdb: Path, loader) -> None:
    from pixel_patrol_base.core.record import Record
    record = loader.load(rgb_lmdb)
    assert isinstance(record, Record)


def test_load_data_is_numpy(rgb_lmdb: Path, loader) -> None:
    record = loader.load(rgb_lmdb)
    data = np.asarray(record.data)
    assert isinstance(data, np.ndarray)


def test_load_shape(rgb_lmdb: Path, loader) -> None:
    record = loader.load(rgb_lmdb)
    assert np.asarray(record.data).shape == (48, 84, 3)


def test_load_dim_order(rgb_lmdb: Path, loader) -> None:
    record = loader.load(rgb_lmdb)
    assert record.dim_order == "YXS"


def test_load_meta_contains_uuid(rgb_lmdb: Path, loader) -> None:
    record = loader.load(rgb_lmdb)
    assert "image-uuid" in record.meta
    assert record.meta["image-uuid"] == "uuid-0"


def test_load_meta_contains_dim_sizes(rgb_lmdb: Path, loader) -> None:
    record = loader.load(rgb_lmdb)
    assert record.meta.get("size_Y") == 48
    assert record.meta.get("size_X") == 84


# ---------------------------------------------------------------------------
# load_range
# ---------------------------------------------------------------------------


def test_load_range_yields_two_records(rgb_lmdb: Path, loader) -> None:
    results = list(loader.load_range(rgb_lmdb, start=0, stop=2))
    assert len(results) == 2


def test_load_range_child_ids_are_strings(rgb_lmdb: Path, loader) -> None:
    for child_id, _ in loader.load_range(rgb_lmdb, start=0, stop=2):
        assert isinstance(child_id, str)


def test_load_range_child_id_unique(rgb_lmdb: Path, loader) -> None:
    ids = [child_id for child_id, _ in loader.load_range(rgb_lmdb, start=0, stop=2)]
    assert len(set(ids)) == len(ids)


def test_load_range_partial(rgb_lmdb: Path, loader) -> None:
    results = list(loader.load_range(rgb_lmdb, start=1, stop=2))
    assert len(results) == 1


def test_load_range_mid_range_start_returns_correct_records(five_image_lmdb: Path, loader) -> None:
    uuids = [rec.meta.get("image-uuid") for _, rec in loader.load_range(five_image_lmdb, start=3, stop=5)]
    assert uuids == ["uuid-3", "uuid-4"]


def test_load_range_meta_uuid_present(rgb_lmdb: Path, loader) -> None:
    uuids = [rec.meta.get("image-uuid") for _, rec in loader.load_range(rgb_lmdb, start=0, stop=2)]
    assert uuids == ["uuid-0", "uuid-1"]


def test_load_range_all_records_have_correct_shape(rgb_lmdb: Path, loader) -> None:
    for _, record in loader.load_range(rgb_lmdb, start=0, stop=2):
        assert np.asarray(record.data).shape == (48, 84, 3)


# ---------------------------------------------------------------------------
# meta parquet sidecar
# ---------------------------------------------------------------------------


def test_load_range_merges_meta_parquet(rgb_lmdb_with_meta_parquet: Path, loader) -> None:
    results = dict(loader.load_range(rgb_lmdb_with_meta_parquet, start=0, stop=2))
    metas = {rec.meta.get("image-uuid"): rec.meta for rec in results.values()}
    assert metas["uuid-0"]["image-region"] == "NECS"
    assert metas["uuid-1"]["image-region"] == "NECS"


def test_load_range_no_meta_parquet_found(rgb_lmdb: Path, loader) -> None:
    # rgb_lmdb has no numeric-ID suffix, so no sidecar can be matched - metadata
    # should still come through from the blosc2 array itself.
    results = list(loader.load_range(rgb_lmdb, start=0, stop=2))
    assert [rec.meta.get("image-uuid") for _, rec in results] == ["uuid-0", "uuid-1"]


# ---------------------------------------------------------------------------
# Loader class attributes (conformance)
# ---------------------------------------------------------------------------


def test_loader_name() -> None:
    assert LmdbLoader.NAME == "aqqua_lmdb"


def test_supported_extensions_contains_lmdb() -> None:
    assert "lmdb" in LmdbLoader.SUPPORTED_EXTENSIONS


def test_folder_extensions_subset_of_supported() -> None:
    assert LmdbLoader.FOLDER_EXTENSIONS <= LmdbLoader.SUPPORTED_EXTENSIONS


def test_container_extensions_subset_of_supported() -> None:
    assert LmdbLoader.CONTAINER_EXTENSIONS <= LmdbLoader.SUPPORTED_EXTENSIONS
