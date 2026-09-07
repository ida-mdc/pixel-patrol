"""Tests for LmdbLoader - read_header, load, load_range, and metadata extraction."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pixel_patrol_base.core.contracts import FileInfo
from pixel_patrol_aqqua.plugins.loaders.lmdb_loader import (
    LmdbLoader,
    _get_channels,
    _resolve_dim_order,
    _load_image_set_toml,
)


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
# _resolve_dim_order
# ---------------------------------------------------------------------------


def test_resolve_dim_order_2d_no_channels() -> None:
    assert _resolve_dim_order((64, 64), []) == "YX"


def test_resolve_dim_order_rgb_channels() -> None:
    channels = [{"name": "red", "kind": "brightfield"}, {"name": "green", "kind": "brightfield"}, {"name": "blue", "kind": "brightfield"}]
    assert _resolve_dim_order((3, 64, 64), channels) == "SYX"


def test_resolve_dim_order_gray_channel() -> None:
    channels = [{"name": "grayscale", "kind": "brightfield"}]
    assert _resolve_dim_order((1, 64, 64), channels) == "CYX"


def test_resolve_dim_order_multichannel_non_color() -> None:
    channels = [{"name": "DAPI", "kind": "fluorescence"}, {"name": "GFP", "kind": "fluorescence"}]
    assert _resolve_dim_order((2, 64, 64), channels) == "CYX"


def test_resolve_dim_order_heuristic_3d_first_dim_3() -> None:
    assert _resolve_dim_order((3, 64, 64), []) == "SYX"


def test_resolve_dim_order_heuristic_3d_first_dim_not_3() -> None:
    assert _resolve_dim_order((1, 64, 64), []) == "CYX"
    assert _resolve_dim_order((4, 64, 64), []) == "CYX"


def test_resolve_dim_order_nd() -> None:
    assert _resolve_dim_order((2, 3, 64, 64), []) == "D0D1D2D3"


# ---------------------------------------------------------------------------
# _get_channels
# ---------------------------------------------------------------------------


def test_get_channels_prefers_blosc2_meta() -> None:
    b2_channels = [{"name": "red", "kind": "brightfield"}]
    toml_channels = [{"name": "grayscale", "kind": "brightfield"}]
    assert _get_channels({"image-channels": b2_channels}, toml_channels) == b2_channels


def test_get_channels_falls_back_to_toml() -> None:
    toml_channels = [{"name": "grayscale", "kind": "brightfield"}]
    assert _get_channels({}, toml_channels) == toml_channels


def test_get_channels_ignores_non_list_blosc2_value() -> None:
    toml_channels = [{"name": "grayscale", "kind": "brightfield"}]
    assert _get_channels({"image-channels": "bad"}, toml_channels) == toml_channels


# ---------------------------------------------------------------------------
# _load_image_set_toml
# ---------------------------------------------------------------------------


def test_load_toml_missing(rgb_lmdb: Path) -> None:
    _load_image_set_toml.cache_clear()
    dataset_meta, channels, defaults = _load_image_set_toml(rgb_lmdb)
    assert dataset_meta == {}
    assert channels == []
    assert defaults == {}


def test_load_toml_dataset_meta(lmdb_with_toml: Path) -> None:
    _load_image_set_toml.cache_clear()
    dataset_meta, _, _ = _load_image_set_toml(lmdb_with_toml)
    assert dataset_meta["image-set-name"] == "Test Dataset"
    assert dataset_meta["image-set-sensor"] == "TestSensor"
    assert dataset_meta["image-set-spectral-resolution"] == "grayscale"


def test_load_toml_channels(lmdb_with_toml: Path) -> None:
    _load_image_set_toml.cache_clear()
    _, channels, _ = _load_image_set_toml(lmdb_with_toml)
    assert len(channels) == 1
    assert channels[0] == {"name": "grayscale", "kind": "brightfield"}


def test_load_toml_defaults(lmdb_with_toml: Path) -> None:
    _load_image_set_toml.cache_clear()
    _, _, defaults = _load_image_set_toml(lmdb_with_toml)
    assert defaults["image-pixel-magnitude"] == 0.5
    assert defaults["image-latitude"] == 54.0
    assert defaults["image-altitude-meters"] == -10


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
    assert info.shape == (3, 48, 84)


def test_read_header_dtype(rgb_lmdb: Path, loader) -> None:
    info = loader.read_header(rgb_lmdb)
    assert info.dtype == np.dtype("uint16")


def test_read_header_dim_order_rgb(rgb_lmdb: Path, loader) -> None:
    info = loader.read_header(rgb_lmdb)
    assert info.dim_order == "SYX"


def test_read_header_grayscale_dim_order(grayscale_lmdb: Path, loader) -> None:
    info = loader.read_header(grayscale_lmdb)
    assert info.dim_order == "CYX"


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
    assert isinstance(np.asarray(record.data), np.ndarray)


def test_load_shape(rgb_lmdb: Path, loader) -> None:
    record = loader.load(rgb_lmdb)
    assert np.asarray(record.data).shape == (3, 48, 84)


def test_load_dim_order(rgb_lmdb: Path, loader) -> None:
    record = loader.load(rgb_lmdb)
    assert record.dim_order == "SYX"


def test_load_meta_contains_uuid(rgb_lmdb: Path, loader) -> None:
    record = loader.load(rgb_lmdb)
    assert record.meta.get("image-uuid") == "uuid-0"


def test_load_meta_contains_dim_sizes(rgb_lmdb: Path, loader) -> None:
    record = loader.load(rgb_lmdb)
    assert record.meta.get("size_S") == 3
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
        assert np.asarray(record.data).shape == (3, 48, 84)


def test_load_range_dim_order_rgb(rgb_lmdb: Path, loader) -> None:
    for _, record in loader.load_range(rgb_lmdb, start=0, stop=2):
        assert record.dim_order == "SYX"


# ---------------------------------------------------------------------------
# meta parquet sidecar
# ---------------------------------------------------------------------------


def test_load_range_merges_meta_parquet(rgb_lmdb_with_meta_parquet: Path, loader) -> None:
    results = dict(loader.load_range(rgb_lmdb_with_meta_parquet, start=0, stop=2))
    metas = {rec.meta.get("image-uuid"): rec.meta for rec in results.values()}
    assert metas["uuid-0"]["image-region"] == "NECS"
    assert metas["uuid-1"]["image-region"] == "NECS"


def test_load_range_no_meta_parquet_found(rgb_lmdb: Path, loader) -> None:
    results = list(loader.load_range(rgb_lmdb, start=0, stop=2))
    assert [rec.meta.get("image-uuid") for _, rec in results] == ["uuid-0", "uuid-1"]


# ---------------------------------------------------------------------------
# toml metadata in records
# ---------------------------------------------------------------------------


def test_load_includes_toml_dataset_meta(lmdb_with_toml: Path, loader) -> None:
    _load_image_set_toml.cache_clear()
    record = loader.load(lmdb_with_toml)
    assert record.meta.get("image-set-name") == "Test Dataset"
    assert record.meta.get("image-set-sensor") == "TestSensor"


def test_load_includes_toml_defaults(lmdb_with_toml: Path, loader) -> None:
    _load_image_set_toml.cache_clear()
    record = loader.load(lmdb_with_toml)
    assert record.meta.get("image-pixel-magnitude") == 0.5
    assert record.meta.get("image-latitude") == 54.0


def test_load_range_includes_toml_metadata(lmdb_with_toml: Path, loader) -> None:
    _load_image_set_toml.cache_clear()
    results = list(loader.load_range(lmdb_with_toml, start=0, stop=1))
    assert len(results) == 1
    _, record = results[0]
    assert record.meta.get("image-set-name") == "Test Dataset"
    assert record.meta.get("image-pixel-magnitude") == 0.5


def test_blosc2_meta_overrides_toml_defaults(tmp_path: Path, loader) -> None:
    """Per-image blosc2 meta takes priority over toml defaults for the same key."""
    import blosc2, lmdb as _lmdb

    # Write an LMDB with image-altitude-meters = -999 in blosc2 per-image meta.
    lmdb_path = tmp_path / "priority.lmdb"
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 65535, (1, 8, 8), dtype=np.uint16)
    b2arr = blosc2.asarray(arr, meta={"image-uuid": "p-0", "image-altitude-meters": -999})
    env = _lmdb.open(str(lmdb_path), map_size=50 * 1024 * 1024, max_dbs=2)
    db = env.open_db(key=b"image_data", integerkey=True, create=True)
    with env.begin(db=db, write=True) as txn:
        txn.put((0).to_bytes(8, byteorder="little"), b2arr.to_cframe())
    env.close()

    # Toml sets image-altitude-meters = -10 as a default.
    (tmp_path / "image-set.toml").write_text(
        '[meta]\nimage-set-name = "P"\n\n[image-defaults]\nimage-altitude-meters = -10\n'
    )
    _load_image_set_toml.cache_clear()
    record = loader.load(lmdb_path)
    # blosc2 per-image value wins over toml default.
    assert record.meta["image-altitude-meters"] == -999


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
