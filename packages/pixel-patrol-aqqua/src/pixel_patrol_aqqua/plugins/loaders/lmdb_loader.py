"""Pixel-Patrol loader for AqQua LMDB files.
Reads images stored in LMDB databases using the AqQua Dataset format
(blosc2-compressed NumPy arrays with metadata).
"""

import logging
import re
import tomllib
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterator, List, Set, Tuple

import blosc2
import lmdb
import numpy as np
import polars as pl

from pixel_patrol_base.core.contracts import FileInfo
from pixel_patrol_base.core.loader_schema import (
    RASTER_IMAGE_LOADER_SCHEMA,
    RASTER_IMAGE_LOADER_SCHEMA_PATTERNS,
)
from pixel_patrol_base.core.record import record_from, Record

logger = logging.getLogger(__name__)

SKIP_KEYS = {"b2nd", "b2frame"}
_COLOR_CHANNEL_NAMES = {"red", "green", "blue"}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _open_lmdb_readonly(lmdb_path: Path):
    """Open an LMDB environment and its ``image_data`` sub-database.

    Returns:
        A tuple of ``(env, db, txn)``.
    """
    env = lmdb.open(
        str(lmdb_path),
        readonly=True,
        readahead=False,
        max_dbs=2,
        lock=False,
    )
    db = env.open_db(key=b"image_data", integerkey=True, create=False)
    txn = env.begin(db=db, write=False)
    return env, db, txn


def _uncompress_blosc2(raw_bytes: bytes) -> blosc2.NDArray:
    """Decompress raw LMDB value bytes into a blosc2 NDArray."""
    return blosc2.ndarray_from_cframe(raw_bytes, True)


@lru_cache(maxsize=16)
def _load_image_set_toml(lmdb_path: Path) -> Tuple[Dict[str, Any], List[Dict], Dict[str, Any]]:
    """Load the image-set.toml sidecar from the LMDB's parent directory.

    Returns:
        (dataset_meta, channels, defaults)
        - dataset_meta: image-set-* fields from [meta]
        - channels: list of {name, kind} dicts from [[meta.image-set-channel]]
        - defaults: dataset-level fallback values from [image-defaults]
    """
    toml_path = lmdb_path.parent / "image-set.toml"
    if not toml_path.exists():
        logger.debug("LmdbLoader: no image-set.toml found next to '%s'", lmdb_path.name)
        return {}, [], {}

    with open(toml_path, "rb") as f:
        data = tomllib.load(f)

    meta_section = data.get("meta", {})

    dataset_meta = {
        k: v for k, v in meta_section.items()
        if k.startswith("image-set-") and isinstance(v, (str, int, float, bool))
    }

    channels = meta_section.get("image-set-channel", [])
    channels = list(channels) if isinstance(channels, list) else []

    defaults = {
        k: v for k, v in data.get("image-defaults", {}).items()
        if v is not None
    }

    logger.debug(
        "LmdbLoader: loaded toml for '%s': %d dataset fields, %d channels, %d defaults",
        lmdb_path.name, len(dataset_meta), len(channels), len(defaults),
    )
    return dataset_meta, channels, defaults


@lru_cache(maxsize=16)
def _load_meta_parquet(lmdb_path: Path) -> Dict[str, Dict[str, Any]]:
    """Load the matching meta parquet sidecar for an LMDB file.

    Expects a sibling file with the same numeric ID suffix:
        *-images-{ID}.lmdb  →  *-meta-{ID}.parquet

    Returns:
        Dict mapping image-uuid -> flat metadata dict (excluding image-uuid itself).
    """
    match = re.search(r"-(\d+)\.lmdb$", lmdb_path.name)
    if not match:
        logger.warning("LmdbLoader: could not extract numeric ID from '%s'", lmdb_path.name)
        return {}

    numeric_id = match.group(1)
    parquet_path = next(lmdb_path.parent.glob(f"*-meta-{numeric_id}.parquet"), None)
    if parquet_path is None:
        logger.warning("LmdbLoader: no meta parquet found for '%s'", lmdb_path.name)
        return {}

    logger.debug("LmdbLoader: loading meta parquet '%s'", parquet_path.name)
    df = pl.read_parquet(parquet_path)

    if "image-uuid" not in df.columns:
        logger.warning("LmdbLoader: meta parquet '%s' has no 'image-uuid' column", parquet_path.name)
        return {}

    result: Dict[str, Dict[str, Any]] = {}
    for row in df.iter_rows(named=True):
        uuid = row.get("image-uuid")
        if uuid is None:
            continue
        result[str(uuid)] = {
            k: v for k, v in row.items()
            if k != "image-uuid" and v is not None
        }

    logger.debug("LmdbLoader: loaded %d meta rows from '%s'", len(result), parquet_path.name)
    return result


def _extract_blosc2_user_meta(array: blosc2.NDArray) -> Dict[str, Any]:
    """Extract the blosc2 user-metadata fields (no numpy conversion needed)."""
    metadata: Dict[str, Any] = {}
    if hasattr(array, "meta") and array.meta is not None:
        for key, value in dict(array.meta).items():
            if key in SKIP_KEYS or value is None:
                continue
            if isinstance(value, np.generic):
                value = value.item()
            elif not isinstance(value, (str, int, float, bool, list, dict)):
                value = str(value)
            metadata[str(key)] = value
    return metadata


def _get_channels(blosc2_user_meta: Dict[str, Any], toml_channels: List[Dict]) -> List[Dict]:
    """Return the channel list, preferring per-image blosc2 meta over the toml."""
    channels = blosc2_user_meta.get("image-channels")
    if isinstance(channels, list) and channels and isinstance(channels[0], dict):
        return channels
    return toml_channels


def _resolve_dim_order(shape: tuple, channels: List[Dict]) -> str:
    """Determine dim_order from the stored shape and channel list.

    AqQua arrays are stored in CYX (or SYX) order: channels first.
    With no channel info a heuristic is applied.
    """
    ndim = len(shape)
    if channels:
        is_color = all(ch.get("name", "").lower() in _COLOR_CHANNEL_NAMES for ch in channels)
        return "SYX" if is_color else "CYX"
    # No channel info: heuristic
    if ndim == 2:
        return "YX"
    if ndim == 3:
        return "SYX" if shape[0] == 3 else "CYX"
    return "".join(f"D{i}" for i in range(ndim))


def _extract_array_meta(np_arr: np.ndarray, dim_order: str) -> Dict[str, Any]:
    """Extract shape/dtype/dim_order metadata from a numpy array."""
    metadata: Dict[str, Any] = {}
    metadata["shape"] = list(np_arr.shape)
    metadata["ndim"] = int(np_arr.ndim)
    metadata["num_pixels"] = int(np_arr.size)
    metadata["dtype"] = np_arr.dtype.name
    metadata["dim_order"] = dim_order

    if dim_order == "YX":
        metadata["size_Y"] = int(np_arr.shape[0])
        metadata["size_X"] = int(np_arr.shape[1])
    elif dim_order == "SYX":
        metadata["size_S"] = int(np_arr.shape[0])
        metadata["size_Y"] = int(np_arr.shape[1])
        metadata["size_X"] = int(np_arr.shape[2])
    elif dim_order == "CYX":
        metadata["size_C"] = int(np_arr.shape[0])
        metadata["size_Y"] = int(np_arr.shape[1])
        metadata["size_X"] = int(np_arr.shape[2])
    elif dim_order == "YXS":
        metadata["size_Y"] = int(np_arr.shape[0])
        metadata["size_X"] = int(np_arr.shape[1])
        metadata["size_S"] = int(np_arr.shape[2])

    return metadata


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


class LmdbLoader:

    NAME = "aqqua_lmdb"
    DESCRIPTION = "Loads images stored as records in an LMDB key-value database (Aqqua format), one sub-image per key."

    SUPPORTED_EXTENSIONS: Set[str] = {"lmdb", "mdb"}

    OUTPUT_SCHEMA: Dict[str, Any] = dict(RASTER_IMAGE_LOADER_SCHEMA)
    OUTPUT_SCHEMA_PATTERNS: List[tuple[str, Any]] = list(RASTER_IMAGE_LOADER_SCHEMA_PATTERNS)

    FOLDER_EXTENSIONS:    Set[str] = {"lmdb"}
    CONTAINER_EXTENSIONS: Set[str] = {"lmdb", "mdb"}

    def is_folder_supported(self, path: Path) -> bool:
        """LMDB databases are directories, not single files."""
        return path.is_dir() and (path / "data.mdb").exists()

    def read_header(self, lmdb_path: Path) -> FileInfo:
        """Open the LMDB, count entries, and return the largest of the first few entries' shape/dtype."""
        _, toml_channels, _ = _load_image_set_toml(lmdb_path)
        env, db, txn = _open_lmdb_readonly(lmdb_path)
        try:
            n_images = txn.stat(db)["entries"]
            if n_images == 0:
                raise RuntimeError(f"LmdbLoader: empty database at '{lmdb_path}'")
            best_nbytes = -1
            shape = dtype = dim_order = None
            with txn.cursor() as cursor:
                cursor.first()
                for _ in range(min(3, n_images)):
                    # Read shape/dtype/channels from the blosc2 frame header - no numpy decompression needed.
                    array = _uncompress_blosc2(cursor.value())
                    b2_meta = _extract_blosc2_user_meta(array)
                    channels = _get_channels(b2_meta, toml_channels)
                    candidate_shape = tuple(int(x) for x in array.shape)
                    candidate_dtype = np.dtype(array.dtype)
                    candidate_dim_order = _resolve_dim_order(candidate_shape, channels)
                    nbytes = int(np.prod(candidate_shape)) * candidate_dtype.itemsize
                    if nbytes > best_nbytes:
                        best_nbytes = nbytes
                        shape, dtype, dim_order = candidate_shape, candidate_dtype, candidate_dim_order
                    if not cursor.next():
                        break
            return FileInfo(shape=shape, dtype=dtype, dim_order=dim_order, n_images=n_images)
        finally:
            env.close()

    def load(self, lmdb_path: Path) -> Record:
        """Load the first image from the LMDB as a Record."""
        dataset_meta, toml_channels, toml_defaults = _load_image_set_toml(lmdb_path)
        parquet_meta = _load_meta_parquet(lmdb_path)
        env, db, txn = _open_lmdb_readonly(lmdb_path)
        try:
            with txn.cursor() as cursor:
                cursor.first()
                array = _uncompress_blosc2(cursor.value())
        finally:
            env.close()
        b2_meta = _extract_blosc2_user_meta(array)
        channels = _get_channels(b2_meta, toml_channels)
        dim_order = _resolve_dim_order(tuple(int(x) for x in array.shape), channels)
        np_array = np.asarray(array)
        arr_meta = _extract_array_meta(np_array, dim_order)
        uuid = b2_meta.get("image-uuid")
        pq_meta = parquet_meta.get(str(uuid), {}) if uuid else {}
        meta = {**toml_defaults, **dataset_meta, **pq_meta, **b2_meta, **arr_meta}
        return record_from(np_array, meta, kind="intensity")

    def load_range(self, lmdb_path: Path, start: int, stop: int) -> Iterator[Tuple[str, Record]]:
        """Yield (child_id, Record) for images [start, stop) from the LMDB."""
        dataset_meta, toml_channels, toml_defaults = _load_image_set_toml(lmdb_path)
        parquet_meta = _load_meta_parquet(lmdb_path)
        env, db, txn = _open_lmdb_readonly(lmdb_path)
        try:
            with txn.cursor() as cursor:
                if not cursor.first():
                    return
                for idx in range(stop):
                    if idx < start:
                        if not cursor.next():
                            return
                        continue
                    raw_key = cursor.key()
                    int_key = int.from_bytes(raw_key, byteorder="little")
                    child_id = str(int_key)
                    try:
                        array = _uncompress_blosc2(cursor.value())
                        b2_meta = _extract_blosc2_user_meta(array)
                        channels = _get_channels(b2_meta, toml_channels)
                        dim_order = _resolve_dim_order(tuple(int(x) for x in array.shape), channels)
                        # Convert to numpy once; reuse for both metadata and Record payload.
                        np_array = np.asarray(array)
                        arr_meta = _extract_array_meta(np_array, dim_order)
                        uuid = b2_meta.get("image-uuid")
                        pq_meta = parquet_meta.get(str(uuid), {}) if uuid else {}
                        meta = {**toml_defaults, **dataset_meta, **pq_meta, **b2_meta, **arr_meta}
                        yield child_id, record_from(np_array, meta, kind="intensity")
                    except Exception as e:
                        logger.exception(
                            "LmdbLoader: failed to read image key %s in '%s': %s",
                            child_id, lmdb_path.name, e,
                        )
                    if not cursor.next():
                        return
        finally:
            env.close()
