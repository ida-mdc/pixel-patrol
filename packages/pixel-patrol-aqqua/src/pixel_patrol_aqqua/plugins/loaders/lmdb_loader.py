"""Pixel-Patrol loader for AqQua LMDB files.
Reads images stored in LMDB databases using the AqQua Dataset format
(blosc2-compressed NumPy arrays with metadata).
"""

import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

import blosc2
import lmdb
import numpy as np
import polars as pl

from pixel_patrol_base.core.contracts import FileInfo
from pixel_patrol_base.core.record import record_from, Record

logger = logging.getLogger(__name__)

SKIP_KEYS = {"b2nd", "b2frame"}

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
def _load_meta_parquet(lmdb_path: Path) -> Dict[str, Dict[str, Any]]:
    """Load the matching meta parquet sidecar for an LMDB file.

    Expects a sibling file with the same numeric ID suffix:
        *-images-{ID}.lmdb  →  *-meta-{ID}.parquet

    Returns:
        Dict mapping image-uuid -> flat metadata dict (excluding image-uuid itself).

    Result is cached per worker process (lru_cache), so each LMDB file's parquet
    is read from disk only once per worker, not once per load_range() call.
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

    logger.info("LmdbLoader: loading meta parquet '%s'", parquet_path.name)
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

    logger.info("LmdbLoader: loaded %d meta rows from '%s'", len(result), parquet_path.name)
    return result


def _extract_blosc2_user_meta(array: blosc2.NDArray) -> Dict[str, Any]:
    """Extract the blosc2 user-metadata fields (no numpy conversion needed)."""
    metadata: Dict[str, Any] = {}
    if hasattr(array, "meta") and array.meta is not None:
        for key, value in dict(array.meta).items():
            if key in SKIP_KEYS:
                continue
            metadata[str(key)] = value if isinstance(value, (str, int, float, bool, list)) else str(value)
    return metadata


def _extract_array_meta(np_arr: np.ndarray) -> Dict[str, Any]:
    """Extract shape/dtype/dim_order metadata from an already-converted numpy array."""
    metadata: Dict[str, Any] = {}
    metadata["shape"] = list(np_arr.shape)
    metadata["ndim"] = int(np_arr.ndim)
    metadata["num_pixels"] = int(np_arr.size)
    metadata["dtype"] = np_arr.dtype.name

    # Interpret common 2-D / 3-D image conventions
    if np_arr.ndim == 2:
        h, w = np_arr.shape
        metadata["dim_order"] = "YX"
        metadata["Y_size"] = int(h)
        metadata["X_size"] = int(w)
    elif np_arr.ndim == 3:
        h, w, c = np_arr.shape
        metadata["dim_order"] = "YXS"
        metadata["Y_size"] = int(h)
        metadata["X_size"] = int(w)
        metadata["C_size"] = int(c)
    else:
        metadata["dim_order"] = "".join(f"D{i}" for i in range(np_arr.ndim))

    return metadata


def _extract_metadata(array: blosc2.NDArray) -> Dict[str, Any]:
    """Build a flat metadata dict from a blosc2 NDArray.

    Combines blosc2 user metadata with shape/dtype info.
    Used by read_header() and load() where a single conversion is fine.
    For load_range(), use _extract_blosc2_user_meta() + _extract_array_meta()
    directly to avoid converting to numpy twice.
    """
    np_arr = np.asarray(array)
    return {**_extract_blosc2_user_meta(array), **_extract_array_meta(np_arr)}


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


class LmdbLoader:

    NAME = "aqqua_lmdb"

    SUPPORTED_EXTENSIONS: Set[str] = {"lmdb", "mdb"}

    OUTPUT_SCHEMA: Dict[str, Any] = {
        "dim_order": str,
        "ndim": int,
        "num_pixels": int,
        "shape": pl.Array,
        "dtype": str,
    }

    OUTPUT_SCHEMA_PATTERNS: List[tuple[str, Any]] = [
        (r"^[A-Za-z]+_size$", int),
    ]

    FOLDER_EXTENSIONS: Set[str] = {"lmdb"}

    @staticmethod
    def is_folder_supported(path: Path) -> bool:
        """LMDB databases are directories, not single files."""
        return path.is_dir() and (path / "data.mdb").exists()

    @staticmethod
    def read_header(lmdb_path: Path) -> FileInfo:
        """Open the LMDB, count entries, and peek the first entry for shape/dtype."""
        env, db, txn = _open_lmdb_readonly(lmdb_path)
        try:
            n_images = txn.stat(db)["entries"]
            if n_images == 0:
                raise RuntimeError(f"LmdbLoader: empty database at '{lmdb_path}'")
            with txn.cursor() as cursor:
                cursor.first()
                array = _uncompress_blosc2(cursor.value())
            meta = _extract_metadata(array)
            shape = tuple(int(x) for x in meta["shape"])
            dim_order = tuple(meta["dim_order"])
            dtype = np.dtype(str(meta["dtype"]))
            return FileInfo(shape=shape, dtype=dtype, dim_order=dim_order, n_images=n_images)
        finally:
            env.close()

    @staticmethod
    def load(lmdb_path: Path) -> Record:
        """Load the first image from the LMDB as a Record."""
        env, db, txn = _open_lmdb_readonly(lmdb_path)
        try:
            with txn.cursor() as cursor:
                cursor.first()
                array = _uncompress_blosc2(cursor.value())
        finally:
            env.close()
        np_array = np.asarray(array)
        meta = {**_extract_blosc2_user_meta(array), **_extract_array_meta(np_array)}
        return record_from(np_array, meta, kind="intensity")

    @staticmethod
    def load_range(lmdb_path: Path, start: int, stop: int) -> Iterator[Tuple[str, Record]]:
        """Yield (child_id, Record) for images [start, stop) from the LMDB."""
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
                        # Convert to numpy once; reuse for both metadata extraction
                        # and the Record payload — avoids decompressing blosc2 twice.
                        np_array = np.asarray(array)
                        meta = {**_extract_blosc2_user_meta(array), **_extract_array_meta(np_array)}
                        uuid = meta.get("image-uuid")
                        if uuid and str(uuid) in parquet_meta:
                            # blosc2 meta takes precedence on key conflicts
                            meta = {**parquet_meta[str(uuid)], **meta}
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
