"""Pixel-Patrol loader for AqQua LMDB files.
Reads images stored in LMDB databases using the AqQua Dataset format
(blosc2-compressed NumPy arrays with metadata).
"""

import logging
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


def _extract_metadata(array: blosc2.NDArray) -> Dict[str, Any]:
    """Build a flat metadata dict from a blosc2 NDArray.

    The blosc2 array carries a ``.meta`` mapping with dataset-specific
    fields. We pull everything from ``.meta`` and add shape/dtype info
    derived from the array itself.
    """
    metadata: Dict[str, Any] = {}

    # ---- blosc2 user metadata ----
    if hasattr(array, "meta") and array.meta is not None:
        for key, value in dict(array.meta).items():
            if key in SKIP_KEYS:
                continue

            metadata[str(key)] = value if isinstance(value, (str, int, float, bool, list)) else str(value)


    # ---- array-derived fields ----
    np_arr = np.asarray(array)
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
        # Assume (H, W, C) — the typical storage order for colour images
        h, w, c = np_arr.shape
        metadata["dim_order"] = "YXC"
        metadata["Y_size"] = int(h)
        metadata["X_size"] = int(w)
        metadata["C_size"] = int(c)
    else:
        metadata["dim_order"] = "".join(
            f"D{i}" for i in range(np_arr.ndim)
        )

    return metadata


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
        meta = _extract_metadata(array)
        return record_from(np.asarray(array), meta, kind="intensity")

    @staticmethod
    def load_range(lmdb_path: Path, start: int, stop: int) -> Iterator[Tuple[str, Record]]:
        """Yield (child_id, Record) for images [start, stop) from the LMDB."""
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
                        meta = _extract_metadata(array)
                        yield child_id, record_from(np.asarray(array), meta, kind="intensity")
                    except Exception as e:
                        logger.exception(
                            "LmdbLoader: failed to read image key %s in '%s': %s",
                            child_id, lmdb_path.name, e,
                        )
                    if not cursor.next():
                        return
        finally:
            env.close()
