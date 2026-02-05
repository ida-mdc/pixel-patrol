"""Pixel-Patrol loader for AqQua LMDB files.
Reads images stored in LMDB databases using the AqQua Dataset format
(blosc2-compressed NumPy arrays with metadata).
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import blosc2
import lmdb
import numpy as np
import polars as pl

from pixel_patrol_base.core.record import record_from

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
    def load(source: str) -> Optional[Dict[str, Any]]:
        """Load all images from a single LMDB file.
        """
        lmdb_path = Path(source)
        env, db, txn = _open_lmdb_readonly(lmdb_path)
        try:
            n_entries = txn.stat(db)["entries"]
            logger.info(
                "LmdbLoader: opening '%s' — %d image(s)", lmdb_path.name, n_entries
            )

            if n_entries == 0:
                return None

            records: Dict[str, Any] = {}
            with txn.cursor() as cursor:
                cursor.first()
                while True:
                    raw_key = cursor.key()
                    int_key = int.from_bytes(raw_key, byteorder="little")
                    child_id = str(int_key)

                    try:
                        array = _uncompress_blosc2(cursor.value())
                        meta = _extract_metadata(array)
                        data = np.asarray(array)

                        records[child_id] = record_from(
                            data, meta, kind="intensity"
                        )
                    except Exception as e:
                        logger.exception(
                            "LmdbLoader: failed to read image key %s in '%s': %s",
                            child_id,
                            lmdb_path.name,
                            e,
                        )

                    if not cursor.next():
                        break

            logger.info(
                "LmdbLoader: loaded %d record(s) from '%s'",
                len(records),
                lmdb_path.name,
            )
            return records if records else None
        finally:
            env.close()
