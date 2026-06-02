"""
NumpyLoader: minimal loader for .npy files used in from-file benchmark runs.

read_header() uses numpy's memory-map mode so the file header is parsed but
no pixel data is read into RAM.  dim_order is inferred from ndim.
"""
from __future__ import annotations

from pathlib import Path
from typing import Set, Tuple

import numpy as np

from pixel_patrol_base.core.contracts import FileInfo
from pixel_patrol_base.core.record import Record, record_from


_DIM_ORDER: dict[int, Tuple[str, ...]] = {
    1: ("X",),
    2: ("Y", "X"),
    3: ("Z", "Y", "X"),
    4: ("Z", "C", "Y", "X"),
}


class NumpyLoader:
    SUPPORTED_EXTENSIONS: Set[str] = {".npy"}

    def read_header(self, file_path: Path) -> FileInfo:
        arr = np.load(file_path, mmap_mode="r")
        dim_order = _DIM_ORDER.get(arr.ndim, tuple(f"D{i}" for i in range(arr.ndim)))
        return FileInfo(shape=arr.shape, dtype=arr.dtype, dim_order=dim_order)

    def load(self, file_path: Path) -> Record:
        arr = np.load(file_path)
        dim_order_tuple = _DIM_ORDER.get(arr.ndim, tuple(f"D{i}" for i in range(arr.ndim)))
        meta = {"shape": list(arr.shape), "ndim": arr.ndim, "dim_order": "".join(dim_order_tuple)}
        return record_from(arr, meta, kind="intensity")
