"""
NullLoader: returns zero-filled arrays without reading disk content.

Used by benchmarks to isolate pipeline overhead from I/O cost.
The loader is configured with a shape/dtype/dim_order/n_images registry keyed
by path string; _discover_files never runs - benchmark_pipeline.py builds a
fake file stream instead.

For large shapes (huge_multidim scenario), load() returns a _LazyZeros proxy
rather than allocating the full array. The proxy allocates only the requested
slice in __getitem__, so MemoryChunkTask workers only allocate their chunk.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterator, Set, Tuple

import numpy as np

from pixel_patrol_base.core.contracts import FileInfo
from pixel_patrol_base.core.record import Record, record_from


class _LazyZeros:
    """Zero array that allocates only on slice, not on construction.

    Lets workers handle a huge logical image without loading it all at once.
    Supports __getitem__ with slice/None tuples (the only form used by the pipeline).
    """
    def __init__(self, shape: tuple, dtype: Any) -> None:
        self.shape = shape
        self.dtype = np.dtype(dtype)

    def __getitem__(self, slices) -> np.ndarray:
        if not isinstance(slices, tuple):
            slices = (slices,)
        out_shape = []
        for i, s in enumerate(slices):
            if isinstance(s, slice):
                out_shape.append(len(range(*s.indices(self.shape[i]))))
        for i in range(len(slices), len(self.shape)):
            out_shape.append(self.shape[i])
        return np.zeros(out_shape, dtype=self.dtype)

    def __array__(self, dtype=None) -> np.ndarray:
        arr = np.zeros(self.shape, dtype=self.dtype)
        return arr if dtype is None else arr.astype(dtype)


class NullLoader:
    """Zero-filling loader for benchmarks - ignores file content, returns zeros.

    entries: maps path string → (shape, dtype, dim_order tuple, n_images).
    lazy_threshold_mb: shapes whose uncompressed size exceeds this use _LazyZeros
                       so workers don't allocate the full array.
    Picklable: entries contains only plain Python types and numpy dtypes.
    """
    SUPPORTED_EXTENSIONS: Set[str] = {"*"}

    def __init__(
        self,
        entries: Dict[str, Tuple[tuple, Any, tuple, int]],
        lazy_threshold_mb: float = 64.0,
    ) -> None:
        self._entries = entries
        self._lazy_bytes = int(lazy_threshold_mb * 1024 * 1024)

    def read_header(self, file_path: Path) -> FileInfo:
        shape, dtype, dim_order, n_images = self._entries[str(file_path)]
        return FileInfo(shape=shape, dtype=dtype, dim_order=dim_order, n_images=n_images)

    def load(self, file_path: Path) -> Record:
        shape, dtype, dim_order, _ = self._entries[str(file_path)]
        size = int(np.prod(shape)) * np.dtype(dtype).itemsize
        data: Any = (
            _LazyZeros(shape, dtype) if size > self._lazy_bytes
            else np.zeros(shape, dtype=dtype)
        )
        meta = {"shape": list(shape), "ndim": len(shape), "dim_order": "".join(dim_order)}
        return record_from(data, meta, kind="intensity")

    def load_range(
        self, file_path: Path, start: int, stop: int
    ) -> Iterator[Tuple[str, Record]]:
        shape, dtype, dim_order, _ = self._entries[str(file_path)]
        meta = {"shape": list(shape), "ndim": len(shape), "dim_order": "".join(dim_order)}
        arr = np.zeros(shape, dtype=dtype)
        for i in range(start, stop):
            yield str(i), record_from(arr.copy(), meta, kind="intensity")
