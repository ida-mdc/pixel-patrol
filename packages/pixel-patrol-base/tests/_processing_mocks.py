"""Shared mock utilities for processing pipeline tests."""
from __future__ import annotations

import logging
import logging.handlers
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

import numpy as np

from pixel_patrol_base.core.contracts import ChunkKind, FileInfo
from pixel_patrol_base.core.record import Record, record_from


@dataclass
class MockEntry:
    shape:     Tuple[int, ...]
    dtype:     Any
    dim_order: Tuple[str, ...]
    n_images:  int  = 1
    fail:      bool = False


class MockLoader:
    SUPPORTED_EXTENSIONS: Set[str] = set()
    FOLDER_EXTENSIONS:    Set[str] = set()
    CONTAINER_EXTENSIONS: Set[str] = {"lmdb"}

    def __init__(self, entries: Dict[str, MockEntry]) -> None:
        self._entries = {k.replace("\\", "/"): v for k, v in entries.items()}

    @staticmethod
    def _key(file_path: Path) -> str:
        return str(file_path).replace("\\", "/")

    def read_header(self, file_path: Path) -> FileInfo:
        e = self._entries.get(self._key(file_path))
        if e is None or e.fail:
            raise FileNotFoundError(file_path)
        return FileInfo(shape=e.shape, dtype=np.dtype(e.dtype), dim_order=e.dim_order, n_images=e.n_images)

    def load(self, file_path: Path) -> Record:
        e = self._entries.get(self._key(file_path))
        if e is None or e.fail:
            raise FileNotFoundError(file_path)
        arr = np.zeros(e.shape, dtype=e.dtype)
        meta = {"shape": list(arr.shape), "ndim": arr.ndim, "dim_order": "".join(e.dim_order)}
        return record_from(arr, meta, kind="intensity")

    def load_range(self, file_path: Path, start: int, stop: int) -> Iterator[Tuple[str, Record]]:
        e = self._entries.get(self._key(file_path))
        if e is None or e.fail:
            raise FileNotFoundError(file_path)
        arr = np.zeros(e.shape, dtype=e.dtype)
        meta = {"shape": list(arr.shape), "ndim": arr.ndim, "dim_order": "".join(e.dim_order)}
        for i in range(start, stop):
            yield str(i), record_from(arr, meta, kind="intensity")


class capture_warnings:
    """Context manager: collect WARNING-level log messages from the processing module."""

    def __init__(self) -> None:
        self._records: List[str] = []
        self._handler: Optional[logging.Handler] = None

    def __enter__(self) -> List[str]:
        self._handler = logging.handlers.MemoryHandler(capacity=1000, flushLevel=logging.CRITICAL)
        self._handler.setLevel(logging.WARNING)
        import pixel_patrol_base.core.processing as _proc
        self._logger = logging.getLogger(_proc.__name__)
        self._logger.addHandler(self._handler)
        return self._records

    def __exit__(self, *_: Any) -> None:
        assert self._handler is not None
        self._handler.flush()
        self._records.extend(r.getMessage() for r in self._handler.buffer)
        self._logger.removeHandler(self._handler)


class _OpenInputSpec:
    axes         = None
    kinds        = None
    capabilities = None
    kind_patterns = None


class MockLeafProcessor:
    CHUNK_KIND   = ChunkKind.LEAF
    OUTPUT       = "features"

    def __init__(self, name: str, output: Dict[str, Any]) -> None:
        self.NAME          = name
        self.OUTPUT_SCHEMA = {k: type(v) for k, v in output.items()}
        self._output       = output
        self.INPUT         = _OpenInputSpec()

    def run_chunk(self, record: Any) -> Dict[str, Any]:
        return dict(self._output)

    def get_aggregation(self, name: str):
        return lambda rows, g_dims: rows[0][name] if rows and name in rows[0] else None


class MockMemoryProcessor:
    CHUNK_KIND   = ChunkKind.MEMORY
    OUTPUT       = "features"

    def __init__(self, name: str, output: Dict[str, Any]) -> None:
        self.NAME          = name
        self.OUTPUT_SCHEMA = {k: type(v) for k, v in output.items()}
        self._output       = output
        self.INPUT         = _OpenInputSpec()

    def run_chunk(self, record: Any) -> Dict[str, Any]:
        return dict(self._output)

    def get_aggregation(self, name: str):
        return lambda rows, g_dims: rows[0][name] if rows and name in rows[0] else None
