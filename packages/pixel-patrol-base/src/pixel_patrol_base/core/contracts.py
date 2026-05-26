from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, Iterable, Iterator, Set, Any, Dict, List, Optional, Tuple, Union

import polars as pl
from pathlib import Path

from dash.development.base_component import Component

from pixel_patrol_base.core.record import Record
from pixel_patrol_base.core.specs import ProcessResult, RecordSpec, ProcessorOutput


class ChunkKind(StrEnum):
    """Declares which level of chunking a processor operates on.

    LEAF        — operates on leaf chunks (user-configured granularity: XY tiles,
                  TZ slices, etc.). Most metric processors are this kind.
    MEMORY      — operates on memory-safe chunks, ignoring user leaf config.
                  Thumbnail is this kind — it computes on the full spatial extent.
    FULL_RECORD — receives the full record with no chunking applied.
                  Not yet handled by the pipeline; reserved for future use.
    """
    LEAF        = "leaf"
    MEMORY      = "memory"
    FULL_RECORD = "full_record"


@dataclass(frozen=True)
class FileInfo:
    """Header-level metadata returned by loader.read_header() — no pixel data loaded.

    Used by _plan_tasks to decide task routing (batch vs. chunk vs. sub-image)
    without loading any pixel data.
    """
    shape:     Tuple[int, ...]
    dtype:     Any                # numpy dtype or compatible
    dim_order: Tuple[str, ...]    # e.g. ('Z', 'Y', 'X')
    n_images:  int = 1            # >1 for container formats (LMDB, multi-series OME-TIFF, …)


class PixelPatrolLoader(Protocol):
    NAME: str
    SUPPORTED_EXTENSIONS: Set[str]
    OUTPUT_SCHEMA: Dict[str, Any]
    OUTPUT_SCHEMA_PATTERNS: List[tuple[str, Any]]
    FOLDER_EXTENSIONS: Set[str]

    def is_folder_supported(self, path: Path) -> bool: ...

    def read_header(self, file_path: Path) -> FileInfo:
        """Read file header only; return shape/dtype/dim_order without loading pixels.

        For container formats (n_images > 1), shape/dtype/dim_order describe a
        representative sub-image (typically the first). n_images is the total count.
        Must be picklable — no open file handles in instance state after return.
        """
        ...

    def load(self, file_path: Path) -> Record:
        """Load pixel data for a single-image file; return a Record.

        For container formats use load_range() instead.
        Must be picklable — no open file handles in instance state after return.
        """
        ...

    def load_range(self, file_path: Path, start: int, stop: int) -> Iterator[Tuple[str, Record]]:
        """Yield (child_id, record) for sub-images [start, stop) one at a time.

        Required only for container formats (FileInfo.n_images > 1).
        child_id is a stable string identifier for the sub-image within the container.
        Streams sub-images so each can be freed before the next is loaded.
        Must be picklable — no open file handles in instance state after return.
        """
        ...


class PixelPatrolProcessor(Protocol):
    NAME: str
    CHUNK_KIND: ChunkKind
    INPUT: RecordSpec
    OUTPUT: ProcessorOutput            # "features" or "record"
    OUTPUT_SCHEMA: Dict[str, Any]
    def run_chunk(self, record: Record) -> Dict[str, Any]: ...
    def get_aggregation(self, name: str) -> Optional[Any]: ...


class PixelPatrolWidget(Protocol):
    NAME: str                       # human readable name
    TAB: str                        # WidgetCategories value
    REQUIRES: Set[str]              # columns required to render
    REQUIRES_PATTERNS: Optional[Iterable[str]]  # optional regexes for dynamic cols

    def layout(self) -> List[Component]: ...
    def register(self, app, df_global: pl.DataFrame) -> None: ...
