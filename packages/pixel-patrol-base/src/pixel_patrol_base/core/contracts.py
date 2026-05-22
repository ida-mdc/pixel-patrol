from enum import StrEnum
from typing import Protocol, Iterable, Set, Any, Dict, List, Optional, Union

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
    """
    LEAF        = "leaf"
    MEMORY      = "memory"
    FULL_RECORD = "full_record"


class PixelPatrolLoader(Protocol):
    NAME: str
    SUPPORTED_EXTENSIONS: Set[str]
    OUTPUT_SCHEMA: Dict[str, Any]
    OUTPUT_SCHEMA_PATTERNS: List[tuple[str, Any]]
    FOLDER_EXTENSIONS: Set[str]
    def is_folder_supported(self, path: Path) -> bool: ...
    def load(self, source: str) -> Union[Record, Dict[str, Record]]: ...


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
