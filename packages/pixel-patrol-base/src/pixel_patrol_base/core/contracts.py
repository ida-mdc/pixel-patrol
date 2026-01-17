from typing import Protocol, Iterable, Set, Any, Dict, List, Optional, Union

import polars as pl
from pathlib import Path

from dash.development.base_component import Component

from pixel_patrol_base.core.record import Record
from pixel_patrol_base.core.specs import ProcessResult, RecordSpec, ProcessorOutput


class PixelPatrolLoader(Protocol):
    NAME: str
    SUPPORTED_EXTENSIONS: Set[str]
    OUTPUT_SCHEMA: Dict[str, Any]
    OUTPUT_SCHEMA_PATTERNS: List[tuple[str, Any]]
    FOLDER_EXTENSIONS: Set[str]
    def is_folder_supported(self, path: Path) -> bool: ...
    def load(self, source: str) -> Union[Record, List[Record]]: ...


class PixelPatrolProcessor(Protocol):
    NAME: str
    INPUT: RecordSpec
    OUTPUT: ProcessorOutput            # "features" or "record"
    def run(self, art: Record) -> ProcessResult: ...


class PixelPatrolWidget(Protocol):
    NAME: str                       # human readable name
    TAB: str                        # WidgetCategories value
    REQUIRES: Set[str]              # columns required to render
    REQUIRES_PATTERNS: Optional[Iterable[str]]  # optional regexes for dynamic cols

    def layout(self) -> List[Component]: ...
    def register(self, app, df_global: pl.DataFrame) -> None: ...
