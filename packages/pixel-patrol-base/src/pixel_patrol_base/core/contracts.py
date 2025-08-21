from typing import Protocol, Iterable, Set, Any, Dict, List

import polars as pl

from pixel_patrol_base.core.artifact import Artifact
from pixel_patrol_base.core.specs import ProcessResult, ArtifactSpec, ProcessorOutput


class PixelPatrolLoader(Protocol):
    NAME: str
    SUPPORTED_EXTENSIONS: Set[str]
    OUTPUT_SCHEMA: Dict[str, Any]
    OUTPUT_SCHEMA_PATTERNS: List[tuple[str, Any]]
    def load(self, source: str) -> Artifact: ...

class PixelPatrolProcessor(Protocol):
    NAME: str
    INPUT: ArtifactSpec
    OUTPUT: ProcessorOutput            # "features" or "artifact"
    def run(self, art: Artifact) -> ProcessResult: ...


class PixelPatrolWidget(Protocol):
    NAME: str                       # human readable name
    TAB: str                        # WidgetCategories value
    REQUIRES: Set[str]              # columns required to render
    REQUIRES_PATTERNS: Iterable[str] | None  # optional regexes for dynamic cols

    def layout(self) -> list: ...
    def register(self, app, df_global: pl.DataFrame) -> None: ...
