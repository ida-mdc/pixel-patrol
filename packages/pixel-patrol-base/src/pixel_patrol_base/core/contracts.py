from typing import Protocol, Union, Iterable, Set, Any, Dict, List

import polars as pl

from .artifact import Artifact
from .specs import ArtifactSpec, OutputKind

ProcessResult = Union[dict, Artifact]

class PixelPatrolLoader(Protocol):
    NAME: str
    OUTPUT_SCHEMA: Dict[str, Any]
    OUTPUT_SCHEMA_PATTERNS: List[tuple[str, Any]]
    def load(self, source: str) -> Artifact: ...

class PixelPatrolProcessor(Protocol):
    NAME: str
    INPUT: ArtifactSpec
    OUTPUT: OutputKind            # "features" or "artifact"
    def run(self, art: Artifact) -> ProcessResult: ...


class PixelPatrolWidget(Protocol):
    NAME: str                       # human readable name
    TAB: str                        # WidgetCategories value
    REQUIRES: Set[str]              # columns required to render
    REQUIRES_PATTERNS: Iterable[str] | None  # optional regexes for dynamic cols

    def layout(self) -> list: ...
    def register(self, app, df_global: pl.DataFrame) -> None: ...
