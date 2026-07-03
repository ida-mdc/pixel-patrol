from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, Iterator, Set, Any, Dict, List, Optional, Tuple

from pathlib import Path

from pixel_patrol_base.core.record import Record
from pixel_patrol_base.core.specs import ProcessResult, RecordSpec, ProcessorOutput


class ChunkKind(StrEnum):
    """Declares which level of chunking a processor operates on.

    LEAF        - operates on leaf chunks (user-configured granularity: XY tiles,
                  TZ slices, etc.). Most metric processors are this kind.
    MEMORY      - operates on memory-safe chunks, ignoring user leaf config.
                  Thumbnail is this kind - it computes on the full spatial extent.
    FULL_RECORD - receives the full record with no chunking applied.
                  Not yet handled by the pipeline; reserved for future use.
    """
    LEAF        = "leaf"
    MEMORY      = "memory"
    FULL_RECORD = "full_record"


@dataclass(frozen=True)
class FileInfo:
    """Header-level metadata returned by loader.read_header() - no pixel data loaded.

    Used by _plan_tasks to decide task routing (batch vs. chunk vs. sub-image)
    without loading any pixel data.
    """
    shape:     Tuple[int, ...]
    dtype:     Any                # numpy dtype or compatible
    dim_order: Tuple[str, ...]    # e.g. ('Z', 'Y', 'X')
    n_images:  int = 1            # >1 for container formats (LMDB, multi-series OME-TIFF, …)


@dataclass(frozen=True)
class MultiFileImage:
    """One logical image whose planes live in separate files.

    A source yields this in place of a single path when several files should be
    loaded and stacked into one image - e.g. one file per channel (Cell Painting),
    Z-slice, or timepoint. A multi-file-aware loader loads each member, squeezes it
    to its spatial plane, stacks them along `axis`, and labels them with `names`
    (e.g. channel names). The pipeline treats it as an opaque task input.
    """
    paths: Tuple[str, ...]        # member files, in stack order
    axis:  str = "C"              # new dimension to stack along (C / Z / T / …)
    names: Tuple[str, ...] = ()   # optional per-member labels (e.g. channel names)


class PixelPatrolLoader(Protocol):
    NAME: str
    SUPPORTED_EXTENSIONS: Set[str]
    OUTPUT_SCHEMA: Dict[str, Any]
    OUTPUT_SCHEMA_PATTERNS: List[tuple[str, Any]]
    FOLDER_EXTENSIONS: Set[str]
    CONTAINER_EXTENSIONS: Set[str]  # extensions that may have n_images > 1; always read_header

    def is_folder_supported(self, path: Path) -> bool: ...

    def read_header(self, file_path: Path) -> FileInfo:
        """Read file header only; return shape/dtype/dim_order without loading pixels.

        For container formats (n_images > 1), shape/dtype/dim_order describe a
        representative sub-image (typically the first). n_images is the total count.
        Must be picklable - no open file handles in instance state after return.
        """
        ...

    def load(self, file_path: Path) -> Record:
        """Load pixel data for a single-image file; return a Record.

        For container formats use load_range() instead.
        Must be picklable - no open file handles in instance state after return.
        """
        ...

    def load_range(self, file_path: Path, start: int, stop: int) -> Iterator[Tuple[str, Record]]:
        """Yield (child_id, record) for sub-images [start, stop) one at a time.

        Required only for container formats (FileInfo.n_images > 1).
        child_id is a stable string identifier for the sub-image within the container.
        Streams sub-images so each can be freed before the next is loaded.
        Must be picklable - no open file handles in instance state after return.
        """
        ...


class PixelPatrolSource(Protocol):
    """Discovers input files and their filesystem-level metadata.

    A source owns the "where" of a run: given base locations it yields one
    (file_path, file_metadata) tuple per matching file, which _plan_tasks then
    turns into processing tasks. The built-in LocalFilesystemSource walks the
    local filesystem; other sources (e.g. an S3 manifest reader) plug in via the
    "pixel_patrol.source_plugins" entry point and yield the same tuple shape.

    A source is orthogonal to the loader: the source decides where files are and
    their filesystem metadata; the loader decides how to read their pixels.

    IO_BOUND (optional, default False) signals that loading is dominated by
    network/IO waits rather than CPU, so the pipeline should use a threaded
    cluster instead of a process pool. Remote sources should set it True.
    """
    NAME: str
    IO_BOUND: bool = False

    def can_handle(self, base: str) -> bool:
        """True if this source recognizes the base (e.g. by URI scheme).

        Used to auto-route a run to the right source when none is named
        explicitly. Sources should match narrowly so they do not shadow one
        another (the local source claims non-URI paths; a remote source claims
        its own scheme).
        """
        ...

    def resolve_bases(self, new_bases: List[str], existing_bases: List[str], base_dir: Optional[Path]) -> List[Any]:
        """Validate/normalize new input bases and merge them with the existing ones.

        Called by Project.add_paths so the source - not the core - owns what makes
        a valid input and how bases are normalized. Invalid bases are dropped
        (with a warning). Returns the full base list to store on the project.

        Each source enforces its own rules: the local source requires an existing
        directory within base_dir and de-duplicates sub/superpaths; a manifest
        source requires a readable .csv; etc. base_dir is the project's (local)
        output directory, provided for sources that resolve inputs relative to it.
        """
        ...

    def discover(
        self,
        bases:               List[Path],
        accepted_extensions: Set[str] | str,
        folder_extensions:   Optional[Set[str]] = None,
        base_dir:            Optional[Path] = None,
    ) -> Iterator[Tuple[Path, Dict[str, Any]]]:
        """Yield (file_path, file_metadata) for every matching file under bases.

        file_metadata must carry the standard columns the pipeline expects:
        path, name, type, parent, depth, size_bytes, file_extension,
        modification_date, imported_path, common_base (and imported_path_short
        when more than one base is given). A source that cannot cheaply provide a
        field (e.g. size_bytes for a remote object) may set it to 0.
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


