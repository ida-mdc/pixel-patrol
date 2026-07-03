from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

from pixel_patrol_base.core.file_system import _discover_files
from pixel_patrol_base.core.uri import is_remote_uri
from pixel_patrol_base.core.validation import resolve_and_validate_project_path
from pixel_patrol_base.utils.path_utils import process_new_paths_for_redundancy


class LocalFilesystemSource:
    """Default source: walks the local filesystem for matching files.

    Thin wrapper around _discover_files so local discovery is a plugin like any
    other source, selectable and replaceable via the source-plugin machinery.
    """
    NAME = "local_filesystem"

    def can_handle(self, base: str) -> bool:
        return not is_remote_uri(base)

    def resolve_bases(self, new_bases: List[str], existing_bases: List[str], base_dir: Optional[Path]) -> List[Path]:
        """A local base must be an existing directory within base_dir; sub/superpath
        redundancy against the existing bases is resolved away."""
        validated = [
            resolved
            for base in new_bases
            if (resolved := resolve_and_validate_project_path(base, base_dir)) is not None
        ]
        existing_paths = {Path(p) for p in existing_bases}
        return list(process_new_paths_for_redundancy(validated, existing_paths))

    def discover(
        self,
        bases:               List[Path],
        accepted_extensions: Set[str] | str,
        folder_extensions:   Optional[Set[str]] = None,
        base_dir:            Optional[Path] = None,
    ) -> Iterator[Tuple[Path, Dict[str, Any]]]:
        yield from _discover_files(bases, accepted_extensions, folder_extensions, base_dir=base_dir)
