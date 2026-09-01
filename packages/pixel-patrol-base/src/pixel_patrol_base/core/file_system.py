import logging
from typing import Any, Callable, Dict, Iterator, List, Optional, Set, Tuple, Union
import os
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# Marks a discovered path as a folder dataset (zarr store, cell folder, …). Private to
# the discovery → planning handoff: the planner pops it, so it never reaches the table.
FOLDER_DATASET_KEY = "__folder_dataset"


def _folder_dataset_size(path: Path) -> int:
    """Total size of the files inside a folder dataset.

    A directory's own st_size is its inode - a few KB whatever it holds - so a
    folder-based record has to add its contents up to report the size_bytes the schema
    promises ("aggregated for folders"). One metadata pass per claimed dataset.
    """
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.stat(os.path.join(root, name)).st_size
            except OSError:
                continue
    return total


def _discover_files(
    bases:               List[Path],
    accepted_extensions: Union[Set[str], str],
    folder_extensions:   Optional[Set[str]] = None,
    base_dir:            Optional[Path] = None,
    is_folder_dataset:   Optional[Callable[[Path], bool]] = None,
) -> Iterator[Tuple[Path, dict]]:
    """Yield (file_path, file_metadata) for every matching file under bases, one at a time.

    accepted_extensions:
      "all"     → accept every file regardless of extension
      Set[str]  → accept only files whose suffix is in the set (dot-prefixed, lowercase)

    folder_extensions: dot-stripped lowercase extensions that identify folder datasets
      (e.g. {"zarr"}).  Matching directories are yielded as files and their contents
      are not descended into.

    is_folder_dataset: callable that returns True for directories that are one dataset.

    file_metadata contains all filesystem attributes compatible with the original
    processing output: path, name, type, parent, depth, size_bytes, file_extension,
    modification_date, imported_path, common_base, and
    imported_path_short (only when len(bases) > 1).  Folder datasets are sized by their
    contents and marked with FOLDER_DATASET_KEY, so planning can route them by header.

    No file is opened or loaded. Runs concurrently with _plan_tasks via the generator
    protocol - yields tasks to workers before the scan completes.
    """
    extensions: Optional[Set[str]] = (
        None if accepted_extensions == "all"
        else {e.lower() if e.startswith(".") else "." + e.lower() for e in accepted_extensions}
    )
    folder_exts: Set[str] = {e.lower().lstrip(".") for e in (folder_extensions or set())}

    str_bases = [str(Path(b).resolve()) for b in bases]
    common_base_path = os.path.commonpath(str_bases) if len(str_bases) > 1 else str_bases[0]
    common_base_name = Path(common_base_path).name or common_base_path
    multiple_bases   = len(bases) > 1

    for base in bases:
        base_path  = Path(base).resolve()
        base_str   = str(base_path)
        path_short = base_str[len(common_base_path):].lstrip(os.sep) if multiple_bases else None

        def _make_meta(path: Path, stat, depth: int) -> Dict[str, Any]:
            ext = path.suffix.lower().lstrip(".")
            anchor = base_dir if base_dir is not None else None
            path_val   = str(path.relative_to(anchor))   if anchor else str(path)
            parent_val = str(path.parent.relative_to(anchor)) if anchor else str(path.parent)
            import_val = str(base_path.relative_to(anchor)) if anchor else base_str
            m: Dict[str, Any] = {
                "path":              path_val,
                "name":              path.name,
                "type":              "file",
                "parent":            parent_val,
                "depth":             depth,
                "size_bytes":        stat.st_size,
                "file_extension":    ext,
                "modification_date": datetime.fromtimestamp(stat.st_mtime),
                "imported_path":     import_val,
                "common_base":       common_base_name,
            }
            if multiple_bases:
                m["imported_path_short"] = path_short
            return m

        for dirpath, dirnames, filenames in os.walk(base_path, topdown=True):
            dir_path = Path(dirpath)
            depth    = len(dir_path.parts) - len(base_path.parts)

            if folder_exts or is_folder_dataset is not None:
                keep_dirs: List[str] = []
                for dname in sorted(dirnames):
                    sub = dir_path / dname
                    ext_raw = sub.suffix.lower().lstrip(".")
                    by_ext = ext_raw in folder_exts and (extensions is None or ("." + ext_raw) in extensions)
                    if by_ext or (is_folder_dataset is not None and is_folder_dataset(sub)):
                        try:
                            stat = sub.stat()
                        except OSError:
                            continue
                        yield sub, {
                            **_make_meta(sub, stat, depth + 1),
                            "size_bytes": _folder_dataset_size(sub),
                            FOLDER_DATASET_KEY: True,
                        }
                    else:
                        keep_dirs.append(dname)
                dirnames[:] = keep_dirs
            else:
                dirnames.sort()

            for fname in sorted(filenames):
                path = dir_path / fname
                ext  = path.suffix.lower()
                if extensions is not None and ext not in extensions:
                    continue
                try:
                    stat = path.stat()
                except OSError:
                    continue
                yield path, _make_meta(path, stat, depth + 1)
