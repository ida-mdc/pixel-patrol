import logging
import math
from typing import Dict, Any, Iterator, List, Literal, Optional, Set, Tuple, Union
import os
from datetime import datetime
from pathlib import Path
import polars as pl
from yaspin import yaspin

from pixel_patrol_base.utils.utils import format_bytes_to_human_readable
from pixel_patrol_base.core.contracts import PixelPatrolLoader

logger = logging.getLogger(__name__)


def _format_size_readable(size_bytes: int) -> str:
    if size_bytes == 0:
        return "0 Bytes"
    names = ("Bytes", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    return f"{round(size_bytes / math.pow(1024, i), 2)} {names[i]}"


def _discover_files(
    bases:               List[Path],
    accepted_extensions: Union[Set[str], str],
    folder_extensions:   Optional[Set[str]] = None,
) -> Iterator[Tuple[Path, dict]]:
    """Yield (file_path, file_metadata) for every matching file under bases, one at a time.

    accepted_extensions:
      "all"     → accept every file regardless of extension
      Set[str]  → accept only files whose suffix is in the set (dot-prefixed, lowercase)

    folder_extensions: dot-stripped lowercase extensions that identify folder datasets
      (e.g. {"zarr"}).  Matching directories are yielded as files and their contents
      are not descended into.

    file_metadata contains all filesystem attributes compatible with the original
    processing output: path, name, type, parent, depth, size_bytes, file_extension,
    modification_date, size_readable, imported_path, common_base, and
    imported_path_short (only when len(bases) > 1).

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
            m: Dict[str, Any] = {
                "path":              str(path),
                "name":              path.name,
                "type":              "file",
                "parent":            str(path.parent),
                "depth":             depth,
                "size_bytes":        stat.st_size,
                "file_extension":    ext,
                "modification_date": datetime.fromtimestamp(stat.st_mtime),
                "size_readable":     _format_size_readable(stat.st_size),
                "imported_path":     base_str,
                "common_base":       common_base_name,
            }
            if multiple_bases:
                m["imported_path_short"] = path_short
            return m

        for dirpath, dirnames, filenames in os.walk(base_path, topdown=True):
            dir_path = Path(dirpath)
            depth    = len(dir_path.parts) - len(base_path.parts)

            if folder_exts:
                keep_dirs: List[str] = []
                for dname in sorted(dirnames):
                    sub = dir_path / dname
                    ext_raw = sub.suffix.lower().lstrip(".")
                    if ext_raw in folder_exts and (extensions is None or ("." + ext_raw) in extensions):
                        try:
                            stat = sub.stat()
                        except OSError:
                            continue
                        yield sub, _make_meta(sub, stat, depth + 1)
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


def make_basic_record(path: Path, base: Path, is_folder: bool = False) -> Dict[str, Any]:
    """
    Create a basic metadata record for a file or folder,
    computing depth relative to `base` and normalizing extensions.
    """
    try:
        st = path.stat() if not is_folder else None
    except Exception as e:
        logger.warning(f"Failed stat for {path}: {e}")
        return {}

    depth = len(path.parts) - len(base.parts)

    record: Dict[str, Any] = {
        "path": str(path),
        "name": path.name,
        "type": "folder" if is_folder else "file",
        "parent": str(path.parent) if path != base else None,
        "depth": depth,
        "size_bytes": 0 if is_folder else st.st_size,
        "modification_date": datetime.fromtimestamp(os.path.getmtime(path)),
        "file_extension": None if is_folder else path.suffix.lstrip(".").lower(),
        "imported_path": str(base),
    }
    return record


def walk_filesystem(
    bases: List[Path],
    accepted_extensions: Set[str] | Literal["all"],
    loader: Optional[PixelPatrolLoader] = None,
) -> pl.DataFrame:
    """
    - Only include files and loader-supported folder datasets (no plain directories).
    - accepted_extensions == "all": include all files + any folder datasets supported by the loader.
    - accepted_extensions is a set: include files with suffix in set; include folder datasets only if they intersect loader.FOLDER_EXTENSIONS.
    """
    records: List[dict] = []
    include_all = accepted_extensions == "all"

    is_folder_check = (loader is not None) and \
                      hasattr(loader, "is_folder_supported")  and \
                      (include_all or
                       not accepted_extensions.isdisjoint(getattr(loader, "FOLDER_EXTENSIONS", set())))
    folder_support_fn = loader.is_folder_supported if is_folder_check else None


    with yaspin(text="Scanning for files to process ...", timer=True).cyan.shark as spinner:
        spinner._interval = .3
        for base in bases:
            for root, dirnames, filenames in os.walk(base, topdown=True):
                dirpath = Path(root)

                keep: List[str] = []

                if is_folder_check:
                    for d in dirnames:
                        sub = dirpath / d
                        if folder_support_fn(sub):
                            records.append(make_basic_record(sub, base, is_folder=False))
                        else:
                            keep.append(d)
                    dirnames[:] = keep

                # Files
                for name in filenames:
                    p = dirpath / name
                    if include_all or p.suffix.lower().lstrip(".") in accepted_extensions:
                        records.append(make_basic_record(p, base, is_folder=False))

    return pl.DataFrame(records) if records else pl.DataFrame()


def _aggregate_folder_sizes(df: pl.DataFrame) -> pl.DataFrame:
    """
    Aggregates file sizes up to their parent folders in the DataFrame.
    Assumes df contains 'path', 'type', 'parent', 'size_bytes', 'depth' columns.
    This version aims to be more Polars-idiomatic.
    """
    if df.is_empty():
        return df

    # Ensure 'size_bytes' is numerical
    df = df.with_columns(pl.col("size_bytes").cast(pl.Int64))

    # Initialize a 'current_size' column that will be updated
    # Files keep their original size. Folders initially have 0 or their own direct size if applicable.
    # The sum for folders will be calculated from their children.
    df = df.with_columns(
        pl.when(pl.col("type") == "file")
        .then(pl.col("size_bytes"))
        .otherwise(0)  # Start folder size from 0, or could be initial direct size if it applies
        .alias("temp_calculated_size")
    )

    # Get unique depths in reverse order to process from leaves upwards
    # Filter out folders at depth 0, as they might not have a parent in the dataframe to aggregate to.
    unique_depths = sorted(df["depth"].unique().to_list(), reverse=True)

    # If your base directory is included as a folder with depth 0 and no parent in the df,
    # the aggregation will stop there. This is generally desired.

    # Iterate from deepest folders up to the base-level folders
    for current_depth in unique_depths:
        # Sum sizes of direct children at (current_depth + 1) for parents at current_depth
        # We need to compute the sum of 'temp_calculated_size' for all children
        # grouped by their 'parent' path (which corresponds to the current folder's path).

        # Calculate children sizes to aggregate to parents at current_depth
        # This aggregates sizes of *all* items (files and subfolders) at depth 'current_depth'
        # based on their 'parent' column.

        children_sums_for_parents = df.filter(pl.col("depth") == current_depth + 1) \
            .group_by("parent") \
            .agg(pl.col("temp_calculated_size").sum().alias("children_total_size"))

        # Now, join these sums back to the main DataFrame
        # Update the 'temp_calculated_size' for folders at 'current_depth'
        # by adding the sum of their children.

        df = df.join(
            children_sums_for_parents,
            left_on="path",  # Folder's path is the parent for its children
            right_on="parent",
            how="left"
        ).with_columns(
            pl.when(pl.col("type") == "folder")
            .then(
                pl.col("temp_calculated_size") + pl.col("children_total_size").fill_null(0)
            )
            .otherwise(pl.col("temp_calculated_size"))  # Files keep their original size
            .alias("temp_calculated_size")
        ).drop("children_total_size")  # Drop the temporary join column

    # After aggregation, the 'temp_calculated_size' column contains the final aggregated sizes.
    # Replace the original 'size_bytes' with this aggregated column.
    df = df.with_columns(pl.col("temp_calculated_size").alias("size_bytes")).drop("temp_calculated_size")

    # Drop the temporary Path objects if they were created before
    # (In this revised version, we don't create path_obj/parent_obj explicitly in the DF)
    # If the initial scan_directory_to_dataframe already returns Path objects and they are stored
    # as object dtype, they would need to be handled, but it's better to store strings then convert as needed.

    return df
