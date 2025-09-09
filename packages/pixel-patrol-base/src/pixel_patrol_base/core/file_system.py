import logging
from typing import Dict, Any, List, Literal, Optional, Set
import os
from datetime import datetime
from pathlib import Path
import zarr
import polars as pl

from pixel_patrol_base.utils.utils import format_bytes_to_human_readable

logger = logging.getLogger(__name__)

def is_zarr_store(path: Path) -> bool:
    """
    Robustly checks if a given path is a Zarr store (v2 or v3).

    This function uses the zarr library to attempt opening the store, which
    correctly handles both Zarr v2 and v3 specifications.

    Args:
        path: The pathlib.Path object to check.

    Returns:
        True if the path is a valid Zarr store, False otherwise.
    """
    try:
        store_obj = zarr.open(store=str(path.absolute()), mode='r')

        if isinstance(store_obj, zarr.Group):
            # A group is "processable" if it has any custom attributes.
            # A generic container group will have empty attrs.
            return bool(store_obj.attrs)

        return True

    except Exception as e:
        # Catches any error, indicating it's not a valid or accessible Zarr store.
        return False


def make_basic_record(path: Path, base: Path, is_folder: bool = False) -> Dict[str, Any]:
    """
    Create a basic metadata record for a file or folder,
    computing depth relative to `base` and normalizing extensions.
    """
    try:
        stat_func = path.stat if not is_folder else lambda: None
        st = stat_func() if not is_folder else None
    except Exception as e:
        logger.warning(f"Failed stat for {path}: {e}")
        return {}

    depth = len(path.parts) - len(base.parts)

    # TODO: I guess we're missing imported_path_short and modification_month that were created in preprocess_files
    # common_base = find_common_base(unique_folders) - should be added after
    # pl.col("modification_date").dt.month().alias("modification_month"),
    # pl.col("imported_path").str.replace(common_base, "", literal=True).alias("imported_path_short"),
    record: Dict[str, Any] = {
        "path": str(path),
        "name": path.name,
        "type": "folder" if is_folder else "file",
        "parent": str(path.parent) if path != base else None,
        "depth": depth,
        "size_bytes": 0 if is_folder else st.st_size,
        "size_readable": "0 Bytes" if is_folder else format_bytes_to_human_readable(st.st_size),
        "modification_date": datetime.fromtimestamp(os.path.getmtime(path)),
        "file_extension": None if is_folder else path.suffix.lstrip(".").lower(),
        "imported_path": str(base),
    }
    return record


def walk_filesystem(
    bases: List[Path],
    accepted_extensions: Set[str] | Literal["all"],
) -> pl.DataFrame:
    """
    - Only include files and Zarr stores (no plain directories).
    - accepted_extensions == "all": include all files + Zarr stores.
    - accepted_extensions is a set: include files with suffix in set; include Zarr if 'zarr' in set.
    """
    records: List[dict] = []
    include_all = accepted_extensions == "all"

    for base in bases:
        for root, dirnames, filenames in os.walk(base, topdown=True):
            dirpath = Path(root)

            # Zarr: add as single item and prune descent
            keep: List[str] = []
            for d in dirnames:
                sub = dirpath / d
                if is_zarr_store(sub):
                    if include_all or ("zarr" in accepted_extensions):
                        records.append(make_basic_record(sub, base, is_folder=False))
                    # do not descend into zarr
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
