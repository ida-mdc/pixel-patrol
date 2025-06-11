import polars as pl
from pathlib import Path
import os
from datetime import datetime
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

def _fetch_single_directory_tree(base_path: Path) -> pl.DataFrame:
    """
    Traverses a single local directory to collect its file and folder structure.
    Returns a DataFrame representing the tree structure with initial sizes.
    """
    if not base_path.is_dir():
        raise ValueError(f"The path '{base_path}' is not a valid directory.")

    tree_data: List[Dict[str, Any]] = []

    for dirpath_str, dirnames, filenames in os.walk(base_path):
        dirpath = Path(dirpath_str)
        current_depth = len(dirpath.parts) - len(base_path.parts)
        parent_path_str = str(dirpath.parent) if dirpath != base_path else None

        # Add folder stats (initial size 0, will be aggregated later)
        tree_data.append({
            "path": str(dirpath),
            "name": dirpath.name,
            "type": "folder",
            "parent": parent_path_str,
            "depth": current_depth,
            "size_bytes": 0, # Initial size, will be summed up
            "modification_date": datetime.fromtimestamp(os.path.getmtime(dirpath)),
            "file_extension": None, # Folders don't have file extensions
        })

        # Add file stats
        for filename in filenames:
            file_path = dirpath / filename
            try:
                file_size = os.path.getsize(file_path)
                tree_data.append({
                    "path": str(file_path),
                    "name": filename,
                    "type": "file",
                    "parent": dirpath_str,
                    "depth": current_depth + 1,
                    "size_bytes": file_size,
                    "modification_date": datetime.fromtimestamp(os.path.getmtime(file_path)),
                    "file_extension": file_path.suffix[1:].lower() if file_path.suffix else "",
                })
            except FileNotFoundError:
                logger.warning(f"File not found during traversal: {file_path}")
            except PermissionError:  # Add specific handling for PermissionError
                logger.warning(f"Permission denied for file: {file_path}")
            except Exception as e:
                logger.error(f"Could not get stats for file {file_path}: {e}",
                             exc_info=True)  # Use error for general exceptions

    if not tree_data:
        return pl.DataFrame([], schema={ # Return empty DF with correct schema
            "path": pl.String, "name": pl.String, "type": pl.String,
            "parent": pl.String, "depth": pl.Int64, "size_bytes": pl.Int64,
            "modification_date": pl.Datetime, "file_extension": pl.String
        })

    return pl.DataFrame(tree_data)


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
