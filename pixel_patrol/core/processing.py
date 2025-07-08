import polars as pl
from pathlib import Path
import os
from typing import List, Optional, Dict, Set, Tuple
import logging

from pixel_patrol.core.file_system import _fetch_single_directory_tree, _aggregate_folder_sizes, make_basic_record
from pixel_patrol.utils.utils import format_bytes_to_human_readable
from pixel_patrol.core.image_operations_and_metadata import extract_image_metadata, available_columns

logger = logging.getLogger(__name__)

########### build_paths_df ##########

PATHS_DF_EXPECTED_SCHEMA = {
    "path": pl.String,
    "name": pl.String,
    "type": pl.String,
    "parent": pl.String,
    "depth": pl.Int64,
    "size_bytes": pl.Int64,
    "modification_date": pl.Datetime(time_unit="us", time_zone=None),
    "file_extension": pl.String,
    "size_readable": pl.String,
    "imported_path": pl.String
}

# TODO: still needs some more clean up
def build_paths_df(paths: List[Path]) -> Optional[pl.DataFrame]:
    """
    Traverses the given directories, collects file and folder metadata,
    aggregates folder sizes, and returns a normalized Polars DataFrame.
    """
    if not paths:
        return pl.DataFrame([], schema=PATHS_DF_EXPECTED_SCHEMA)

    all_trees: List[pl.DataFrame] = []
    for base in paths:
        try:
            tree = _fetch_single_directory_tree(base)
            tree = tree.with_columns(pl.lit(str(base)).alias("imported_path"))
            all_trees.append(tree)
        except ValueError as e:
            logger.warning(f"Skipping invalid directory {base!r}: {e}")
        except OSError as e:
            logger.warning(f"I/O error processing {base!r}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error processing {base!r}: {e}", exc_info=True)

    if not all_trees:
        return pl.DataFrame([], schema=PATHS_DF_EXPECTED_SCHEMA)

    df = pl.concat(all_trees, how="vertical_relaxed")

    # normalize file extensions
    df = df.with_columns(
        pl.when(pl.col("type") == "file")
          .then(
              pl.coalesce(
                  pl.col("file_extension").str.to_lowercase().fill_null(""),
                  pl.col("name").str.extract(r"\.([^.]+)$", 1).str.to_lowercase().fill_null("")
              )
          )
          .otherwise(pl.lit(None))
          .alias("file_extension")
    )

    # aggregate folder sizes and add human-readable sizes
    df = _aggregate_folder_sizes(df)
    df = df.with_columns(
        pl.col("size_bytes")
          .map_elements(format_bytes_to_human_readable)
          .alias("size_readable")
    )

    # enforce schema order and types
    return df.select([
        pl.col(c).cast(t) if c in df.columns
        else pl.lit(None, dtype=t).alias(c)
        for c, t in PATHS_DF_EXPECTED_SCHEMA.items()
    ])



######### build_images_df helpers + function ##########

def find_common_base(paths: List[str]) -> str:
    """
    Finds the common base path among a list of paths.
    """
    if not paths:
        return ""
    if len(paths) == 1:
        return str(Path(paths[0]).parent) + "/"  # Ensure it ends with a slash if it's a directory

    # Convert to Path objects to use their methods
    path_objects = [Path(p) for p in paths]

    # Find the shortest path, as it might be part of the common base
    shortest_path = min(path_objects, key=lambda p: len(str(p)))

    common_parts = []
    for part in shortest_path.parts:
        if all(part in p.parts for p in path_objects):
            common_parts.append(part)
        else:
            break

    # Reconstruct the common base
    common_base = Path(*common_parts)

    # Ensure it ends with a separator if it's a directory
    return str(common_base) + "/" if common_base.is_dir() else str(common_base)

################################## Shared Helpers for build_images_df_from functions ###################################

def _paths_from_dirs(
    bases: List[Path],
    accepted_extensions: Set[str]
) -> List[Tuple[Path, Path]]:
    """
    Walk each base dir, filter by extension, and return a list of (file_path, base_dir) tuples.
    """
    matched: List[Tuple[Path, Path]] = []
    for base in bases:
        for root, _, files in os.walk(base):
            for name in files:
                ext = Path(name).suffix.lower().lstrip('.')
                if ext in accepted_extensions:
                    matched.append((Path(root) / name, base))
    return matched

def _filter_paths_df(paths_df: pl.DataFrame, extensions: Set[str]) -> pl.DataFrame:
    """Return only file rows whose extension (lower-cased) is in our set."""
    return paths_df.filter(
        (pl.col("type") == "file")
        & pl.col("file_extension").str.to_lowercase().is_in(list(extensions))
    )


def _get_deep_image_metadata_df(paths: List[Path], required_cols: List[str]) -> pl.DataFrame:
    """Loop over paths, extract_image_metadata, return DataFrame (may be empty)."""
    metadata = []
    for p in paths:
        try:
            meta = extract_image_metadata(p, required_cols)
            if meta:
                metadata.append({"path": str(p), **meta})
        except Exception as e:
            logger.warning(f"Metadata extraction failed for {p}: {e}")
    return pl.DataFrame(metadata)


def _finalize(basic: pl.DataFrame, deep: pl.DataFrame) -> pl.DataFrame:
    """Join basic + deep on 'path' and enforce schema order/types."""
    joined = basic.join(deep, on="path", how="left")
    # ensure all columns from PATHS_DF_EXPECTED_SCHEMA appear, cast appropriately
    cols = []
    for col_name, col_type in PATHS_DF_EXPECTED_SCHEMA.items():
        if col_name in joined.columns:
            cols.append(pl.col(col_name).cast(col_type))
        else:
            cols.append(pl.lit(None, dtype=col_type).alias(col_name))
    return joined.select(cols)

################################### Main Function to Build Images DataFrame ###################################

def build_images_df_from_paths_df(paths_df: pl.DataFrame,  extensions: Set[str]) -> Optional[pl.DataFrame]:
    """
    Extracts deep image-specific metadata for files after filtering for file extensions.

    Args:
        paths_df: A Polars DataFrame containing basic file system metadata for files.
        extensions: A set of file extensions to filter image files (e.g., {".jpg", ".png"}).
    Returns:
        An Optional Polars DataFrame containing combined basic and image-specific metadata,
        or None if no valid image data is available.
    """
    filtered_df = _filter_paths_df(paths_df, extensions)
    if filtered_df.is_empty():
        logger.warning(f"No image files found in paths_df for extensions {extensions}")
        return None

    # Enforce match to PATHS_DF_EXPECTED_SCHEMA by casting existing columns and filling missing ones with nulls
    # TODO: I'm pretty sure its not needed as we build paths_df with the same make_basic_record function
    basic_file_df = filtered_df.select([
        pl.col(c).cast(t)
        for c, t in PATHS_DF_EXPECTED_SCHEMA.items()
        if c in filtered_df.columns
    ] + [
        pl.lit(None, dtype=PATHS_DF_EXPECTED_SCHEMA[c]).alias(c)
        for c in PATHS_DF_EXPECTED_SCHEMA
        if c not in filtered_df.columns
    ])

    # TODO: speed test - paths to list instead of all polars native operations?
    # but.. extract_image_metadata is operating on individual paths, so it should be fine
    required = available_columns()
    paths = [Path(p) for p in basic_file_df["path"].to_list()]
    deep_image_df = _get_deep_image_metadata_df(paths, required)

    return _finalize(basic_file_df, deep_image_df)


def build_images_df_from_file_system(bases: List[Path], selected_extensions: Set[str]) -> Optional[pl.DataFrame]:
    """
    Performs a single-pass scan over the file system to find image files,
    collect their basic file system metadata, and extract deep image-specific metadata.
    Returns a complete images_df.
    """

    path_base_pairs = _paths_from_dirs(bases, selected_extensions)
    if not path_base_pairs:
        logger.warning(
            f"No image files found in the provided directories for extensions: {selected_extensions}"
        )
        return None

    basic_file_df = pl.DataFrame([
        make_basic_record(path, base=base, is_folder=False)
        for path, base in path_base_pairs
    ])

    deep_image_df = _get_deep_image_metadata_df([p for p, _ in path_base_pairs], available_columns())

    return _finalize(basic_file_df, deep_image_df)


###################################### Other #####################################################

def count_file_extensions(paths_df: Optional[pl.DataFrame]) -> Dict[str, int]:
    """
    Counts file extensions from a Polars DataFrame containing file system paths.
    Returns a dictionary { 'extension': count, 'all_files': total_count } for all files.
    Files without extensions are discarded.

    Args:
        paths_df: The Polars DataFrame containing file system path data,
                  expected to have 'type' and 'file_extension' columns.

    Returns:
        A dictionary with file extension counts and a total count under 'all_files'.
        Returns {'all_files': 0} if paths_df is not available or empty.
    """
    if paths_df is None or paths_df.is_empty():
        logger.warning("No paths DataFrame provided or it's empty. Cannot count file extensions.")
        return {"all_files": 0}

    required_cols = {"type", "file_extension"}
    if not required_cols.issubset(paths_df.columns):
        logger.error(
            f"Paths DataFrame is missing required columns for extension counting: {required_cols - set(paths_df.columns)}")
        return {"all_files": paths_df.height}

    df_files = paths_df.filter(pl.col("type") == "file")
    df_files = df_files.filter(pl.col("file_extension").is_not_null() & (pl.col("file_extension") != ""))

    if df_files.is_empty():
        logger.info("No files with valid extensions found in the DataFrame.")
        return {"all_files": 0}

    grouped = df_files.group_by("file_extension").agg(pl.count().alias("count"))
    result = {row["file_extension"]: row["count"] for row in grouped.iter_rows(named=True)}
    result["all_files"] = df_files.height

    return result
