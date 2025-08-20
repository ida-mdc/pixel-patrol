import importlib
import logging
import os
import re
from pathlib import Path
from typing import List, Optional, Dict, Set, Tuple, Type

import polars as pl

from pixel_patrol.core.file_system import _fetch_single_directory_tree, _aggregate_folder_sizes, make_basic_record
from pixel_patrol.core.image_operations_and_metadata import get_all_image_properties
from pixel_patrol.core.loader_interface import PixelPatrolLoader
from pixel_patrol.core.processor_interface import PixelPatrolProcessor
from pixel_patrol.plugins import discover_loader, discover_processor_plugins
from pixel_patrol.utils.df_utils import normalize_file_extension
from pixel_patrol.utils.path_utils import find_common_base
from pixel_patrol.utils.utils import format_bytes_to_human_readable

logger = logging.getLogger(__name__)


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


def _postprocess_basic_file_metadata_df(df: pl.DataFrame) -> pl.DataFrame:
    if df.is_empty():
        return df

    common_base = find_common_base(df["imported_path"].unique().to_list())

    df = df.with_columns([
        pl.col("modification_date").dt.month().alias("modification_month"),
        pl.col("imported_path").str.replace(common_base, "", literal=True).alias("imported_path_short"),
        pl.col("size_bytes").map_elements(format_bytes_to_human_readable).alias("size_readable"),
    ])

    return df


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
            logger.warning(f"Error processing path {base!r}: {e}. Skipping")
        except OSError as e:
            logger.warning(f"I/O error processing {base!r}: {e}. Skipping")
        except Exception as e:
            logger.error(f"Unexpected error processing {base!r}: {e}. Skipping", exc_info=True)

    if not all_trees:
        return pl.DataFrame([], schema=PATHS_DF_EXPECTED_SCHEMA)

    df = pl.concat(all_trees, how="vertical_relaxed")

    df = normalize_file_extension(df)

    # aggregate folder sizes and add human-readable sizes
    df = _aggregate_folder_sizes(df)

    return _postprocess_basic_file_metadata_df(df)


def _scan_dirs_for_extensions(
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


def _get_deep_image_df(paths: List[Path], loader: str, read_pixel_data: bool = True) -> pl.DataFrame:
    """Loop over paths, get_all_image_properties, return DataFrame (may be empty).
    Optimized to minimize Python loop overhead where possible.
    """
    loader_instance = discover_loader(loader_id=loader)
    processors = discover_processor_plugins()

    static_features_specs: Dict[str, pl.DataType] = {}
    dynamic_schema_patterns: List[Tuple[str, pl.DataType]] = []

    for processor in processors:
        static_features_specs.update(processor.get_specification())
        dynamic_schema_patterns.extend(processor.get_dynamic_specification_patterns())

    images_dicts = []

    for p in paths:
        # try:
        # get_all_image_properties is still the bottleneck here if it reads pixels
        # This part remains in the Python loop as it depends on external I/O and Dask/NumPy processing.
        image_dict = get_all_image_properties(p, read_pixel_data, loader_instance, processors)
        if image_dict:
            # Add 'path' at this stage, as it's directly available from the loop
            images_dicts.append({"path": str(p), **image_dict})
        # except Exception as e:
        #     logger.warning(f"Metadata extraction failed for {p}: {e}")

    if not images_dicts:
        # Return an empty DataFrame with a predefined schema if no data was processed
        # This prevents an empty DataFrame with inferred 'Object' types from being created.
        # We'll build the final schema first.
        # Find all unique keys across all collected dicts
        all_discovered_keys = set()
        for row_dict in images_dicts:
            all_discovered_keys.update(row_dict.keys())

        # Combine static schema with types derived from dynamic patterns
        final_polars_schema = static_features_specs.copy()
        for key in all_discovered_keys:
            if key in final_polars_schema:
                continue  # Already defined statically

            for pattern_str, dtype in dynamic_schema_patterns:
                if re.match(pattern_str, key):
                    final_polars_schema[key] = dtype
                    break  # Move to next key once a pattern matches

        return pl.DataFrame(images_dicts, schema=final_polars_schema)

    return pl.DataFrame(images_dicts)


def build_images_df_from_paths_df(paths_df: pl.DataFrame,  extensions: Set[str], loader: str) -> Optional[pl.DataFrame]:
    """
    Extracts deep image-specific metadata for files after filtering for file extensions.

    Args:
        paths_df: A Polars DataFrame containing basic file system metadata for files.
        extensions: A set of file extensions to filter image files (e.g., {".jpg", ".png"}).
        loader: The identifier of the loader to be used for processing files.
    Returns:
        An Optional Polars DataFrame containing combined basic and image-specific metadata,
        or None if no valid image data is available.
    """
    filtered_df = _filter_paths_df(paths_df, extensions)
    if filtered_df.is_empty():
        logger.warning(f"No image files found in paths_df for extensions {extensions}")
        return None

    basic_file_df = _postprocess_basic_file_metadata_df(filtered_df)

    # TODO: speed test - paths to list instead of all polars native operations?
    # but.. get_all_image_properties is operating on individual paths, so it should be fine
    paths = [Path(p) for p in basic_file_df["path"].to_list()]
    deep_image_df = _get_deep_image_df(paths, loader)

    if len(deep_image_df) > 0:
        joined = basic_file_df.join(deep_image_df, on="path", how="left")
        return joined
    else:
        return basic_file_df


def build_images_df_from_file_system(bases: List[Path], selected_extensions: Set[str], loader: str) -> Optional[pl.DataFrame]:
    """
    Performs a single-pass scan over the file system to find image files,
    collect their basic file system metadata, and extract deep image-specific metadata.
    Returns a complete images_df.
    """

    path_base_pairs = _scan_dirs_for_extensions(bases, selected_extensions)
    if not path_base_pairs:
        logger.warning(
            f"No image files found in the provided directories for extensions: {selected_extensions}"
        )
        return None

    basic_file_df = pl.DataFrame([
        make_basic_record(path, base=base, is_folder=False)
        for path, base in path_base_pairs
    ])

    basic_file_df = normalize_file_extension(basic_file_df)
    basic_file_df = _postprocess_basic_file_metadata_df(basic_file_df)

    deep_image_df = _get_deep_image_df([p for p, _ in path_base_pairs], loader)

    joined = basic_file_df.join(deep_image_df, on="path", how="left")

    return joined


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
