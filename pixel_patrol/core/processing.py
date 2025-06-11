import polars as pl
from pathlib import Path
from typing import List, Optional, Dict
import logging

from pixel_patrol.core.file_system import _fetch_single_directory_tree, _aggregate_folder_sizes
from pixel_patrol.utils.utils import format_bytes_to_human_readable
from pixel_patrol.core.project_settings import Settings
from pixel_patrol.core.image_operations_and_metadata import extract_image_metadata
from pixel_patrol.utils.widget import get_required_columns
from pixel_patrol.widgets.widget_interface import PixelPatrolWidget

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


def build_paths_df(paths: List[Path]) -> Optional[pl.DataFrame]:
    """
    Preprocesses the files and folders in the given paths into a single Polars DataFrame (paths_df).
    This function traverses directories, collects file system metadata for all entries,
    aggregates folder sizes, and adds a human-readable size column.
    """
    logger.debug(f"\n--- DEBUG: Entering build_paths_df ---")
    logger.debug(f"DEBUG: Input 'paths': {paths}")

    if not paths:
        logger.debug("DEBUG: 'paths' is empty. Returning empty DataFrame with schema.")
        return pl.DataFrame([], schema=PATHS_DF_EXPECTED_SCHEMA)

    all_trees: List[pl.DataFrame] = []
    logger.debug(f"DEBUG: Starting to process {len(paths)} base paths.")
    for base_path in paths:
        logger.debug(f"DEBUG: Processing base_path: {base_path}")
        try:
            tree_df = _fetch_single_directory_tree(base_path)
            logger.debug(
                f"DEBUG: _fetch_single_directory_tree for '{base_path}' returned shape: {tree_df.shape}, schema: {tree_df.schema}")
            tree_df = tree_df.with_columns(pl.lit(str(base_path)).alias("imported_path"))
            all_trees.append(tree_df)
        except ValueError as e:
            logger.warning(f"Error processing path '{base_path}': {e}. Skipping.")
            logger.debug(f"DEBUG: Caught ValueError for '{base_path}': {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred for path '{base_path}': {e}. Skipping.",
                         exc_info=True)
            logger.debug(f"DEBUG: Caught unexpected Exception for '{base_path}': {e}")

    logger.debug(f"DEBUG: Finished processing all base paths. 'all_trees' contains {len(all_trees)} DataFrames.")
    if not all_trees:
        logger.debug("DEBUG: 'all_trees' is empty. Returning empty DataFrame with schema.")
        return pl.DataFrame([], schema=PATHS_DF_EXPECTED_SCHEMA)

    logger.debug(f"DEBUG: 'all_trees' is not empty. Proceeding with pl.concat.")
    paths_df = pl.concat(all_trees, how="vertical_relaxed")
    logger.debug(f"DEBUG: After pl.concat, 'paths_df' shape: {paths_df.shape}, schema: {paths_df.schema}")

    paths_df = paths_df.with_columns(
        pl.when(pl.col("type") == "file")
        .then(
            # Coalesce the existing 'file_extension' (from _fetch_single_directory_tree)
            # with extracting from 'name' if the existing one is null or empty.
            # Convert to lowercase and fill nulls with empty string.
            pl.coalesce(
                pl.col("file_extension").str.to_lowercase().fill_null(""),
                pl.col("name").str.extract(r"\.([^.]+)$", 1).str.to_lowercase().fill_null("")
            )
        )
        .otherwise(pl.lit(None, dtype=pl.String))  # Folders have no extension, keep as None/null
        .alias("file_extension")  # Apply this back to the 'file_extension' column
    )

    paths_df = _aggregate_folder_sizes(paths_df)
    logger.debug(f"DEBUG: After _aggregate_folder_sizes, 'paths_df' shape: {paths_df.shape}, schema: {paths_df.schema}")

    paths_df = paths_df.with_columns(
        pl.col("size_bytes")
        .map_elements(lambda s: format_bytes_to_human_readable(s), return_dtype=pl.String)
        .alias("size_readable")
    )
    logger.debug(f"DEBUG: After adding 'size_readable', 'paths_df' shape: {paths_df.shape}, schema: {paths_df.schema}")

    logger.debug(f"DEBUG: Before final select. 'paths_df' columns: {paths_df.columns}")

    select_expressions = []
    current_columns = set(paths_df.columns)

    for col_name, col_dtype in PATHS_DF_EXPECTED_SCHEMA.items():
        if col_name in current_columns:
            # If column exists, cast it to the expected dtype
            select_expressions.append(pl.col(col_name).cast(col_dtype))
        else:
            # If column does not exist, create it with nulls and the expected dtype
            select_expressions.append(pl.lit(None, dtype=col_dtype).alias(col_name))

    # Perform the final select and cast in one go, which also enforces the order
    paths_df = paths_df.select(select_expressions)

    logger.debug(f"DEBUG: After final select, 'paths_df' shape: {paths_df.shape}, schema: {paths_df.schema}")

    logger.debug(f"--- DEBUG: Exiting build_paths_df successfully ---")

    return paths_df


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


def aggregate_processing_result(original_df: pl.DataFrame, processed_files: pl.DataFrame) -> pl.DataFrame:
    """
    Aggregates the processed file metadata back into the original DataFrame.
    """
    original_cols = set(original_df.columns)
    # Use 'name' for joining as it's the common identifier used in extract_metadata
    processed_full_dataframe = original_df.join(processed_files, on="path", how="left", suffix="_new")

    for col in processed_files.columns:
        if col == "name":
            continue
        new_col_name = f"{col}_new"
        if new_col_name in processed_full_dataframe.columns:  # Check if the new column exists after join
            if col in original_cols:
                # If the column already exists in original_df, update using coalesce:
                # take the new value when present, otherwise the original.
                processed_full_dataframe = processed_full_dataframe.with_columns(
                    pl.coalesce(pl.col(new_col_name), pl.col(col)).alias(col)
                ).drop(new_col_name)
            else:
                # If it's a completely new column, just rename the _new column
                processed_full_dataframe = processed_full_dataframe.rename({new_col_name: col})
    return processed_full_dataframe


def preprocess_files(df: pl.DataFrame) -> pl.DataFrame:
    """
    Preprocesses files by aggregating statistics from multiple imported paths
    and filtering based on selected folders.

    Parameters:
    - dataframe: The Polars DataFrame containing raw file system data.

    Returns:
    - A Polars DataFrame containing the aggregated and processed file information.
    """
    if df.is_empty() or "imported_path" not in df.columns:
        logger.warning("DataFrame is empty or missing 'imported_path' column. Skipping preprocessing.")
        return df

    # Get all unique values in the "imported_path" column
    unique_folders = df["imported_path"].unique().to_list()

    common_base = find_common_base(unique_folders)

    df = df.with_columns([
        pl.col("modification_date").dt.month().alias("modification_month"),
        pl.col("imported_path").str.replace(common_base, "", literal=True).alias("imported_path_short"),
    ])

    return df


def build_images_df(paths_df: pl.DataFrame, settings: Settings, widgets: List[PixelPatrolWidget]) -> \
        Optional[pl.DataFrame]:
    """
    Builds the images DataFrame by filtering, preprocessing, extracting metadata.

    Args:
        paths_df: The Polars DataFrame containing all file system paths.
        settings: The project settings, including selected file extensions.
        widgets: A list of PixelPatrolWidget instances used to determine required metadata columns.

    Returns:
        An Optional Polars DataFrame containing information about image files,
        or None if no image data is available after filtering and processing.
    """

    if paths_df is None or paths_df.is_empty():
        logger.warning("No paths DataFrame provided or it's empty. Cannot build images DataFrame.")
        return None

    # 1. Filter for files only
    dataframe_images = paths_df.filter(pl.col("type").eq("file"))

    # 2. Filter based on selected file extensions
    selected_file_extensions = settings.selected_file_extensions
    if selected_file_extensions:
        # Convert selected_file_extensions to lowercase for case-insensitive comparison
        lower_selected_ext = {ext.lower() for ext in selected_file_extensions}
        dataframe_images = dataframe_images.filter(
            pl.col("file_extension").str.to_lowercase().is_in(list(lower_selected_ext))
        )
        logger.info(
            f"Filtered images DataFrame for extensions: {lower_selected_ext}. Rows remaining: {dataframe_images.height}")
    else:
        logger.info("No specific file extensions selected. All files are considered for image processing.")

    if dataframe_images.is_empty():
        logger.warning("No image files found after filtering by type and extension. images_df will be None.")
        return None

    # 3. Preprocess files (add short names, modification period, etc.)
    # Note: `preprocess_files` now expects a `imported_path` column,
    # which is added by `build_paths_df` in the updated `processing.py`.
    dataframe_images = preprocess_files(dataframe_images)
    logger.info(f"Images DataFrame preprocessed. Columns: {dataframe_images.columns}")

    # 5. Get required columns from widgets
    # This function is now imported from pixel_patrol.utils.widget
    required_columns = get_required_columns(widgets)
    logger.info(f"Required columns for metadata extraction: {required_columns}")

    # 6. Extract metadata and process files
    if required_columns:
        logger.info("Extracting metadata for images...")

        metadata_list = [
            extract_image_metadata(Path(path_str), required_columns)
            for path_str in dataframe_images["path"].to_list()  # Convert series to list
        ]

        # Convert list of dicts to a DataFrame
        if metadata_list:
            # Filter out None/empty dicts first
            valid_metadata = [m for m in metadata_list if m]
            if valid_metadata:
                processed_metadata_df = pl.DataFrame(valid_metadata)

                # Ensure the 'path' column is present in processed_metadata_df for joining
                # Create a series with the original paths from dataframe_images
                paths_series = pl.Series("path", dataframe_images["path"].to_list())

                # Add 'path' column to processed_metadata_df if it's not already there.
                # If valid_metadata was empty or only had dicts without 'path', this ensures it exists.
                # It's safer to ensure the join key is always present in both DFs.
                if "path" not in processed_metadata_df.columns:
                    processed_metadata_df = processed_metadata_df.with_columns(paths_series)
                else:
                    # If 'path' is already there, ensure it's in the correct order for joining
                    # and that it perfectly matches the original dataframe_images paths.
                    # This might be an over-caution, but ensures robustness for the join.
                    processed_metadata_df = processed_metadata_df.with_columns(
                        paths_series.alias("path")  # Overwrite to ensure exact alignment
                    )

                # Aggregate processing result (join metadata back to the main DataFrame)
                # Join on 'path' as extract_image_metadata receives Path objects
                dataframe_images = aggregate_processing_result(dataframe_images, processed_metadata_df)
                logger.info("Metadata extraction and aggregation complete.")
            else:
                logger.warning("No valid metadata extracted for any images.")
        else:
            logger.warning("Metadata extraction yielded no results.")
    else:
        logger.info("No specific metadata columns required by widgets. Skipping metadata extraction.")

    return dataframe_images


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
