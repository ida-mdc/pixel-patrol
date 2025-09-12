import logging
import os
from pathlib import Path
from typing import List, Optional, Dict, Set, Tuple

import polars as pl

import time

from pixel_patrol_base.core.contracts import PixelPatrolLoader, PixelPatrolProcessor
from pixel_patrol_base.core.feature_schema import merge_output_schemas, coerce_row_types
from pixel_patrol_base.core.file_system import walk_filesystem
from pixel_patrol_base.plugin_registry import discover_processor_plugins
from pixel_patrol_base.utils.df_utils import normalize_file_extension, postprocess_basic_file_metadata_df
from pixel_patrol_base.core.specs import is_artifact_matching_processor


logger = logging.getLogger(__name__)


PATHS_DF_EXPECTED_SCHEMA = {  # TODO: delete or rename - as paths_df is retired.
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


def _build_deep_artifact_df(paths: List[Path], loader_instance: PixelPatrolLoader) -> pl.DataFrame:
    """Loop over paths, get_all_artifact_properties, return DataFrame (may be empty).
    Optimized to minimize Python loop overhead where possible.
    """
    processors = discover_processor_plugins()

    static, patterns = merge_output_schemas(processors)

    rows = []

    for p in paths:
        artifact_dict = get_all_artifact_properties(p, loader_instance, processors)
        if artifact_dict:
            rows.append({"path": str(p), **artifact_dict})

    # 1) coerce types row-wise based on declared schema/patterns
    rows = [coerce_row_types(r, static, patterns) for r in rows]

    # 2) make columns consistent (fill missing keys with None)
    all_cols = sorted(set().union(*[r.keys() for r in rows]))
    rows = [{c: r.get(c, None) for c in all_cols} for r in rows]

    return pl.DataFrame(rows)


def get_all_artifact_properties(file_path: Path, loader: PixelPatrolLoader, processors: List[PixelPatrolProcessor]) -> Dict:
    start_total_time = time.monotonic()

    if not file_path.exists():
        logger.warning(f"File not found: '{file_path}'. Cannot extract metadata.")
        return {}

    extracted_properties = {}

    logger.info(f"Attempting to load '{file_path}' with loader: {loader.NAME}")

    start_load_time = time.monotonic()
    try:
        art = loader.load(str(file_path))
        metadata = dict(art.meta)
    except Exception as e:
        logger.info(f"Loader '{loader.NAME}' failed with exception, skipping: {e}")
        return {}

    load_duration = time.monotonic() - start_load_time
    logger.info(f"Loading with '{loader.NAME}' took {load_duration:.4f} seconds.")

    # Always process using Artifact; processors opt-in via INPUT spec
    extracted_properties.update(metadata)
    for P in processors:
        if not is_artifact_matching_processor(art, P.INPUT):
            continue
        out = P.run(art)
        if isinstance(out, dict):
            extracted_properties.update(out)
        else:
            art = out  # chainable: processors may transform the artifact
            extracted_properties.update(art.meta)

    total_duration = time.monotonic() - start_total_time
    logger.info(f"Successfully loaded and processed '{file_path}'. Total time: {total_duration} seconds.")
    return extracted_properties



def build_artifacts_df(
    bases: List[Path],
    selected_extensions: Set[str] | str,
    loader: Optional[PixelPatrolLoader],
) -> Optional[pl.DataFrame]:

    basic = _build_basic_file_df(bases, loader=loader, accepted_extensions=selected_extensions)
    if loader is None or basic is None: return basic

    deep = _build_deep_artifact_df([Path(p) for p in basic["path"].to_list()], loader)

    return basic.join(deep, on="path", how="left")


def _build_basic_file_df(bases, loader, accepted_extensions):
    basic = walk_filesystem(bases, loader=loader, accepted_extensions=accepted_extensions)
    if basic.is_empty(): return None
    basic = postprocess_basic_file_metadata_df(normalize_file_extension(basic))

    return basic


# TODO: delete or rename as paths_df is retired
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
