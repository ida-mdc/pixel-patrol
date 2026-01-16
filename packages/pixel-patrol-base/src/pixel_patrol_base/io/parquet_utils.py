from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)


def _serialize_ndarray_columns_dataframe(polars_df: pl.DataFrame) -> pl.DataFrame:
    """
    Serializes columns containing numpy ndarrays to lists of int64 for compatibility with Parquet.
    This is necessary because Polars does not support direct serialization of numpy ndarrays.
    Args:
        df: The Polars DataFrame to process.
    Returns:
        A Polars DataFrame with ndarray columns serialized to lists.
    """
    for col in polars_df.columns:
        if polars_df[col].dtype == pl.Object:
            try:
                # Attempt to convert ndarray columns to lists
                polars_df = polars_df.with_columns(
                    pl.col(col).map_elements(
                        lambda x: x.tolist() if isinstance(x, np.ndarray) else x,
                        return_dtype=pl.List(pl.Int64),
                    )
                )
                # logger.info(f"Project IO: Successfully serialized column '{col}' from ndarray to list.")
            except Exception as e:
                logger.warning(
                    f"Project IO: Failed to serialize column '{col}' to list. Error: {e}. This column will be excluded from the Parquet export."
                )
                polars_df = polars_df.drop(col)
    return polars_df


def _deserialize_ndarray_columns_dataframe(polars_df: pl.DataFrame) -> pl.DataFrame:
    """
    Deserializes columns containing lists of int64 back to numpy ndarrays.
    Args:
        polars_df: The Polars DataFrame to process.
    Returns:
        A Polars DataFrame with list columns deserialized to ndarrays.
    """
    # 1. Identify columns that need conversion
    target_cols = [
        col for col in polars_df.columns if polars_df[col].dtype == pl.List(pl.Int64)
    ]

    if not target_cols:
        return polars_df

    # 2. Build a list of expressions to apply all at once
    expressions = []
    for col in target_cols:
        expressions.append(
            pl.col(col)
            .map_elements(
                lambda x: np.array(x) if isinstance(x, (list, pl.Series)) else x,
                return_dtype=pl.Object,
            )
            .alias(col)  # Ensure we overwrite the existing column
        )

    # 3. Apply all transformations in a single pass
    try:
        polars_df = polars_df.with_columns(expressions)
        # logger.info(f"Project IO: Deserialized {len(target_cols)} columns to ndarray.")
    except Exception as e:
        # If the batch fails, you might need to fall back to the loop
        # to identify exactly which column failed, or log the generic error.
        logger.warning(f"Project IO: Batch deserialization failed. Error: {e}")

    return polars_df


def write_dataframe_to_parquet(
    df: Optional[pl.DataFrame],
    base_filename: str,
    target_dir: Path,
    *,
    compression: str | None = None,
) -> Optional[Path]:
    """Serialize the dataframe (if present) to Parquet under ``target_dir/base_filename``."""
    if df is None:
        return None

    target_dir.mkdir(parents=True, exist_ok=True)
    df = _serialize_ndarray_columns_dataframe(df)

    empty_struct_cols = [
        name for name, dtype in df.schema.items() if isinstance(dtype, pl.Struct) and not dtype.fields
    ]
    for col in empty_struct_cols:
        df = df.with_columns(pl.lit(None).alias(col))

    file_path = target_dir / base_filename
    data_name = file_path.stem
    try:
        df.write_parquet(file_path, compression=compression)
        return file_path
    except Exception as exc:
        logger.warning(
            "Project IO: Could not write %s data (%s) to directory %s: %s",
            data_name,
            base_filename,
            target_dir,
            exc,
        )
        return None


def read_dataframe_from_parquet(
    file_path: Path, src_archive: Path
) -> Optional[pl.DataFrame]:
    """Helper to read an optional Polars DataFrame from a Parquet file."""
    if not file_path.exists():
        return None
    data_name = file_path.stem
    try:
        df = pl.read_parquet(file_path)
        df = _deserialize_ndarray_columns_dataframe(df)
        return df
    except Exception as e:
        logger.warning(
            f"Project IO: Could not read {data_name} data from '{file_path.name}' "
            f"in archive '{src_archive.name}'. Data not loaded. Error: {e}"
        )
        return None
