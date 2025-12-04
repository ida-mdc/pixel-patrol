from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)


def _serialize_ndarray_columns_dataframe(polars_df: pl.DataFrame) -> pl.DataFrame:
    """Convert ndarray columns to list form so Polars/Parquet can persist them."""
    for col in polars_df.columns:
        if polars_df[col].dtype == pl.Object:
            try:
                polars_df = polars_df.with_columns(
                    pl.col(col).map_elements(
                        lambda x: x.tolist() if isinstance(x, np.ndarray) else x,
                        return_dtype=pl.List(pl.Int64),
                    )
                )
            except Exception as exc:  # pragma: no cover - best-effort downgrade
                logger.warning(
                    "Project IO: Failed to serialize column '%s' to list. Dropping column. Error: %s",
                    col,
                    exc,
                )
                polars_df = polars_df.drop(col)
    return polars_df


def _deserialize_ndarray_columns_dataframe(polars_df: pl.DataFrame) -> pl.DataFrame:
    """Convert list columns that used to hold ndarrays back into ndarray objects."""
    for col in polars_df.columns:
        if polars_df[col].dtype == pl.List(pl.Int64):
            try:
                polars_df = polars_df.with_columns(
                    pl.col(col)
                    .map_elements(
                        lambda x: np.array(x) if isinstance(x, (list, pl.Series)) else x,
                        return_dtype=pl.Object,
                    )
                    .cast(pl.Object)
                )
            except Exception as exc:  # pragma: no cover - best-effort downgrade
                logger.warning(
                    "Project IO: Failed to deserialize column '%s' to ndarray. Dropping column. Error: %s",
                    col,
                    exc,
                )
                polars_df = polars_df.drop(col)
    return polars_df


def _write_dataframe_to_parquet(
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
