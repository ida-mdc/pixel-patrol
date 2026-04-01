from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


def _serialize_ndarray_columns_dataframe(polars_df: pl.DataFrame) -> pl.DataFrame:
    """
    Serializes numpy-ndarray columns (stored as pl.Object) to raw bytes (pl.Binary).

    The call site that reads back the bytes is responsible for knowing the dtype
    (e.g. int32 for histogram counts, uint8 for thumbnails).
    """
    for col in polars_df.columns:
        if polars_df[col].dtype != pl.Object:
            continue
        non_null = polars_df[col].drop_nulls()
        if non_null.is_empty():
            continue
        sample = non_null[0]
        if not isinstance(sample, np.ndarray):
            continue
        dtype_str = str(sample.dtype)
        try:
            polars_df = polars_df.with_columns(
                pl.col(col).map_elements(
                    lambda x, d=dtype_str: (
                        x.astype(d).tobytes() if isinstance(x, np.ndarray) else None
                    ),
                    return_dtype=pl.Binary,
                )
            )
        except Exception as e:
            logger.warning(
                "Project IO: Failed to serialize column '%s' to bytes. "
                "Error: %s. Column will be excluded from Parquet export.",
                col, e,
            )
            polars_df = polars_df.drop(col)
    return polars_df



def _promote_uniform_binary_to_fixed(arrow_table: pa.Table) -> pa.Table:
    """
    Promotes binary columns whose values all share the same byte length to
    FIXED_LEN_BYTE_ARRAY (``pa.binary(N)``).

    Parquet stores FIXED_LEN_BYTE_ARRAY pages contiguously, which allows
    column-projection readers (e.g. hyparquet) to scan them ~6× faster than
    variable-length BYTE_ARRAY pages.  Applies to both serialized ndarray
    columns and any other uniform-length binary column (e.g. thumbnails).
    """
    field_list = list(arrow_table.schema)
    col_list = list(arrow_table.columns)

    for i, (field, col) in enumerate(zip(field_list, col_list)):
        if not (pa.types.is_binary(field.type) or pa.types.is_large_binary(field.type)):
            continue
        flat = col.combine_chunks()
        valid = flat.filter(pc.is_valid(flat))
        if len(valid) == 0:
            continue
        byte_lengths = pc.binary_length(valid)
        min_len = pc.min(byte_lengths).as_py()
        max_len = pc.max(byte_lengths).as_py()
        if min_len != max_len or min_len is None or min_len == 0:
            continue
        N = min_len
        try:
            fixed = pa.array(flat.to_pylist(), type=pa.binary(N))
            field_list[i] = pa.field(field.name, pa.binary(N))
            col_list[i] = fixed
        except Exception as e:
            logger.warning(
                "Project IO: Could not promote '%s' to fixed_size_binary(%d): %s",
                field.name, N, e,
            )

    new_schema = pa.schema(field_list, metadata=arrow_table.schema.metadata)
    return pa.table(
        dict(zip(arrow_table.schema.names, col_list)),
        schema=new_schema,
    )


def write_dataframe_to_parquet(
    df: Optional[pl.DataFrame],
    base_filename: str,
    target_dir: Path,
    *,
    compression: str | None = None,
) -> Optional[Path]:
    """Serialize the dataframe (if present) to Parquet under ``target_dir/base_filename``.

    Numpy ndarray columns are serialized to raw bytes and written as
    FIXED_LEN_BYTE_ARRAY.  Uniform-length binary columns (e.g. thumbnails)
    are promoted to FIXED_LEN_BYTE_ARRAY automatically.
    """
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
    try:
        arrow_table = df.to_arrow()
        arrow_table = _promote_uniform_binary_to_fixed(arrow_table)
        pq.write_table(arrow_table, file_path, compression=compression or "zstd")
        return file_path
    except Exception as exc:
        logger.warning(
            "Project IO: Could not write %s data (%s) to directory %s: %s",
            file_path.stem, base_filename, target_dir, exc,
        )
        return None


def read_dataframe_from_parquet(
    file_path: Path, src_archive: Path
) -> Optional[pl.DataFrame]:
    """Helper to read an optional Polars DataFrame from a Parquet file."""
    if not file_path.exists():
        return None
    try:
        return pl.read_parquet(file_path)
    except Exception as e:
        logger.warning(
            "Project IO: Could not read %s data from '%s' in archive '%s'. "
            "Data not loaded. Error: %s",
            file_path.stem, file_path.name, src_archive.name, e,
        )
        return None
