import polars as pl
from polars import List as PolarsList
from pixel_patrol_base.utils.path_utils import find_common_base
from pixel_patrol_base.utils.utils import format_bytes_to_human_readable
import numpy as np
from typing import List, Dict, Any


def normalize_file_extension(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        pl.when(pl.col("type") == "file")
          .then(
              pl.coalesce(
                  pl.col("file_extension").str.to_lowercase().fill_null(""),
                  pl.col("name")
                    .str.extract(r"\.([^.]+)$", 1)
                    .str.to_lowercase()
                    .fill_null("")
              )
          )
          .otherwise(pl.lit(None))
          .alias("file_extension")
    )

def postprocess_basic_file_metadata_df(df: pl.DataFrame) -> pl.DataFrame:
    if df.is_empty():
        return df

    common_base = find_common_base(df["imported_path"].unique().to_list())

    df = df.with_columns([
        pl.col("modification_date").dt.month().alias("modification_month"),
        pl.col("imported_path").str.replace(common_base, "", literal=True).alias("imported_path_short"),
        pl.col("size_bytes").map_elements(format_bytes_to_human_readable).alias("size_readable"),
    ])

    return df

def get_polars_type(py_type):
    if issubclass(py_type, (np.integer,)): py_type = int
    if issubclass(py_type, (np.floating,)): py_type = float
    if issubclass(py_type, (np.bool_,)): py_type = bool
    if issubclass(py_type, bool): return pl.Boolean
    if issubclass(py_type, int): return pl.Int64
    if issubclass(py_type, float): return pl.Float64
    if issubclass(py_type, str): return pl.String
    if issubclass(py_type, (np.ndarray,)): return PolarsList(pl.Unknown)
    if issubclass(py_type, (dict,)):      return pl.Object
    if issubclass(py_type, (list, tuple)): return PolarsList(pl.Unknown)
    return pl.String

def _get_record_schema(rows: list[dict]) -> dict:
    dynamic_schema = {}
    for row in rows:
        for key, value in row.items():
            if value is None:
                continue

            current_type = dynamic_schema.get(key)
            proposed_type = get_polars_type(type(value))

            if current_type is None:
                dynamic_schema[key] = proposed_type
            elif current_type != proposed_type:
                numeric = {pl.Boolean, pl.Int64, pl.Float64}
                if current_type in numeric and proposed_type in numeric:
                    dynamic_schema[key] = pl.Float64 if (pl.Float64 in (current_type, proposed_type)) else pl.Int64
                elif str(current_type).startswith("List") or str(proposed_type).startswith("List"):
                    dynamic_schema[key] = PolarsList(pl.Unknown)
                else:
                    dynamic_schema[key] = pl.String
    return dynamic_schema


def rows_to_flexible_df(rows: List[Dict[str, Any]]) -> pl.DataFrame:
    """
    Builds a Polars DataFrame from a list of dictionaries, handling inconsistent
    columns and data types by building a robust schema first.
    """
    if not rows:
        return pl.DataFrame()

    all_cols = sorted(set().union(*[r.keys() for r in rows]))
    norm_rows: List[Dict[str, Any]] = [{c: r.get(c, None) for c in all_cols} for r in rows]

    for r in norm_rows:
        for k, v in r.items():
            if isinstance(v, (np.ndarray, tuple)):
                r[k] = v.tolist() if isinstance(v, np.ndarray) else list(v)
            elif isinstance(v, np.generic):
                r[k] = v.item()

    record_schema = _get_record_schema(norm_rows)

    list_cols: list[str] = [k for k, dt in record_schema.items() if str(dt).startswith("List")]
    for r in norm_rows:
        for k in list_cols:
            v = r.get(k)
            if v is None or isinstance(v, list):
                continue
            assert v is not None  # help type checker
            r[k] = [v]

    return pl.DataFrame(norm_rows, schema_overrides=record_schema)