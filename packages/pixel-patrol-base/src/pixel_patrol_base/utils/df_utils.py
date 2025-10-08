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
