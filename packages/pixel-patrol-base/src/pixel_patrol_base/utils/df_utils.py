import os
import polars as pl
from pixel_patrol_base.utils.path_utils import find_common_base
from pathlib import PurePath


def add_parent_level_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Compute parent0, parent1, ... from the (relative) path column."""
    if "path" not in df.columns or df.is_empty():
        return df
    df = df.with_columns(
        pl.col("path")
        .str.split(os.sep)
        .list.eval(pl.element().filter(pl.element().str.len_chars() > 0))
        .alias("_parts")
    ).with_columns(
        pl.col("_parts").list.slice(0, pl.col("_parts").list.len() - 1).alias("_parts")
    )
    max_depth = int(df["_parts"].list.len().max() or 0)
    if max_depth == 0:
        return df.drop("_parts")
    return df.with_columns([
        pl.col("_parts").list.get(i, null_on_oob=True).alias(f"parent{i}")
        for i in range(max_depth)
    ]).drop("_parts")



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
    common_base_name = PurePath(common_base).name or common_base

    df = df.with_columns([
        pl.col("modification_date").dt.month().alias("modification_month"),
        pl.lit(common_base_name).alias("common_base"),
    ])

    if df["imported_path"].n_unique() > 1:
        df = df.with_columns(
            pl.col("imported_path").str.replace(common_base, "", literal=True).alias("imported_path_short")
        )

    return df
