import polars as pl


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