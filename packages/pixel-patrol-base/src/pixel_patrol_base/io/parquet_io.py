"""
Project persistence via a single parquet file.
Provenance metadata (project name, flavor, authors, base_dir, paths, etc.)
is stored in the parquet footer — zero overhead on data reads.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Union, Tuple, Optional, Literal
import io

import polars as pl
import pyarrow.parquet as pq

from pixel_patrol_base.core.project_metadata import ProjectMetadata

logger = logging.getLogger(__name__)


def write_chunk(df: pl.DataFrame, path: Path, compression: Literal["lz4", "uncompressed", "snappy", "gzip", "brotli", "zstd"] = "zstd") -> Optional[Path]:
    """
    Write a single DataFrame chunk to parquet. Used for intermediate batch files.
    Handles empty struct columns which pyarrow/polars cannot serialize.
    Returns the path on success, None on failure.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    empty_struct_cols = [
        name for name, dtype in df.schema.items()
        if isinstance(dtype, pl.Struct) and not dtype.fields
    ]
    for col in empty_struct_cols:
        df = df.with_columns(pl.lit(None).alias(col))

    try:
        df.write_parquet(path, compression=compression)
        return path
    except Exception as exc:
        logger.warning("Parquet IO: Could not write chunk '%s': %s", path.name, exc)
        return None


def save_parquet(
    df: pl.DataFrame,
    dest: Path,
    metadata: ProjectMetadata,
) -> None:
    """
    Write records_df to a parquet file with provenance metadata in the footer.

    Args:
        df:       The records DataFrame to save.
        dest:     Destination path (will get .parquet suffix if missing).
        metadata: ProjectMetadata written to the footer (includes project_name).
    """
    dest = Path(dest)
    if dest.suffix.lower() != ".parquet":
        dest = dest.with_suffix(".parquet")

    dest.parent.mkdir(parents=True, exist_ok=True)

    footer_meta = metadata.to_parquet_meta()

    table = df.to_arrow()
    existing_meta = table.schema.metadata or {}
    encoded = {k.encode(): v.encode() for k, v in footer_meta.items()}
    table = table.replace_schema_metadata({**existing_meta, **encoded})
    pq.write_table(table, dest, compression="zstd")
    logger.info("Parquet IO: Saved '%s' to '%s'.", metadata.project_name, dest)


def load_parquet(src: Path) -> Tuple[pl.DataFrame, ProjectMetadata]:
    """
    Load a parquet file written by save_parquet.
    Reads the footer first (fast, no row data), then the full DataFrame.

    Args:
        src: Path to the parquet file.
    Returns:
        (records_df, metadata)
    """
    src = Path(src)
    if not src.exists():
        raise FileNotFoundError(f"Parquet file not found: {src}")
    if src.suffix.lower() != ".parquet":
        raise ValueError(f"Expected a .parquet file, got: {src}")

    try:
        raw_meta = {
            k.decode(): v.decode()
            for k, v in (pq.read_metadata(src).metadata or {}).items()
        }
    except Exception as e:
        raise ValueError(f"Could not read parquet metadata from '{src}': {e}") from e

    metadata = ProjectMetadata.from_parquet_meta(raw_meta)

    try:
        records_df = pl.read_parquet(src)
    except Exception as e:
        logger.warning("Parquet IO: Could not read DataFrame from '%s': %s", src, e)
        records_df = pl.DataFrame()

    logger.info(
        "Parquet IO: Loaded '%s' (flavor=%s, authors=%s) from '%s'.",
        metadata.project_name, metadata.flavor, metadata.authors, src,
    )
    return records_df, metadata


def resolve_report_source(
    source: Union["Project", Path],  # type: ignore[name-defined]
) -> Tuple[pl.DataFrame, ProjectMetadata]:
    """
    Resolve report data from either a live Project or a parquet file path.
    Returns (records_df, metadata).
    """
    if isinstance(source, Path):
        return load_parquet(source)

    # Live Project — use in-memory data directly
    from pixel_patrol_base.core.project import Project
    if isinstance(source, Project):
        return source.records_df, source.metadata

    raise TypeError(f"Expected Project or Path, got {type(source)}")


def to_parquet_bytes(df: pl.DataFrame, metadata: ProjectMetadata) -> bytes:
    """Serialise to in-memory bytes for browser download. No disk I/O."""
    buf = io.BytesIO()
    _write_with_metadata(df, metadata, buf)
    return buf.getvalue()


def _write_with_metadata(df: pl.DataFrame, metadata: ProjectMetadata, dest) -> None:
    """Write df to dest (Path or BytesIO) with metadata in the parquet footer."""
    table = df.to_arrow()
    existing = table.schema.metadata or {}
    encoded = {k.encode(): v.encode() for k, v in metadata.to_parquet_meta().items()}
    table = table.replace_schema_metadata({**existing, **encoded})
    pq.write_table(table, dest, compression="zstd")