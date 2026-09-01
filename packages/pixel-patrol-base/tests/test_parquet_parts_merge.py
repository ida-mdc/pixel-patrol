"""Tests for save_parquet_from_parts merging parts with DIFFERENT columns.

This is exactly the shape produced when one image's row has metric columns and
another image's row doesn't (e.g. every processor failed on it - see
test_processor_failure_isolation.py / the _rollup always-emit-row fix). A bug in
this schema-unification path would silently corrupt or drop data rather than error,
so it's worth pinning down directly rather than only exercising it indirectly.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from pixel_patrol_base.core.processing import save_parquet_from_parts
from pixel_patrol_base.core.processing_config import ProcessingConfig
from pixel_patrol_base.core.project import Project


@pytest.fixture
def metadata(tmp_path: Path):
    (tmp_path / "images").mkdir()
    proj = Project("test", tmp_path / "images", output_path=tmp_path / "out.parquet")
    return ProcessingConfig().metadata.populate_from_project(proj)


def test_missing_column_in_one_part_is_null_filled(tmp_path, metadata):
    """Part B lacks 'mean_intensity' entirely (the all-processors-failed row shape)."""
    part_a = tmp_path / "part_0000.parquet"
    part_b = tmp_path / "part_0001.parquet"
    pl.DataFrame({"file_path": ["/a.tif"], "num_pixels": [4096], "mean_intensity": [12.5]}).write_parquet(part_a)
    pl.DataFrame({"file_path": ["/b.tif"], "num_pixels": [0]}).write_parquet(part_b)

    dest = tmp_path / "merged.parquet"
    n_rows = save_parquet_from_parts([part_a, part_b], dest, metadata)

    assert n_rows == 2
    result = pl.read_parquet(dest).sort("file_path")
    assert result.columns.__contains__("mean_intensity")
    assert result["mean_intensity"].to_list() == [12.5, None]
    assert result["num_pixels"].to_list() == [4096, 0]


def test_column_missing_from_every_part_is_dropped(tmp_path, metadata):
    """A column that's all-null across every part gets dropped, not kept as dead weight."""
    part_a = tmp_path / "part_0000.parquet"
    part_b = tmp_path / "part_0001.parquet"
    pl.DataFrame({"file_path": ["/a.tif"], "thumbnail": pl.Series([None], dtype=pl.Binary)}).write_parquet(part_a)
    pl.DataFrame({"file_path": ["/b.tif"], "thumbnail": pl.Series([None], dtype=pl.Binary)}).write_parquet(part_b)

    dest = tmp_path / "merged.parquet"
    save_parquet_from_parts([part_a, part_b], dest, metadata)

    result = pl.read_parquet(dest)
    assert "thumbnail" not in result.columns
    assert "file_path" in result.columns


def test_int_column_shrinks_to_narrowest_type_that_fits_across_all_parts(tmp_path, metadata):
    part_a = tmp_path / "part_0000.parquet"
    part_b = tmp_path / "part_0001.parquet"
    pl.DataFrame({"file_path": ["/a.tif"], "num_pixels": [10]}).write_parquet(part_a)
    pl.DataFrame({"file_path": ["/b.tif"], "num_pixels": [300]}).write_parquet(part_b)  # exceeds int8, fits int16

    dest = tmp_path / "merged.parquet"
    save_parquet_from_parts([part_a, part_b], dest, metadata)

    result = pl.read_parquet(dest).sort("file_path")
    assert result.schema["num_pixels"] == pl.Int16
    assert result["num_pixels"].to_list() == [10, 300]


def test_many_parts_with_staggered_columns_all_merge_without_error(tmp_path, metadata):
    """3 parts, no two sharing the exact same column set - stresses the general case,
    not just a clean 2-part all-fail/all-succeed split."""
    parts = []
    for i, cols in enumerate([
        {"file_path": ["/a.tif"], "num_pixels": [10], "mean_intensity": [1.0]},
        {"file_path": ["/b.tif"], "num_pixels": [0]},
        {"file_path": ["/c.tif"], "num_pixels": [20], "histogram_min": [0.0]},
    ]):
        p = tmp_path / f"part_{i:04d}.parquet"
        pl.DataFrame(cols).write_parquet(p)
        parts.append(p)

    dest = tmp_path / "merged.parquet"
    n_rows = save_parquet_from_parts(parts, dest, metadata)

    assert n_rows == 3
    result = pl.read_parquet(dest).sort("file_path")
    assert set(result.columns) >= {"file_path", "num_pixels", "mean_intensity", "histogram_min"}
    assert result["mean_intensity"].to_list() == [1.0, None, None]
    assert result["histogram_min"].to_list() == [None, None, 0.0]
