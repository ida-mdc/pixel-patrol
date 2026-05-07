"""
Tests for parquet-based project persistence (save_parquet / load_parquet round-trip).

Replaces the old zip-based export/import tests.
"""
import numpy as np
import pytest
from pathlib import Path
import polars as pl
import logging

from pixel_patrol_base.core.project import Project
from pixel_patrol_base.core.project_metadata import ProjectMetadata
from pixel_patrol_base.core.processing_config import ProcessingConfig
from pixel_patrol_base.io.parquet_io import save_parquet, load_parquet
from pixel_patrol_base import api

logging.basicConfig(level=logging.INFO)


# --- Tests for save_parquet ---

def test_save_parquet_creates_file(tmp_path: Path):
    """save_parquet writes a .parquet file at the expected path."""
    df = pl.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    metadata = ProjectMetadata(project_name="test_save")
    dest = tmp_path / "output.parquet"

    save_parquet(df, dest, metadata)

    assert dest.exists()


def test_save_parquet_adds_suffix_if_missing(tmp_path: Path):
    """save_parquet adds .parquet suffix when missing."""
    df = pl.DataFrame({"a": [1]})
    metadata = ProjectMetadata(project_name="test_suffix")
    dest = tmp_path / "output"

    save_parquet(df, dest, metadata)

    assert (tmp_path / "output.parquet").exists()


def test_save_parquet_creates_parent_directories(tmp_path: Path):
    """save_parquet creates non-existent parent directories."""
    df = pl.DataFrame({"a": [1]})
    metadata = ProjectMetadata(project_name="test_parents")
    dest = tmp_path / "new_dir" / "sub_dir" / "output.parquet"

    save_parquet(df, dest, metadata)

    assert dest.exists()


# --- Tests for load_parquet ---

def test_load_parquet_round_trip(tmp_path: Path):
    """Data and metadata survive a save/load round-trip."""
    df = pl.DataFrame({"col1": [10, 20], "col2": ["a", "b"]})
    metadata = ProjectMetadata(
        project_name="round_trip",
        flavor="test_flavor",
        description="Authors: alice, bob",
    )
    dest = tmp_path / "round_trip.parquet"
    save_parquet(df, dest, metadata)

    loaded_df, loaded_meta = load_parquet(dest)

    assert loaded_df.shape == df.shape
    assert loaded_df.columns == df.columns
    assert loaded_df["col1"].to_list() == [10, 20]
    assert loaded_df["col2"].to_list() == ["a", "b"]
    assert loaded_meta.project_name == "round_trip"
    assert loaded_meta.flavor == "test_flavor"
    assert loaded_meta.description == "Authors: alice, bob"


def test_load_parquet_preserves_metadata_fields(tmp_path: Path):
    """All ProjectMetadata fields survive round-trip."""
    metadata = ProjectMetadata(
        project_name="full_meta",
        flavor="my_flavor",
        description="Authors: deborah",
        base_dir="/some/path",
        paths=["/some/path/a", "/some/path/b"],
    )
    df = pl.DataFrame({"x": [1]})
    dest = tmp_path / "meta_test.parquet"
    save_parquet(df, dest, metadata)

    _, loaded_meta = load_parquet(dest)

    assert loaded_meta.project_name == "full_meta"
    assert loaded_meta.flavor == "my_flavor"
    assert loaded_meta.description == "Authors: deborah"
    assert loaded_meta.base_dir == "/some/path"
    assert loaded_meta.paths == ["/some/path/a", "/some/path/b"]
    assert loaded_meta.version != ""  # auto-populated
    assert loaded_meta.created_at != ""


def test_load_parquet_nonexistent_file(tmp_path: Path):
    """load_parquet raises FileNotFoundError for missing files."""
    with pytest.raises(FileNotFoundError):
        load_parquet(tmp_path / "nonexistent.parquet")


def test_load_parquet_wrong_suffix(tmp_path: Path):
    """load_parquet raises ValueError for non-.parquet files."""
    bad_file = tmp_path / "data.csv"
    bad_file.write_text("a,b\n1,2")
    with pytest.raises(ValueError, match="Expected a .parquet file"):
        load_parquet(bad_file)


def test_load_parquet_corrupted_file(tmp_path: Path):
    """load_parquet raises ValueError for corrupted files."""
    bad_file = tmp_path / "bad.parquet"
    bad_file.write_bytes(b"THIS IS NOT PARQUET")
    with pytest.raises(ValueError, match="Could not read parquet metadata"):
        load_parquet(bad_file)

# --- Tests for full project save/load cycle ---

def test_project_process_saves_parquet(project_with_all_data: Project, tmp_path: Path):
    """process_records saves a parquet with metadata in output_dir."""
    # project_with_all_data already ran process_records via conftest
    # Check that a parquet exists in output_dir
    if project_with_all_data.output_path:
        assert project_with_all_data.output_path.exists()

        loaded_df, loaded_meta = load_parquet(project_with_all_data.output_path)
        assert loaded_df.height == project_with_all_data.records_df.height
        assert loaded_meta.project_name == project_with_all_data.name


def test_api_load_returns_df_and_metadata(tmp_path: Path):
    """api.load returns (records_df, metadata) tuple."""
    df = pl.DataFrame({"path": ["/a/b.png"], "size": [100]})
    metadata = ProjectMetadata(project_name="api_load_test", description="Authors: tester")
    dest = tmp_path / "api_test.parquet"
    save_parquet(df, dest, metadata)

    loaded_df, loaded_meta = api.load(dest)

    assert loaded_df.height == 1
    assert loaded_meta.project_name == "api_load_test"
    assert loaded_meta.description == "Authors: tester"


def test_full_save_load_cycle_with_float_columns(tmp_path: Path):
    """Float columns survive save/load with exact values."""
    df = pl.DataFrame({
        "path": ["/img1.tif", "/img2.tif"],
        "mean": [123.456, 789.012],
        "std": [1.5, 2.5],
    })
    metadata = ProjectMetadata(project_name="float_test")
    dest = tmp_path / "float_test.parquet"
    save_parquet(df, dest, metadata)

    loaded_df, _ = load_parquet(dest)

    for col in ["mean", "std"]:
        np.testing.assert_allclose(
            loaded_df[col].to_list(),
            df[col].to_list(),
            rtol=0, atol=0,
            err_msg=f"Column '{col}' values changed after save/load",
        )


def test_full_save_load_cycle_with_list_columns(tmp_path: Path):
    """List columns survive save/load."""
    df = pl.DataFrame({
        "path": ["/img.tif"],
        "histogram": [[1, 2, 3, 4, 5]],
    })
    metadata = ProjectMetadata(project_name="list_test")
    dest = tmp_path / "list_test.parquet"
    save_parquet(df, dest, metadata)

    loaded_df, _ = load_parquet(dest)

    assert loaded_df["histogram"].to_list() == [[1, 2, 3, 4, 5]]