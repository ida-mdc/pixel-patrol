import pytest
import polars as pl
from pathlib import Path
from datetime import datetime, timezone
import os

from pixel_patrol.core.file_system import _fetch_single_directory_tree, _aggregate_folder_sizes
from pixel_patrol.core.processing import PATHS_DF_EXPECTED_SCHEMA  # To validate schemas


# --- Fixtures for _fetch_single_directory_tree tests ---

@pytest.fixture
def complex_temp_dir(tmp_path: Path) -> Path:
    """
    Creates a complex temporary directory structure with files for testing _fetch_single_directory_tree.
    Structure:
    tmp_path/
    ├── file1.txt (size: 10)
    ├── subdir_a/
    │   ├── fileA.jpg (size: 20)
    │   └── subdir_aa/
    │       └── fileAA.csv (size: 30)
    └── subdir_b/
        └── fileB.png (size: 40)
    """
    root = tmp_path / "complex_test_root"
    root.mkdir()

    # Create files with specific content to control size_bytes
    (root / "file1.txt").write_bytes(b'a' * 10)
    subdir_a = root / "subdir_a"
    subdir_a.mkdir()
    (subdir_a / "fileA.jpg").write_bytes(b'b' * 20)
    subdir_aa = subdir_a / "subdir_aa"
    subdir_aa.mkdir()
    (subdir_aa / "fileAA.csv").write_bytes(b'c' * 30)
    subdir_b = root / "subdir_b"
    subdir_b.mkdir()
    (subdir_b / "fileB.png").write_bytes(b'd' * 40)

    # Use a fixed modification time for deterministic tests
    fixed_timestamp = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc).timestamp()

    # Set mtime for all created paths
    for p in root.rglob('*'):
        os.utime(p, (fixed_timestamp, fixed_timestamp))

    return root


@pytest.fixture
def empty_temp_dir(tmp_path: Path) -> Path:
    """Creates an empty temporary directory."""
    empty_dir = tmp_path / "empty_dir"
    empty_dir.mkdir()
    return empty_dir


@pytest.fixture
def single_file_dir(tmp_path: Path) -> Path:
    """Creates a temporary directory with a single file."""
    single_file_root = tmp_path / "single_file_root"
    single_file_root.mkdir()
    (single_file_root / "single_file.txt").write_text("content")  # Small arbitrary size
    return single_file_root


# --- Tests for _fetch_single_directory_tree ---

def test_fetch_single_directory_tree_basic(complex_temp_dir: Path):
    """
    Tests _fetch_single_directory_tree with a basic directory structure.
    Verifies column types, content, depth, and parent relationships.
    """
    df = _fetch_single_directory_tree(complex_temp_dir)

    assert isinstance(df, pl.DataFrame)
    assert not df.is_empty()

    # Verify expected columns and their types
    expected_initial_schema = {
        "path": pl.String,
        "name": pl.String,
        "type": pl.String,
        "parent": pl.String,
        "depth": pl.Int64,
        "size_bytes": pl.Int64,
        "modification_date": pl.Datetime(time_unit="us", time_zone=None),  # Stored as UTC, but no tzinfo
        "file_extension": pl.String,
    }
    assert df.schema == expected_initial_schema

    # Convert to dictionary for easier assertion
    df_dict = df.sort("path").to_dicts()

    # Define expected data based on complex_temp_dir fixture
    expected_data = [
        # Root directory itself - parent is None as it's the root of the scanned tree
        {"path": str(complex_temp_dir), "name": "complex_test_root", "type": "folder", "parent": None, "depth": 0,
         "size_bytes": 0, "file_extension": None},
        {"path": str(complex_temp_dir / "file1.txt"), "name": "file1.txt", "type": "file",
         "parent": str(complex_temp_dir), "depth": 1, "size_bytes": 10, "file_extension": "txt"},
        {"path": str(complex_temp_dir / "subdir_a"), "name": "subdir_a", "type": "folder",
         "parent": str(complex_temp_dir), "depth": 1, "size_bytes": 0, "file_extension": None},
        {"path": str(complex_temp_dir / "subdir_a" / "fileA.jpg"), "name": "fileA.jpg", "type": "file",
         "parent": str(complex_temp_dir / "subdir_a"), "depth": 2, "size_bytes": 20, "file_extension": "jpg"},
        {"path": str(complex_temp_dir / "subdir_a" / "subdir_aa"), "name": "subdir_aa", "type": "folder",
         "parent": str(complex_temp_dir / "subdir_a"), "depth": 2, "size_bytes": 0, "file_extension": None},
        {"path": str(complex_temp_dir / "subdir_a" / "subdir_aa" / "fileAA.csv"), "name": "fileAA.csv", "type": "file",
         "parent": str(complex_temp_dir / "subdir_a" / "subdir_aa"), "depth": 3, "size_bytes": 30,
         "file_extension": "csv"},
        {"path": str(complex_temp_dir / "subdir_b"), "name": "subdir_b", "type": "folder",
         "parent": str(complex_temp_dir), "depth": 1, "size_bytes": 0, "file_extension": None},
        {"path": str(complex_temp_dir / "subdir_b" / "fileB.png"), "name": "fileB.png", "type": "file",
         "parent": str(complex_temp_dir / "subdir_b"), "depth": 2, "size_bytes": 40, "file_extension": "png"},
    ]

    # Compare relevant fields, excluding modification_date as it's dynamic unless mocked globally
    # It's better to verify that it exists and is of the correct type.
    for i, expected_row in enumerate(expected_data):
        actual_row = df_dict[i]
        for key, expected_value in expected_row.items():
            assert actual_row[key] == expected_value, f"Mismatch in row {i}, key '{key}'"
        assert isinstance(actual_row["modification_date"], datetime)


def test_fetch_single_directory_tree_empty_dir(empty_temp_dir: Path):
    """Tests _fetch_single_directory_tree with an empty directory."""
    df = _fetch_single_directory_tree(empty_temp_dir)

    assert isinstance(df, pl.DataFrame)
    # An empty directory should still yield one row for the directory itself
    assert len(df) == 1
    assert df["path"][0] == str(empty_temp_dir)
    assert df["type"][0] == "folder"
    assert df["size_bytes"][0] == 0
    assert df["depth"][0] == 0
    assert df["parent"][0] is None  # Root of the scanned tree


def test_fetch_single_directory_tree_single_file_dir(single_file_dir: Path):
    """Tests _fetch_single_directory_tree with a directory containing only one file."""
    df = _fetch_single_directory_tree(single_file_dir)

    assert isinstance(df, pl.DataFrame)
    assert len(df) == 2  # One for the folder, one for the file

    df_dict = df.sort("path").to_dicts()

    expected_data = [
        # Root directory itself - parent is None
        {"path": str(single_file_dir), "name": "single_file_root", "type": "folder", "parent": None, "depth": 0,
         "size_bytes": 0, "file_extension": None},
        {"path": str(single_file_dir / "single_file.txt"), "name": "single_file.txt", "type": "file",
         "parent": str(single_file_dir), "depth": 1, "size_bytes": len("content".encode('utf-8')),
         "file_extension": "txt"},
    ]

    for i, expected_row in enumerate(expected_data):
        actual_row = df_dict[i]
        for key, expected_value in expected_row.items():
            assert actual_row[key] == expected_value, f"Mismatch in row {i}, key '{key}'"
        assert isinstance(actual_row["modification_date"], datetime)


def test_fetch_single_directory_tree_not_a_directory(tmp_path: Path):
    """Tests _fetch_single_directory_tree raises ValueError if path is not a directory."""
    not_a_dir = tmp_path / "not_a_dir.txt"
    not_a_dir.touch()  # Create it as a file

    with pytest.raises(ValueError, match=f"The path '{not_a_dir}' is not a valid directory."):
        _fetch_single_directory_tree(not_a_dir)


def test_fetch_single_directory_tree_non_existent_path(tmp_path: Path):
    """Tests _fetch_single_directory_tree raises ValueError if path does not exist."""
    non_existent = tmp_path / "i_do_not_exist"

    with pytest.raises(ValueError, match=f"The path '{non_existent}' is not a valid directory."):
        _fetch_single_directory_tree(non_existent)


# --- Tests for _aggregate_folder_sizes ---

@pytest.fixture
def sample_flat_df() -> pl.DataFrame:
    """Provides a flat DataFrame with only files, no folders."""
    return pl.DataFrame({
        "path": ["/a/file1.txt", "/a/file2.jpg", "/b/file3.csv"],
        "name": ["file1.txt", "file2.jpg", "file3.csv"],
        "type": ["file", "file", "file"],
        "parent": ["/a", "/a", "/b"],
        "depth": [1, 1, 1],
        "size_bytes": [100, 200, 50],
        "modification_date": [datetime.now()] * 3,
        "file_extension": ["txt", "jpg", "csv"],
    })


@pytest.fixture
def sample_simple_nested_df() -> pl.DataFrame:
    """Provides a DataFrame for a simple nested structure."""
    return pl.DataFrame({
        "path": ["/root", "/root/file1.txt", "/root/subdir", "/root/subdir/file2.txt"],
        "name": ["root", "file1.txt", "subdir", "file2.txt"],
        "type": ["folder", "file", "folder", "file"],
        "parent": [None, "/root", "/root", "/root/subdir"],
        "depth": [0, 1, 1, 2],
        "size_bytes": [0, 100, 0, 50],  # Initial folder sizes are 0
        "modification_date": [datetime.now()] * 4,
        "file_extension": [None, "txt", None, "txt"],
    })


@pytest.fixture
def sample_multiple_roots_df() -> pl.DataFrame:
    """Provides a DataFrame for multiple independent root folders."""
    return pl.DataFrame({
        "path": ["/rootA", "/rootA/fileA.txt", "/rootB", "/rootB/fileB.txt"],
        "name": ["rootA", "fileA.txt", "rootB", "fileB.txt"],
        "type": ["folder", "file", "folder", "file"],
        "parent": [None, "/rootA", None, "/rootB"],
        "depth": [0, 1, 0, 1],
        "size_bytes": [0, 10, 0, 20],
        "modification_date": [datetime.now()] * 4,
        "file_extension": [None, "txt", None, "txt"],
    })


@pytest.fixture
def sample_deep_nested_df() -> pl.DataFrame:
    """Provides a DataFrame for a deeply nested structure."""
    return pl.DataFrame({
        "path": ["/a", "/a/b", "/a/b/c", "/a/b/c/file.txt"],
        "name": ["a", "b", "c", "file.txt"],
        "type": ["folder", "folder", "folder", "file"],
        "parent": [None, "/a", "/a/b", "/a/b/c"],
        "depth": [0, 1, 2, 3],
        "size_bytes": [0, 0, 0, 75],
        "modification_date": [datetime.now()] * 4,
        "file_extension": [None, None, None, "txt"],
    })


@pytest.fixture
def sample_empty_folders_df() -> pl.DataFrame:
    """Provides a DataFrame with empty folders and files elsewhere."""
    return pl.DataFrame({
        "path": ["/a", "/a/empty_subdir", "/a/file.txt", "/b", "/b/another_empty_dir"],
        "name": ["a", "empty_subdir", "file.txt", "b", "another_empty_dir"],
        "type": ["folder", "folder", "file", "folder", "folder"],
        "parent": [None, "/a", "/a", None, "/b"],
        "depth": [0, 1, 1, 0, 1],
        "size_bytes": [0, 0, 100, 0, 0],
        "modification_date": [datetime.now()] * 5,
        "file_extension": [None, None, "txt", None, None],
    })


def test_aggregate_folder_sizes_flat(sample_flat_df: pl.DataFrame):
    """Tests aggregation on a flat DataFrame with only files."""
    aggregated_df = _aggregate_folder_sizes(sample_flat_df)
    assert aggregated_df["size_bytes"].to_list() == [100, 200, 50]  # Sizes should be unchanged for files


def test_aggregate_folder_sizes_simple_nested(sample_simple_nested_df: pl.DataFrame):
    """Tests aggregation on a simply nested directory structure."""
    aggregated_df = _aggregate_folder_sizes(sample_simple_nested_df).sort("path")

    # Expected sizes:
    # /root/subdir/file2.txt: 50
    # /root/subdir: 50 (contains file2.txt)
    # /root/file1.txt: 100
    # /root: 150 (contains file1.txt + subdir's aggregated size)
    expected_sizes_bytes = [150, 100, 50, 50]  # Order after sort: /root, /root/file1, /root/subdir, /root/subdir/file2

    # Sort the actual DF by path to match expected_sizes_bytes order
    actual_sizes = aggregated_df.select(pl.col("path"), pl.col("size_bytes")).to_dicts()

    # Re-order expected data based on path for direct comparison
    expected_data_sorted = {
        "/root": 150,
        "/root/file1.txt": 100,
        "/root/subdir": 50,
        "/root/subdir/file2.txt": 50,
    }

    for row in actual_sizes:
        assert row["size_bytes"] == expected_data_sorted[row["path"]]


def test_aggregate_folder_sizes_multiple_roots(sample_multiple_roots_df: pl.DataFrame):
    """Tests aggregation when there are multiple independent root folders."""
    aggregated_df = _aggregate_folder_sizes(sample_multiple_roots_df).sort("path")

    # Expected sizes:
    # /rootA/fileA.txt: 10
    # /rootA: 10
    # /rootB/fileB.txt: 20
    # /rootB: 20
    expected_data_sorted = {
        "/rootA": 10,
        "/rootA/fileA.txt": 10,
        "/rootB": 20,
        "/rootB/fileB.txt": 20,
    }

    actual_sizes = aggregated_df.select(pl.col("path"), pl.col("size_bytes")).to_dicts()

    for row in actual_sizes:
        assert row["size_bytes"] == expected_data_sorted[row["path"]]


def test_aggregate_folder_sizes_deep_nested(sample_deep_nested_df: pl.DataFrame):
    """Tests aggregation on a deeply nested structure."""
    aggregated_df = _aggregate_folder_sizes(sample_deep_nested_df).sort("path")

    # Expected sizes:
    # /a/b/c/file.txt: 75
    # /a/b/c: 75
    # /a/b: 75
    # /a: 75
    expected_data_sorted = {
        "/a": 75,
        "/a/b": 75,
        "/a/b/c": 75,
        "/a/b/c/file.txt": 75,
    }

    actual_sizes = aggregated_df.select(pl.col("path"), pl.col("size_bytes")).to_dicts()

    for row in actual_sizes:
        assert row["size_bytes"] == expected_data_sorted[row["path"]]


def test_aggregate_folder_sizes_empty_folders(sample_empty_folders_df: pl.DataFrame):
    """Tests aggregation with empty folders (should remain 0 size)."""
    aggregated_df = _aggregate_folder_sizes(sample_empty_folders_df).sort("path")

    expected_data_sorted = {
        "/a": 100,  # Contains file.txt
        "/a/empty_subdir": 0,  # Empty
        "/a/file.txt": 100,
        "/b": 0,  # Contains only empty_dir
        "/b/another_empty_dir": 0,  # Empty
    }

    actual_sizes = aggregated_df.select(pl.col("path"), pl.col("size_bytes")).to_dicts()

    for row in actual_sizes:
        assert row["size_bytes"] == expected_data_sorted[row["path"]]


def test_aggregate_folder_sizes_empty_input_df():
    """Tests aggregation with an empty input DataFrame."""
    empty_df = pl.DataFrame([], schema=PATHS_DF_EXPECTED_SCHEMA)  # Use correct schema
    aggregated_df = _aggregate_folder_sizes(empty_df)
    assert aggregated_df.is_empty()
    assert aggregated_df.schema == empty_df.schema  # Schema should be preserved