import pytest
import polars as pl
from pathlib import Path
from datetime import datetime, timezone
import os


from pixel_patrol_base.core.file_system import walk_filesystem, _aggregate_folder_sizes
from pixel_patrol_base.core.processing import PATHS_DF_EXPECTED_SCHEMA
from pixel_patrol_base.utils.utils import format_bytes_to_human_readable

# --- Fixtures for _fetch_single_directory_tree ---

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

def _assert_directory_tree_df(df: pl.DataFrame, expected_data: list[dict], imported_path: str):
    """
    Helper function to assert the content and schema of the DataFrame returned by _fetch_single_directory_tree.
    Handles dynamic values like modification_date.
    """
    assert isinstance(df, pl.DataFrame)
    assert not df.is_empty()

    expected_schema = {
        "path": pl.String,
        "name": pl.String,
        "type": pl.String,
        "parent": pl.String, # This should be pl.String or pl.Null if parent can be None
        "depth": pl.Int64,
        "size_bytes": pl.Int64,
        "modification_date": pl.Datetime(time_unit="us", time_zone=None),
        "file_extension": pl.String, # This should be pl.String or pl.Null if extension can be None
        "imported_path": pl.String,
    }

    # Adjusting for the possibility of Null types for 'parent' and 'file_extension'
    # Polars might infer Null for columns that are entirely None in a small dataset,
    # or String if there's a mix. To be robust, we check the underlying type.
    actual_schema = df.schema

    for col, expected_type in expected_schema.items():
        assert col in actual_schema, f"Column '{col}' missing from actual schema"
        # For 'parent' and 'file_extension', allow either String or Null
        if col in ["parent", "file_extension"]:
            assert actual_schema[col] == expected_type or actual_schema[col] == pl.Null, \
                f"Mismatch in schema for column '{col}': Expected {expected_type} or {pl.Null}, got {actual_schema[col]}"
        else:
            assert actual_schema[col] == expected_type, \
                f"Mismatch in schema for column '{col}': Expected {expected_type}, got {actual_schema[col]}"


    df_dict = df.sort("path").to_dicts()
    expected_data.sort(key=lambda x: x['path'])

    assert len(df_dict) == len(expected_data), "Number of rows mismatch"

    for i, expected_row in enumerate(expected_data):
        actual_row = df_dict[i]
        for key, expected_value in expected_row.items():
            if key == "modification_date":
                assert isinstance(actual_row[key], datetime), f"Mismatch in row {i}, key '{key}': type"
            elif key == "imported_path":
                assert actual_row[key] == imported_path, f"Mismatch in row {i}, key '{key}'"
            else:
                assert actual_row[key] == expected_value, f"Mismatch in row {i}, key '{key}'"

def test_fetch_single_directory_tree_complex_structure(complex_temp_dir: Path):
    """
    Tests _fetch_single_directory_tree with a complex directory structure.
    Verifies column types, content, depth, parent, and imported_path.
    """
    df = walk_filesystem([complex_temp_dir], accepted_extensions="all")
    base_imported_path = str(complex_temp_dir)

    expected_data = [
        {"path": str(complex_temp_dir / "file1.txt"), "name": "file1.txt", "type": "file",
          "parent": str(complex_temp_dir), "depth": 1, "size_bytes": 10, "file_extension": "txt",
          "imported_path": base_imported_path},
        {"path": str(complex_temp_dir / "subdir_a" / "fileA.jpg"), "name": "fileA.jpg", "type": "file",
          "parent": str(complex_temp_dir / "subdir_a"), "depth": 2, "size_bytes": 20, "file_extension": "jpg",
          "imported_path": base_imported_path},
        {"path": str(complex_temp_dir / "subdir_a" / "subdir_aa" / "fileAA.csv"), "name": "fileAA.csv",
          "type": "file", "parent": str(complex_temp_dir / "subdir_a" / "subdir_aa"), "depth": 3,
          "size_bytes": 30, "file_extension": "csv",
          "imported_path": base_imported_path},
        {"path": str(complex_temp_dir / "subdir_b" / "fileB.png"), "name": "fileB.png", "type": "file",
          "parent": str(complex_temp_dir / "subdir_b"), "depth": 2, "size_bytes": 40,
          "file_extension": "png",
          "imported_path": base_imported_path},
    ]
    _assert_directory_tree_df(df, expected_data, base_imported_path)


def test_walk_filesystem_empty_dir(empty_temp_dir: Path):
    """Tests _fetch_single_directory_tree with an empty directory."""
    df = walk_filesystem([empty_temp_dir], accepted_extensions="all")
    assert df.is_empty()


def test_walk_filesystem_single_file_dir(single_file_dir: Path):
    """Tests _fetch_single_directory_tree with a directory containing only one file."""
    df = walk_filesystem([single_file_dir], accepted_extensions="all")
    base_imported_path = str(single_file_dir)
    expected_data = [{
        "path": str(single_file_dir / "single_file.txt"), "name": "single_file.txt", "type": "file",
        "parent": str(single_file_dir), "depth": 1,
        "size_bytes": len("content".encode('utf-8')),
        "file_extension": "txt", "imported_path": base_imported_path,
    }]
    _assert_directory_tree_df(df, expected_data, base_imported_path)


@pytest.mark.parametrize("path_type_creator", [
    lambda tmp_path: (tmp_path / "not_a_dir.txt").touch() or (tmp_path / "not_a_dir.txt"),
    lambda tmp_path: tmp_path / "i_do_not_exist",
])
def test_walk_filesystem_invalid_paths(tmp_path: Path, path_type_creator):
    invalid_path = path_type_creator(tmp_path)
    df = walk_filesystem([invalid_path], accepted_extensions="all")
    assert isinstance(df, pl.DataFrame) and df.is_empty()


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
        "imported_path": ["/a", "/a", "/b"],
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
        "imported_path": ["/root"] * 4,
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
        "imported_path": ["/rootA", "/rootA", "/rootB", "/rootB"],
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
        "imported_path": ["/a"] * 4,
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
        "imported_path": ["/a", "/a", "/a", "/b", "/b"],
    })


@pytest.mark.parametrize(
    "input_df_fixture, expected_sizes_map",
    [
        ("sample_flat_df", {"/a/file1.txt": 100, "/a/file2.jpg": 200, "/b/file3.csv": 50}),
        ("sample_simple_nested_df", {"/root": 150, "/root/file1.txt": 100, "/root/subdir": 50, "/root/subdir/file2.txt": 50}),
        ("sample_multiple_roots_df", {"/rootA": 10, "/rootA/fileA.txt": 10, "/rootB": 20, "/rootB/fileB.txt": 20}),
        ("sample_deep_nested_df", {"/a": 75, "/a/b": 75, "/a/b/c": 75, "/a/b/c/file.txt": 75}),
        ("sample_empty_folders_df", {"/a": 100, "/a/empty_subdir": 0, "/a/file.txt": 100, "/b": 0, "/b/another_empty_dir": 0}),
    ]
)
def test_aggregate_folder_sizes(request, input_df_fixture, expected_sizes_map):
    """Tests aggregation on various directory structures."""
    input_df = request.getfixturevalue(input_df_fixture)
    aggregated_df = _aggregate_folder_sizes(input_df).sort("path")

    actual_sizes = aggregated_df.select(pl.col("path"), pl.col("size_bytes")).to_dicts()

    for row in actual_sizes:
        assert row["size_bytes"] == expected_sizes_map[row["path"]]


def test_aggregate_folder_sizes_empty_input_df():
    """Tests aggregation with an empty input DataFrame."""
    empty_df = pl.DataFrame([], schema=PATHS_DF_EXPECTED_SCHEMA)
    aggregated_df = _aggregate_folder_sizes(empty_df)
    assert aggregated_df.is_empty()
    assert aggregated_df.schema == empty_df.schema