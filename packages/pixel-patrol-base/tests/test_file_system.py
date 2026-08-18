import pytest
import polars as pl
import re
from pathlib import Path
from datetime import datetime, timezone
import os


from pixel_patrol_base.core.file_system import FOLDER_DATASET_KEY, _discover_files

_PARENT_LEVEL_RE = re.compile(r"^parent\d+$")  # parent0, parent1, ...

# --- Fixture ---

@pytest.fixture
def complex_temp_dir(tmp_path: Path) -> Path:
    """
    Creates a complex temporary directory structure with files for testing _discover_files.
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
def deep_many_files_dir(tmp_path: Path) -> Path:
    """Directory with many, differently deep files."""
    root = tmp_path / "deep_many_files_test_root"
    root.mkdir()
    # 200 files at depth=2
    for level1 in range(200):
        subdir = root / f"subdir_{level1}"
        subdir.mkdir()
        (subdir / f"file{level1}.txt").write_bytes(bytes(str(level1), "utf8") * 10)
    # 100 files at depth=3
    subdir_b = root / "subdir_b"
    subdir_b.mkdir()
    for level2 in range(100):
        subdir = subdir_b / f"subdir_b{level2}"
        subdir.mkdir()
        (subdir / f"file_{level2}.txt").write_bytes(bytes(str(level2), "utf8") * 20)
    # super deep file (2*26 parents)
    deep_subdir = root
    for subdir in list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"):
        deep_subdir = deep_subdir / f"subdir_{subdir}"
    deep_subdir.mkdir(parents=True)
    (deep_subdir / f"file_deep.txt").write_bytes(b"deep" * 10)

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


@pytest.fixture
def sample_simple_nested_df() -> pl.DataFrame:
    """Provides a DataFrame for a simple nested structure."""
    return pl.DataFrame({
        "path": ["/root", "/root/file1.txt", "/root/subdir", "/root/subdir/file2.txt"],
        "name": ["root", "file1.txt", "subdir", "file2.txt"],
        "type": ["folder", "file", "folder", "file"],
        "parent0": [None, None, None, "subdir"],
        "depth": [0, 1, 1, 2],
        "size_bytes": [0, 100, 0, 50],  # Initial folder sizes are 0
        "modification_date": [datetime.now()] * 4,
        "file_extension": [None, "txt", None, "txt"],
        "imported_path": ["/root"] * 4,
    })


def _assert_directory_tree_df(df: pl.DataFrame, expected_data: list[dict], imported_path: str):
    """
    Helper to assert the content and schema of a walk_filesystem DataFrame.
    Handles dynamic values like modification_date.
    """
    assert isinstance(df, pl.DataFrame)
    assert not df.is_empty()

    expected_schema = {
        "path":              pl.String,
        "name":              pl.String,
        "type":              pl.String,
        "depth":             pl.Int64,
        "size_bytes":        pl.Int64,
        "modification_date": pl.Datetime(time_unit="us", time_zone=None),
        "file_extension":    pl.String,
        "imported_path":     pl.String,
    }

    actual_schema = df.schema

    for col, expected_type in expected_schema.items():
        assert col in actual_schema, f"Column '{col}' missing from actual schema"
        if col == "file_extension":
            assert actual_schema[col] == expected_type or actual_schema[col] == pl.Null, \
                f"Mismatch in schema for column '{col}': Expected {expected_type} or {pl.Null}, got {actual_schema[col]}"
        else:
            assert actual_schema[col] == expected_type, \
                f"Mismatch in schema for column '{col}': Expected {expected_type}, got {actual_schema[col]}"

    # Check parentN columns
    expected_parent_cols = sorted({k for row in expected_data for k in row if _PARENT_LEVEL_RE.match(k)})
    actual_parent_cols = sorted(c for c in actual_schema if _PARENT_LEVEL_RE.match(c))
    assert actual_parent_cols == expected_parent_cols, \
        f"Mismatch in parent level columns: expected {expected_parent_cols}, got {actual_parent_cols}"
    for col in actual_parent_cols:
        assert actual_schema[col] == pl.String, \
            f"Mismatch in schema for column '{col}': Expected {pl.String}, got {actual_schema[col]}"

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
        # Levels the file is not deep enough to have must be null.
        for col in actual_parent_cols:
            if col not in expected_row:
                assert actual_row[col] is None, f"Mismatch in row {i}, key '{col}': expected None"


# TODO: these tests use walk_filesystem which doesn't exist in this branch;
# rewrite to call _discover_files + add_parent_level_columns directly.

def test_fetch_single_directory_tree_complex_structure(complex_temp_dir: Path):
    """
    Tests walk_filesystem with a complex directory structure.
    Verifies column types, content, depth, parentN, and imported_path.
    """
    df = walk_filesystem([complex_temp_dir], accepted_extensions="all")
    base_imported_path = str(complex_temp_dir)

    expected_data = [
        {"path": str(complex_temp_dir / "file1.txt"), "name": "file1.txt", "type": "file",
          "depth": 1, "size_bytes": 10, "file_extension": "txt",
          "imported_path": base_imported_path},
        {"path": str(complex_temp_dir / "subdir_a" / "fileA.jpg"), "name": "fileA.jpg", "type": "file",
          "parent0": "subdir_a",
          "depth": 2, "size_bytes": 20, "file_extension": "jpg",
          "imported_path": base_imported_path},
        {"path": str(complex_temp_dir / "subdir_a" / "subdir_aa" / "fileAA.csv"), "name": "fileAA.csv",
          "type": "file",
          "parent0": "subdir_a", "parent1": "subdir_aa",
          "depth": 3, "size_bytes": 30, "file_extension": "csv",
          "imported_path": base_imported_path},
        {"path": str(complex_temp_dir / "subdir_b" / "fileB.png"), "name": "fileB.png", "type": "file",
          "parent0": "subdir_b",
          "depth": 2, "size_bytes": 40,
          "file_extension": "png",
          "imported_path": base_imported_path},
    ]
    _assert_directory_tree_df(df, expected_data, base_imported_path)


def test_deep_many_files_directory(deep_many_files_dir: Path):
    df = walk_filesystem([deep_many_files_dir], accepted_extensions="all")
    assert len(df) == 301
    assert {"parent0", "parent1", "parent2", "parent51"}.issubset(set(df.columns))


def test_walk_filesystem_empty_dir(empty_temp_dir: Path):
    """Tests walk_filesystem with an empty directory."""
    df = walk_filesystem([empty_temp_dir], accepted_extensions="all")
    assert df.is_empty()


def test_walk_filesystem_single_file_dir(single_file_dir: Path):
    """Tests walk_filesystem with a directory containing only one file."""
    df = walk_filesystem([single_file_dir], accepted_extensions="all")
    base_imported_path = str(single_file_dir)
    expected_data = [{
        "path": str(single_file_dir / "single_file.txt"), "name": "single_file.txt", "type": "file",
        "depth": 1,
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


# --- Tests for _discover_files relative-path conversion ---

def _discover_by_name(bases: list[Path], base_dir: Path | None = None) -> dict[str, dict]:
    """Run _discover_files and key the yielded metadata by file name for assertions."""
    return {
        meta["name"]: meta
        for _, meta in _discover_files(bases, "all", base_dir=base_dir)
    }


def test_discover_files_paths_relative_to_base_dir(complex_temp_dir: Path):
    """With base_dir set, path/imported_path are stored relative to it."""
    root = complex_temp_dir.resolve()
    meta = _discover_by_name([root], base_dir=root)

    assert meta["file1.txt"]["path"] == "file1.txt"
    assert meta["file1.txt"]["imported_path"] == "."

    assert meta["fileA.jpg"]["path"] == os.path.join("subdir_a", "fileA.jpg")

    assert meta["fileAA.csv"]["path"] == os.path.join("subdir_a", "subdir_aa", "fileAA.csv")


# TODO: parentN is not in raw _discover_files output in this branch (added post-hoc);
# rewrite these three tests to call add_parent_level_columns on a DataFrame built from
# _discover_files, and assert on that.

def test_discover_files_parent_levels_are_numbered_top_down(complex_temp_dir: Path):
    """parent0 is the first directory below base_dir, parent1 the next one down."""
    root = complex_temp_dir.resolve()
    meta = _discover_by_name([root], base_dir=root)

    # Directly in the base directory - not deep enough to have any level.
    assert "parent0" not in meta["file1.txt"]
    assert meta["fileA.jpg"]["parent0"] == "subdir_a"
    assert meta["fileB.png"]["parent0"] == "subdir_b"
    assert meta["fileAA.csv"]["parent0"] == "subdir_a"
    assert meta["fileAA.csv"]["parent1"] == "subdir_aa"


def test_discover_files_base_dir_ancestor_of_base(complex_temp_dir: Path):
    """When base_dir is an ancestor of the scanned base, paths stay relative to base_dir
    and imported_path names the base's location beneath it."""
    root = complex_temp_dir.resolve()
    subdir = root / "subdir_a"
    meta = _discover_by_name([subdir], base_dir=root)

    assert meta["fileA.jpg"]["path"] == os.path.join("subdir_a", "fileA.jpg")
    assert meta["fileA.jpg"]["imported_path"] == "subdir_a"
    assert meta["fileA.jpg"]["parent0"] == "subdir_a"
    assert meta["fileAA.csv"]["parent0"] == "subdir_a"
    assert meta["fileAA.csv"]["parent1"] == "subdir_aa"


def test_discover_files_absolute_paths_without_base_dir(complex_temp_dir: Path):
    """Without base_dir, path/imported_path remain absolute (default behavior)."""
    root = complex_temp_dir.resolve()
    meta = _discover_by_name([root])

    assert meta["file1.txt"]["path"] == str(root / "file1.txt")
    assert meta["file1.txt"]["imported_path"] == str(root)


def test_discover_files_parent_levels_without_base_dir(complex_temp_dir: Path):
    """Without base_dir the levels are counted from the scanned base, so they stay
    directory names rather than turning into the absolute path's leading components."""
    root = complex_temp_dir.resolve()
    meta = _discover_by_name([root])

    assert "parent0" not in meta["file1.txt"]
    assert meta["fileB.png"]["parent0"] == "subdir_b"
    assert meta["fileAA.csv"]["parent0"] == "subdir_a"
    assert meta["fileAA.csv"]["parent1"] == "subdir_aa"


# --- Tests for folder-dataset discovery ---

def _dataset_dir(root: Path, name: str, *files: tuple[str, int]) -> Path:
    """A directory that a loader claims as one dataset, with sized files inside it."""
    d = root / name
    d.mkdir(parents=True)
    for rel, size in files or (("volume.tif", 1),):
        (d / rel).parent.mkdir(parents=True, exist_ok=True)
        (d / rel).write_bytes(b"x" * size)
    return d


def test_directory_claimed_by_loader_is_yielded_as_one_dataset(tmp_path: Path):
    """A dataset directory with no telltale suffix is discovered, and not walked into."""
    _dataset_dir(tmp_path, "cell_a", ("source.tif", 1000), ("nested/chunk", 250))
    (tmp_path / "loose.tif").write_bytes(b"x" * 40)

    found = {
        meta["name"]: meta
        for _, meta in _discover_files(
            [tmp_path], "all", is_folder_dataset=lambda p: p.name == "cell_a"
        )
    }

    assert set(found) == {"cell_a", "loose.tif"}
    assert found["cell_a"][FOLDER_DATASET_KEY] is True
    assert FOLDER_DATASET_KEY not in found["loose.tif"]
    assert found["loose.tif"]["size_bytes"] == 40
    # A directory's own st_size is its inode, so the dataset has to report the total
    # of the files inside it - including nested ones.
    assert found["cell_a"]["size_bytes"] == 1250


def test_claimed_directory_is_yielded_even_when_extensions_are_restricted(tmp_path: Path):
    """A folder dataset has no meaningful suffix, so the extension filter can't judge it."""
    _dataset_dir(tmp_path, "cell_a")

    names = [
        meta["name"]
        for _, meta in _discover_files(
            [tmp_path], {".nd2"}, is_folder_dataset=lambda p: p.name == "cell_a"
        )
    ]

    assert names == ["cell_a"]


def test_unclaimed_directories_are_still_walked_into(tmp_path: Path):
    """Refusing a directory must leave discovery of its contents unchanged."""
    _dataset_dir(tmp_path, "plain_dir", ("inside.tif", 1))

    names = [
        meta["name"]
        for _, meta in _discover_files([tmp_path], "all", is_folder_dataset=lambda p: False)
    ]

    assert names == ["inside.tif"]


def test_extension_matched_folder_datasets_are_marked_and_sized_too(tmp_path: Path):
    """The zarr-style path behaves identically, so planning has one rule to follow."""
    _dataset_dir(tmp_path, "store.zarr", (".zarray", 1), ("0.0", 500))

    found = {
        meta["name"]: meta
        for _, meta in _discover_files([tmp_path], "all", folder_extensions={"zarr"})
    }

    assert found["store.zarr"][FOLDER_DATASET_KEY] is True
    assert found["store.zarr"]["size_bytes"] == 501  # .zarray (1 byte) + chunk (500)
