import pytest
from pathlib import Path
from datetime import datetime, timezone
import os


from pixel_patrol_base.core.file_system import FOLDER_DATASET_KEY, _discover_files


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


def test_discover_files_base_dir_ancestor_of_base(complex_temp_dir: Path):
    """When base_dir is an ancestor of the scanned base, paths stay relative to base_dir
    and imported_path names the base's location beneath it."""
    root = complex_temp_dir.resolve()
    subdir = root / "subdir_a"
    meta = _discover_by_name([subdir], base_dir=root)

    assert meta["fileA.jpg"]["path"] == os.path.join("subdir_a", "fileA.jpg")
    assert meta["fileA.jpg"]["imported_path"] == "subdir_a"


def test_discover_files_absolute_paths_without_base_dir(complex_temp_dir: Path):
    """Without base_dir, path/imported_path remain absolute (default behavior)."""
    root = complex_temp_dir.resolve()
    meta = _discover_by_name([root])

    assert meta["file1.txt"]["path"] == str(root / "file1.txt")
    assert meta["file1.txt"]["imported_path"] == str(root)


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
