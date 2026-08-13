import pytest
from pathlib import Path
from datetime import datetime, timezone
import os


from pixel_patrol_base.core.file_system import _discover_files

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
    """With base_dir set, path/parent/imported_path are stored relative to it."""
    root = complex_temp_dir.resolve()
    meta = _discover_by_name([root], base_dir=root)

    assert meta["file1.txt"]["path"] == "file1.txt"
    assert meta["file1.txt"]["parent"] == "."
    assert meta["file1.txt"]["imported_path"] == "."

    assert meta["fileA.jpg"]["path"] == os.path.join("subdir_a", "fileA.jpg")
    assert meta["fileA.jpg"]["parent"] == "subdir_a"

    assert meta["fileAA.csv"]["path"] == os.path.join("subdir_a", "subdir_aa", "fileAA.csv")
    assert meta["fileAA.csv"]["parent"] == os.path.join("subdir_a", "subdir_aa")


def test_discover_files_base_dir_ancestor_of_base(complex_temp_dir: Path):
    """When base_dir is an ancestor of the scanned base, paths stay relative to base_dir
    and imported_path names the base's location beneath it."""
    root = complex_temp_dir.resolve()
    subdir = root / "subdir_a"
    meta = _discover_by_name([subdir], base_dir=root)

    assert meta["fileA.jpg"]["path"] == os.path.join("subdir_a", "fileA.jpg")
    assert meta["fileA.jpg"]["parent"] == "subdir_a"
    assert meta["fileA.jpg"]["imported_path"] == "subdir_a"


def test_discover_files_absolute_paths_without_base_dir(complex_temp_dir: Path):
    """Without base_dir, path/parent/imported_path remain absolute (default behavior)."""
    root = complex_temp_dir.resolve()
    meta = _discover_by_name([root])

    assert meta["file1.txt"]["path"] == str(root / "file1.txt")
    assert meta["file1.txt"]["parent"] == str(root)
    assert meta["file1.txt"]["imported_path"] == str(root)
