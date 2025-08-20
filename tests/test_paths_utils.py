import pytest
from pathlib import Path
from pixel_patrol_base.utils.path_utils import find_common_base
import os


def test_find_common_base_multiple_paths():
    """Test finding a common base with multiple paths."""
    paths = [
        str(Path("/home/user/data/photos/2023/image1.jpg")),
        str(Path("/home/user/data/photos/2023/vacation/image2.png")),
        str(Path("/home/user/data/photos/2024/image3.gif")),
    ]
    expected = str(Path("/home/user/data/photos")) + str(Path().anchor or "/")
    # Remove double slashes if any
    expected = expected.replace("//", "/").replace("\\\\", "\\")
    result = find_common_base(paths)
    assert result.rstrip("/\\") == expected.rstrip("/\\")


def test_find_common_base_single_path():
    """Test finding common base with a single path."""
    # The function returns the parent directory for a single file path
    file_path = Path("/home/user/data/image.jpg")
    paths = [str(file_path)]
    expected = str(file_path.parent)
    result = find_common_base(paths)
    # Normalize slashes for comparison
    assert Path(result) == Path(expected)


def test_find_common_base_empty_list():
    """Test finding common base with an empty list."""
    paths: list[str] = []
    assert find_common_base(paths) == ""


def test_find_common_base_same_paths():
    """Test finding common base with all paths being identical."""
    paths = [str(Path("/home/user/data/images/")), str(Path("/home/user/data/images"))]
    expected = str(Path("/home/user/data/images")) + str(Path().anchor or "/")
    expected = expected.replace("//", "/").replace("\\\\", "\\")
    result = find_common_base(paths)
    assert result.rstrip("/\\") == expected.rstrip("/\\")


def test_find_common_base_common_base_is_root():
    """Test finding common base when the common base is the root directory."""
    paths = [str(Path("/a/b/c/file1.txt")), str(Path("/a/b/d/file2.txt"))]
    expected = str(Path("/a/b")) + str(Path().anchor or "/")
    expected = expected.replace("//", "/").replace("\\\\", "\\")
    result = find_common_base(paths)
    assert result.rstrip("/\\") == expected.rstrip("/\\")
