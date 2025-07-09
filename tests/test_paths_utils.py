import pytest
from pathlib import Path
from pixel_patrol.utils.path_utils import find_common_base
import os


def test_find_common_base_multiple_paths():
    """Test finding a common base with multiple paths."""
    paths = [
        "/home/user/data/photos/2023/image1.jpg",
        "/home/user/data/photos/2023/vacation/image2.png",
        "/home/user/data/photos/2024/image3.gif",
    ]
    assert find_common_base(paths) == "/home/user/data/photos/"


def test_find_common_base_single_path():
    """Test finding common base with a single path."""
    paths = [
        "/home/user/data/image.jpg",
    ]
    assert find_common_base(paths) == "/home/user/data/"


def test_find_common_base_empty_list():
    """Test finding common base with an empty list."""
    paths: list[str] = []
    assert find_common_base(paths) == ""


def test_find_common_base_same_paths():
    """Test finding common base with all paths being identical."""
    paths = [
        "/home/user/data/images/",
        "/home/user/data/images",
    ]
    assert find_common_base(paths) == "/home/user/data/images/"


def test_find_common_base_common_base_is_root():
    """Test finding common base when the common base is the root directory."""
    paths = [
        "/a/b/c/file1.txt",
        "/a/b/d/file2.txt",
    ]
    assert find_common_base(paths) == "/a/b/"
