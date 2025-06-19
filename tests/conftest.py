import pytest
from pathlib import Path
import polars as pl
from typing import List
from datetime import datetime
from pixel_patrol.core.project import Project
from pixel_patrol.core.project_settings import Settings
from pixel_patrol.core.processing import PATHS_DF_EXPECTED_SCHEMA
from pixel_patrol.utils.utils import format_bytes_to_human_readable # For testing the formatted column


@pytest.fixture
def mock_project_name() -> str:
    return "My Test Project"


@pytest.fixture
def project_instance(mock_project_name: str, tmp_path: Path) -> Project:
    """Provides a Project instance with a base directory set (e.g., tmp_path) directly upon creation."""
    from pixel_patrol import api
    return api.create_project(mock_project_name, tmp_path)


@pytest.fixture
def temp_test_dirs(tmp_path: Path) -> List[Path]:
    """Creates temporary directories for testing purposes."""
    dir1 = tmp_path / "test_dir1"
    dir2 = tmp_path / "test_dir2"
    subdir_a = dir1 / "subdir_a"

    # Create the directories
    dir1.mkdir()
    dir2.mkdir()
    subdir_a.mkdir()

    # You might also want to create some files within these directories
    # if your processing.process_paths expects to find files.
    (dir1 / "fileA.jpg").touch()
    (subdir_a / "fileC.txt").touch()
    (dir2 / "fileB.png").touch()

    return [dir1, dir2]

@pytest.fixture
def mock_temp_file_system(tmp_path: Path) -> List[Path]:
    """
    Creates a temporary directory structure with specific file sizes
    to match the expectations of build_paths_df test's expected data.
    """
    # Create the directory structure
    dir1 = tmp_path / "test_dir1"
    dir1.mkdir()
    (dir1 / "fileA.jpg").write_bytes(b"abc")  # 3 bytes
    subdir_a = dir1 / "subdir_a"
    subdir_a.mkdir()
    (subdir_a / "fileC.txt").write_bytes(b"abcd")  # 4 bytes

    dir2 = tmp_path / "test_dir2"
    dir2.mkdir()
    (dir2 / "fileB.png").write_bytes(b"abcde")  # 5 bytes

    # Return the root paths that would be passed to build_paths_df
    return [dir1, dir2]

@pytest.fixture
def mock_settings() -> Settings:
    """Provides a default Settings instance."""
    return Settings(selected_file_extensions={"jpg", "png", "tif", "jpeg"})


@pytest.fixture
def mock_paths_df_content(tmp_path: Path) -> pl.DataFrame:
    """
    Provides a simple mock DataFrame representing file system data.
    Ensures it matches PATHS_DF_EXPECTED_SCHEMA and uses dynamic paths relative to tmp_path.
    """
    # Use tmp_path to construct absolute paths
    base_path_str = str(tmp_path)
    dir1_path = tmp_path / "test_dir1"
    dir2_path = tmp_path / "test_dir2"
    subdir_a_path = dir1_path / "subdir_a"

    data = [
        {"path": str(dir1_path), "name": "test_dir1", "type": "folder", "parent": base_path_str, "depth": 0,
         "size_bytes": 1024, "modification_date": datetime.now(), "file_extension": None},
        {"path": str(dir1_path / "fileA.jpg"), "name": "fileA.jpg", "type": "file", "parent": str(dir1_path),
         "depth": 1, "size_bytes": 512, "modification_date": datetime.now(), "file_extension": "jpg"},
        {"path": str(subdir_a_path), "name": "subdir_a", "type": "folder", "parent": str(dir1_path), "depth": 1,
         "size_bytes": 512, "modification_date": datetime.now(), "file_extension": None},
        {"path": str(subdir_a_path / "fileC.txt"), "name": "fileC.txt", "type": "file", "parent": str(subdir_a_path),
         "depth": 2, "size_bytes": 512, "modification_date": datetime.now(), "file_extension": "txt"},
        {"path": str(dir2_path), "name": "test_dir2", "type": "folder", "parent": base_path_str, "depth": 0,
         "size_bytes": 2048, "modification_date": datetime.now(), "file_extension": None},
        {"path": str(dir2_path / "fileB.png"), "name": "fileB.png", "type": "file", "parent": str(dir2_path),
         "depth": 1, "size_bytes": 2048, "modification_date": datetime.now(), "file_extension": "png"},
    ]

    df = pl.DataFrame(data).with_columns(
        pl.col("size_bytes").map_elements(format_bytes_to_human_readable, return_dtype=pl.String).alias("size_readable")
    ).select(*[pl.col(name).cast(dtype) for name, dtype in PATHS_DF_EXPECTED_SCHEMA.items()])
    return df


@pytest.fixture
def mock_temp_file_system_complex(tmp_path: Path) -> List[Path]:
    """
    Creates a temporary directory structure with specific file sizes
    to test complex size aggregation scenarios in build_paths_df.
    Structure:
    - root_dir/
      - sub_dir_a/
        - fileA.txt (10 bytes)
        - sub_sub_dir_a/
          - fileB.txt (20 bytes)
      - sub_dir_b/
        - fileC.txt (30 bytes)
    """
    root_dir = tmp_path / "root_dir"
    root_dir.mkdir()

    sub_dir_a = root_dir / "sub_dir_a"
    sub_dir_a.mkdir()
    (sub_dir_a / "fileA.txt").write_bytes(b"0123456789") # 10 bytes

    sub_sub_dir_a = sub_dir_a / "sub_sub_dir_a"
    sub_sub_dir_a.mkdir()
    (sub_sub_dir_a / "fileB.txt").write_bytes(b"01234567890123456789") # 20 bytes

    sub_dir_b = root_dir / "sub_dir_b"
    sub_dir_b.mkdir()
    (sub_dir_b / "fileC.txt").write_bytes(b"012345678901234567890123456789") # 30 bytes

    return [root_dir]

@pytest.fixture
def mock_temp_file_system_edge_cases(tmp_path: Path) -> List[Path]:
    """
    Creates temporary files to test various file extension edge cases.
    """
    test_dir = tmp_path / "extension_test_dir"
    test_dir.mkdir()

    (test_dir / "file_no_ext").write_bytes(b"content")
    (test_dir / "archive.tar.gz").write_bytes(b"content")
    (test_dir / "image.JPEG").write_bytes(b"content") # Mixed case
    (test_dir / ".hidden_file").write_bytes(b"content") # Hidden file
    (test_dir / "regular.png").write_bytes(b"content")

    return [test_dir]
