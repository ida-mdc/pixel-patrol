import pytest
from pathlib import Path
from typing import List, Dict, Any
from PIL import Image
import numpy as np
from pixel_patrol_base.core.project import Project
from pixel_patrol_base.core.project_settings import Settings

@pytest.fixture
def mock_project_name() -> str:
    return "My Test Project"


@pytest.fixture
def project_instance(mock_project_name: str, tmp_path: Path) -> Project:
    """Provides a Project instance with a base directory set (e.g., tmp_path) directly upon creation."""
    from pixel_patrol_base import api
    return api.create_project(mock_project_name, tmp_path, loader="bioio")


@pytest.fixture
def temp_test_dirs(tmp_path: Path) -> List[Path]:
    """
    Creates temporary directories for testing purposes with a guaranteed set of processable images.
    This fixture now serves as the consolidated "image-rich" temporary directory.
    """
    dir1 = tmp_path / "test_dir1"
    dir2 = tmp_path / "test_dir2"
    subdir_a = dir1 / "subdir_a"

    dir1.mkdir()
    dir2.mkdir()
    subdir_a.mkdir()

    img_array = np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8)
    Image.fromarray(img_array).save(dir1 / "valid_image1.jpg")
    Image.fromarray(img_array).save(dir2 / "valid_image2.png")
    (subdir_a / "fileC.txt").touch() # Still keep other file types

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
def patch_tree(mocker):
    return lambda df: mocker.patch(
        "pixel_patrol_base.core.processing._fetch_single_directory_tree",
        return_value=df,
    )


@pytest.fixture
def mock_settings() -> Settings:
    """Provides a default Settings instance."""
    return Settings(selected_file_extensions={"jpg", "png", "tif", "jpeg"})


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

@pytest.fixture
def project_with_minimal_data(project_instance: Project, temp_test_dirs: list[Path]) -> Project:
    """
    Provides a Project with base_dir and paths_df (minimal data).
    It interacts directly with the Project object methods.
    """
    project = project_instance # Already has base_dir set by fixture
    project.add_paths(temp_test_dirs) # Direct call to Project method
    return project

@pytest.fixture
def project_with_all_data(project_instance: Project, temp_test_dirs: list[Path]) -> Project:
    """
    Provides a Project with base_dir, paths_df, images_df, and custom settings,
    guaranteeing processable images.
    """
    project = project_instance # Already has base_dir set by fixture

    # Add paths from the image-rich directory structure
    project.add_paths(temp_test_dirs)

    # Set some custom settings for image processing
    new_settings = Settings(
        cmap="viridis",
        n_example_files=5,
        selected_file_extensions={"jpg", "png", "gif"} # Match extensions
    )
    project.set_settings(new_settings)

    project.process_artifacts()

    return project


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Provides the path to the 'ella_extras/data' directory."""
    return Path(__file__).parent / "data"


@pytest.fixture(scope="module")
def expected_image_data() -> Dict[str, Dict[str, Any]]:
    """
    Defines expected properties for each test image file.
    Adjust y_size and x_size if your actual images are not 16x16.
    """
    return {
        "cyx_16bit.tif": {
            "C_size": 2,
            "Y_size": 35,
            "X_size": 39,
            "Z_size": 1,
            "T_size": 1,
            "M_size": None,
            "dtype": "uint16",
            "ndim": 5,
        },
        "rgb.bmp": {
            "C_size": 1,
            "Y_size": 48,
            "X_size": 48,
            "Z_size": 1,
            "T_size": 1,
            "S_size": 3,
            "M_size": None,
            "dtype": "uint8",
            "ndim": 6,
        },
        "tcyx_8bit.tif": {
            "T_size": 10,
            "C_size": 3,
            "Y_size": 31,
            "X_size": 33,
            "Z_size": 1,
            "M_size": None,
            "dtype": "uint8",
            "ndim": 5,
        },
        "yx_8bit.jpeg": {
            "C_size": 1,
            "Y_size": 78,
            "X_size": 180,
            "Z_size": 1,
            "T_size": 1,
            "S_size": 3,
            "M_size": None,
            "dtype": "uint8",
            "ndim": 6,
        },
        "yx_8bit.png": {
            "C_size": 1,
            "Y_size": 33,
            "X_size": 33,
            "Z_size": 1,
            "T_size": 1,
            "M_size": None,
            "dtype": "uint8",
            "ndim": 5,
        },
        "yx_rgb.png": {
            "C_size": 1,
            "Y_size": 111,
            "X_size": 322,
            "Z_size": 1,
            "T_size": 1,
            "S_size": 3,
            "M_size": None,
            "dtype": "uint8",
            "ndim": 6,
        },
        "zyx_16bit.tif": {
            "Z_size": 5,
            "C_size": 1,
            "Y_size": 25,
            "X_size": 26,
            "T_size": 1,
            "M_size": None,
            "dtype": "uint16",
            "ndim": 5,
        },
    }
