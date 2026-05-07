from pathlib import Path
from typing import List

import numpy as np
import pytest
from PIL import Image

from pixel_patrol_base.core.project import Project


@pytest.fixture
def mock_project_name() -> str:
    return "My Test Project"


@pytest.fixture
def project_instance(mock_project_name: str, tmp_path: Path) -> Project:
    """Provides a Project instance with a base directory set directly upon creation."""
    from pixel_patrol_base import api
    return api.create_project(mock_project_name, tmp_path)


@pytest.fixture
def temp_test_dirs(tmp_path: Path) -> List[Path]:
    """Creates temporary directories with processable images for integration tests."""
    dir1 = tmp_path / "test_dir1"
    dir2 = tmp_path / "test_dir2"
    subdir_a = dir1 / "subdir_a"

    dir1.mkdir()
    dir2.mkdir()
    subdir_a.mkdir()

    img_array = np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8)
    Image.fromarray(img_array).save(dir1 / "valid_image1.jpg")
    Image.fromarray(img_array).save(dir2 / "valid_image2.png")
    (subdir_a / "fileC.txt").touch()

    return [dir1, dir2]


@pytest.fixture
def project_with_all_data(project_instance: Project, temp_test_dirs: List[Path]) -> Project:
    """Provides a Project with paths added and records fully processed."""
    project_instance.add_paths(temp_test_dirs)
    project_instance.process_records()
    return project_instance