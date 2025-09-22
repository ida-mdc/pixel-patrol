from pathlib import Path
from typing import Dict, Iterable, List

import pytest


# --- Paths -------------------------------------------------------------------

def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _example_bioio_dir() -> Path:
    return _repo_root() / "examples" / "datasets" / "bioio"

# --- Fixtures ----------------------------------------------------------------

@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """
    The directory that contains the example image data used by tests.
    """
    d = _example_bioio_dir()
    if not d.exists():
        pytest.skip(f"Example data directory not found: {d}")
    return d


@pytest.fixture(scope="session")
def expected_image_data() -> Dict[str, Dict[str, object]]:
    """
    Optional ground truths for a few known files. Tests remain robust if a file isn’t listed here.
    Keep this intentionally light-weight and non-brittle.
    """
    return {
        # PNG grayscale
        "yx_8bit.png": {
            "dtype": "uint8",
            "min_ndim": 2,
        },
        # PNG RGB (may be loaded as 3-channel or squeezed; we assert consistency, not exact values)
        "yx_rgb.png": {
            "min_ndim": 2,
        },
        # TIFF examples
        "tcyx_8bit.tif": {"dtype": "uint8"},
        "zyx_16bit.tif": {"dtype": "uint16"},
        "cyx_16bit.tif": {"dtype": "uint16"},
        # BMP/JPEG
        "rgb.bmp": {"min_ndim": 2},
        "yx_8bit.jpeg": {"dtype": "uint8"},
    }


# --- Auto-parametrize any test that asks for `image_file_path` ----------------

def _iter_image_files(root: Path) -> Iterable[Path]:
    # Accept anything under the folder except an explicit "not_an_image.txt".
    for p in root.rglob("*"):
        if p.is_file() and p.name != "not_an_image.txt":
            yield p


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """
    If a test function declares an `image_file_path` parameter, parametrize it
    over all files in the example bioio dataset.
    """
    if "image_file_path" in metafunc.fixturenames:
        root = _example_bioio_dir()
        files: List[Path] = list(_iter_image_files(root))
        if not files:
            pytest.skip(f"No example images found under {root}")
        metafunc.parametrize("image_file_path", files, ids=[f.name for f in files])
