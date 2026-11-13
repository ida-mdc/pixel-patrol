from pathlib import Path
import tomllib
import pytest

PACKAGES_DIR = Path(__file__).parents[2]
PYPROJECT = PACKAGES_DIR / "pixel-patrol" / "pyproject.toml"


def _dependency_dirs():
    with open(PYPROJECT, "rb") as f:
        data = tomllib.load(f)
    deps = data["project"]["dependencies"]
    return [
        PACKAGES_DIR / dep.split("==")[0]
        for dep in deps
        if dep.startswith("pixel-patrol-")
    ]


@pytest.mark.parametrize("package_dir", _dependency_dirs())
@pytest.mark.parametrize("filename", ["README.md", "LICENSE"])
def test_package_has_required_file(package_dir, filename):
    assert (package_dir / filename).exists(), f"{package_dir.name} is missing {filename}"
