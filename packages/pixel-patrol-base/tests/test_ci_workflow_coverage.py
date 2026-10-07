"""Check that all packages have their unit tests run in CI"""

import subprocess
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
PACKAGES_DIR = REPO_ROOT / "packages"
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "python-tests.yml"


def _get_tracked_package_dirs() -> set:
    """Get all files in packages/ tracked with git"""
    git_ls_cmd = "git ls-files -z packages".split(" ")
    out = subprocess.run(git_ls_cmd, cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout
    tracked_package_dirs = {Path(str(p)) for p in out.split("\0") if p}
    tracked_package_names = {p.relative_to("packages").parts[0] for p in tracked_package_dirs}
    return tracked_package_names


def _has_package_test(pkg_name: str) -> bool:
    pkg_dir = PACKAGES_DIR / pkg_name
    has_tests = any((pkg_dir / "tests").rglob("test_*.py"))
    return has_tests


def _packages_in_ci_matrix() -> set:
    workflow = yaml.safe_load(WORKFLOW_PATH.read_text())
    testing_matrix_ci = set(workflow["jobs"]["test-packages"]["strategy"]["matrix"]["package"])
    return testing_matrix_ci


def test_all_packages_with_tests_are_covered_by_ci():
    packages = _get_tracked_package_dirs()
    packages_with_test = set(filter(_has_package_test, packages))
    packages_tested_in_ci = _packages_in_ci_matrix()
    missing = packages_with_test - packages_tested_in_ci
    assert not missing, ("packages have a test suite but are missing from python-tests.yml's matrix", sorted(missing))
