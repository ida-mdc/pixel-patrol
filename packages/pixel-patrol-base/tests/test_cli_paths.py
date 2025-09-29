from __future__ import annotations

import pytest
import sys
from pathlib import Path
from typing import Any, Union, Iterable  # Added Union and Iterable

from click.testing import CliRunner

local_src = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(local_src))

sys.modules.pop("pixel_patrol_base", None)
from pixel_patrol_base import cli as cli_module


def _setup_fake_cli(monkeypatch, dataset_root: Path):
    """Wire the CLI entry points to lightweight fakes for deterministic assertions."""
    dummy_project: dict[str, Any] = {}
    added_paths: list[Path] = []
    exported_to: list[Path] = []

    def fake_create_project(name: str, base_dir: str, loader: str | None = None):
        assert Path(base_dir).resolve() == dataset_root.resolve()
        dummy_project["name"] = name
        return dummy_project

    def fake_add_paths(project, paths: Union[str, Path, Iterable[Union[str, Path]]]):
        assert project is dummy_project

        # Standardize input to an iterable (handles single Path/string vs. tuple from CLI)
        paths_to_process = [paths] if not isinstance(paths, (tuple, list)) else paths

        for p_input in paths_to_process:
            candidate_path = Path(p_input)

            if candidate_path.is_absolute():
                # Handles paths that were already absolute (e.g., in test_export_accepts_absolute_paths)
                resolved_path = candidate_path.resolve()
            else:
                # Handles relative path strings (e.g., 'pngs', 'tifs') by resolving them against the base dir
                resolved_path = (dataset_root / candidate_path).resolve()

            added_paths.append(resolved_path)

        return project

    ##

    def fake_set_settings(project, settings):
        return project

    def fake_process_files(project):
        return project

    def fake_export_project(project, destination: Path):
        exported_to.append(Path(destination).resolve())

    monkeypatch.setattr(cli_module, "create_project", fake_create_project)
    monkeypatch.setattr(cli_module, "add_paths", fake_add_paths)
    monkeypatch.setattr(cli_module, "set_settings", fake_set_settings)
    monkeypatch.setattr(cli_module, "process_files", fake_process_files)
    monkeypatch.setattr(cli_module, "export_project", fake_export_project)

    return added_paths, exported_to


def test_export_accepts_relative_base_and_paths(monkeypatch, tmp_path):
    runner = CliRunner()

    dataset_root = tmp_path / "dataset"
    pngs_dir = dataset_root / "pngs"
    tifs_dir = dataset_root / "tifs"
    pngs_dir.mkdir(parents=True)
    tifs_dir.mkdir()

    added_paths, exported_to = _setup_fake_cli(monkeypatch, dataset_root)

    monkeypatch.chdir(dataset_root.parent)

    result = runner.invoke(
        cli_module.cli,
        [
            "export",
            "dataset",
            "-o",
            "out.zip",
            "-p",
            "pngs",
            "-p",
            "tifs",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    assert added_paths == [pngs_dir.resolve(), tifs_dir.resolve()]
    assert exported_to == [Path("out.zip").resolve()]


def test_export_accepts_absolute_paths(monkeypatch, tmp_path):
    runner = CliRunner()

    dataset_root = tmp_path / "dataset"
    pngs_dir = dataset_root / "pngs"
    tifs_dir = dataset_root / "tifs"
    pngs_dir.mkdir(parents=True)
    tifs_dir.mkdir()

    added_paths, exported_to = _setup_fake_cli(monkeypatch, dataset_root)

    output_zip = tmp_path / "report.zip"

    result = runner.invoke(
        cli_module.cli,
        [
            "export",
            str(dataset_root),
            "-o",
            str(output_zip),
            "-p",
            str(pngs_dir),
            "-p",
            str(tifs_dir),
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    assert added_paths == [pngs_dir.resolve(), tifs_dir.resolve()]
    assert exported_to == [output_zip.resolve()]