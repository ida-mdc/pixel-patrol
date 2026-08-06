from __future__ import annotations

import pytest
import sys
from pathlib import Path
from typing import Any, Union, Iterable

from click.testing import CliRunner

local_src = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(local_src))

sys.modules.pop("pixel_patrol_base", None)
from pixel_patrol_base import cli as cli_module


def _setup_fake_cli(monkeypatch, dataset_root: Path, create_output: bool = False):
    """Wire the CLI entry points to lightweight fakes for deterministic assertions."""
    dummy_project: dict[str, Any] = {}
    added_paths: list[Path] = []
    view_calls: list[Path] = []

    def fake_create_project(name: str, base_dir: str, loader: str | None = None, output_path: str | Path | None = None):
        assert Path(base_dir).resolve() == dataset_root.resolve()
        dummy_project["name"] = name
        dummy_project["output_path"] = Path(output_path) if output_path else None
        return dummy_project

    def fake_add_paths(project, paths: Union[str, Path, Iterable[Union[str, Path]]]):
        assert project is dummy_project

        paths_to_process = [paths] if not isinstance(paths, (tuple, list)) else paths

        for p_input in paths_to_process:
            candidate_path = Path(p_input)

            if candidate_path.is_absolute():
                resolved_path = candidate_path.resolve()
            else:
                resolved_path = (dataset_root / candidate_path).resolve()

            added_paths.append(resolved_path)

        return project

    def fake_process_files(project, **kwargs):
        if create_output and dummy_project.get("output_path"):
            dummy_project["output_path"].touch()
        return project

    def fake_view(path, **kwargs):
        view_calls.append(path)

    monkeypatch.setattr(cli_module, "create_project", fake_create_project)
    monkeypatch.setattr(cli_module, "add_paths", fake_add_paths)
    monkeypatch.setattr(cli_module, "process_files", fake_process_files)
    monkeypatch.setattr(cli_module, "api_view", fake_view)

    return added_paths, view_calls


def test_process_accepts_relative_base_and_paths(monkeypatch, tmp_path):
    runner = CliRunner()

    dataset_root = tmp_path / "dataset"
    pngs_dir = dataset_root / "pngs"
    tifs_dir = dataset_root / "tifs"
    pngs_dir.mkdir(parents=True)
    tifs_dir.mkdir()

    output_path = tmp_path / "out.parquet"

    added_paths, _ = _setup_fake_cli(monkeypatch, dataset_root)

    monkeypatch.chdir(dataset_root.parent)

    result = runner.invoke(
        cli_module.cli,
        [
            "process",
            "dataset",
            "-o",
            str(output_path),
            "-p",
            "pngs",
            "-p",
            "tifs",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    assert added_paths == [pngs_dir.resolve(), tifs_dir.resolve()]


def test_process_accepts_absolute_paths(monkeypatch, tmp_path):
    runner = CliRunner()

    dataset_root = tmp_path / "dataset"
    pngs_dir = dataset_root / "pngs"
    tifs_dir = dataset_root / "tifs"
    pngs_dir.mkdir(parents=True)
    tifs_dir.mkdir()

    output_path = tmp_path / "out.parquet"

    added_paths, _ = _setup_fake_cli(monkeypatch, dataset_root)

    result = runner.invoke(
        cli_module.cli,
        [
            "process",
            str(dataset_root),
            "-o",
            str(output_path),
            "-p",
            str(pngs_dir),
            "-p",
            str(tifs_dir),
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    assert added_paths == [pngs_dir.resolve(), tifs_dir.resolve()]


def test_process_view_flag_opens_viewer(monkeypatch, tmp_path):
    runner = CliRunner()
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    output_path = tmp_path / "out.parquet"

    _, view_calls = _setup_fake_cli(monkeypatch, dataset_root, create_output=True)

    result = runner.invoke(
        cli_module.cli,
        ["process", str(dataset_root), "-o", str(output_path), "--view"],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    assert view_calls == [output_path]


def test_process_view_flag_skipped_when_no_output(monkeypatch, tmp_path):
    runner = CliRunner()
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    output_path = tmp_path / "out.parquet"

    _, view_calls = _setup_fake_cli(monkeypatch, dataset_root, create_output=False)

    result = runner.invoke(
        cli_module.cli,
        ["process", str(dataset_root), "-o", str(output_path), "--view"],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    assert view_calls == []


def test_process_view_flag_corrects_missing_extension(monkeypatch, tmp_path):
    """--output without .parquet extension should still open the viewer with the corrected path."""
    runner = CliRunner()
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    output_no_ext = tmp_path / "out"  # no .parquet
    output_corrected = tmp_path / "out.parquet"

    _, view_calls = _setup_fake_cli(monkeypatch, dataset_root, create_output=False)
    # fake_process_files won't create the file; touch it manually to simulate the corrected path
    output_corrected.touch()

    result = runner.invoke(
        cli_module.cli,
        ["process", str(dataset_root), "-o", str(output_no_ext), "--view"],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    assert view_calls == [output_corrected]


