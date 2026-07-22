"""Tests for Project._save_result's part-file selection: it must save exactly the
parts the current run's writer produced, not whatever part_*.parquet files happen
to be sitting in the parts directory (e.g. stale leftovers from a failed cleanup)."""
from __future__ import annotations

import logging
from pathlib import Path

import polars as pl
import pytest

from pixel_patrol_base.core.processing_config import ProcessingConfig
from pixel_patrol_base.core.project import Project


@pytest.fixture
def project(tmp_path: Path) -> Project:
    base_dir = tmp_path / "images"
    base_dir.mkdir()
    proj = Project("test", base_dir, output_path=tmp_path / "out.parquet")
    proj.metadata = ProcessingConfig().metadata.populate_from_project(proj)
    return proj


def _write_part(path: Path, values: list[int]) -> None:
    pl.DataFrame({"file_name": [str(v) for v in values], "size_bytes": values}).write_parquet(path)


def test_stale_leftover_part_is_ignored(project: Project, tmp_path: Path, caplog):
    parts_dir = tmp_path / "_parts_out"
    parts_dir.mkdir()
    _write_part(parts_dir / "part_0000.parquet", [1, 2])
    _write_part(parts_dir / "part_0001.parquet", [3, 4])  # stale: not in this run's part_paths

    stats = {"part_paths": [str(parts_dir / "part_0000.parquet")]}
    with caplog.at_level(logging.WARNING):
        project._save_result(None, stats, parts_dir, ProcessingConfig(), [])

    assert "leftover" in caplog.text
    saved = pl.read_parquet(project.output_path)
    assert sorted(saved["size_bytes"].to_list()) == [1, 2]


def test_missing_expected_part_warns(project: Project, tmp_path: Path, caplog):
    parts_dir = tmp_path / "_parts_out"
    parts_dir.mkdir()
    _write_part(parts_dir / "part_0000.parquet", [1, 2])

    stats = {"part_paths": [
        str(parts_dir / "part_0000.parquet"),
        str(parts_dir / "part_0001.parquet"),  # expected but never made it to disk
    ]}
    with caplog.at_level(logging.WARNING):
        project._save_result(None, stats, parts_dir, ProcessingConfig(), [])

    assert "missing" in caplog.text
    saved = pl.read_parquet(project.output_path)
    assert sorted(saved["size_bytes"].to_list()) == [1, 2]


def test_no_stale_no_warning(project: Project, tmp_path: Path, caplog):
    parts_dir = tmp_path / "_parts_out"
    parts_dir.mkdir()
    _write_part(parts_dir / "part_0000.parquet", [1, 2])

    stats = {"part_paths": [str(parts_dir / "part_0000.parquet")]}
    with caplog.at_level(logging.WARNING):
        project._save_result(None, stats, parts_dir, ProcessingConfig(), [])

    assert "leftover" not in caplog.text
    assert "missing" not in caplog.text
