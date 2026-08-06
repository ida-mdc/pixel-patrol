"""Unit tests for _log_processing_summary's 'Done:' line."""
from __future__ import annotations

import logging

from pixel_patrol_base.core.project import _log_processing_summary


def test_shows_only_file_count_when_no_containers(caplog):
    stats = {"wall_s": 10.0, "n_files": 5, "n_images_processed": 5, "n_images_failed": 0}
    with caplog.at_level(logging.INFO):
        _log_processing_summary("proj", stats)
    assert "5 files in" in caplog.text
    assert "images" not in caplog.text


def test_shows_files_x_images_when_containers_present(caplog):
    stats = {"wall_s": 10.0, "n_files": 69, "n_images_processed": 408, "n_images_failed": 9}
    with caplog.at_level(logging.INFO):
        _log_processing_summary("proj", stats)
    assert "69 files x 417 images in" in caplog.text
