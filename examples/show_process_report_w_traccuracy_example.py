#!/usr/bin/env python
import logging
from pathlib import Path

from pixel_patrol.api import (
    show_report, create_project,
)
from pixel_patrol.core.project_settings import Settings

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    base_path = Path(__file__).resolve().parent / "data" / "traccuracy_generated"
    project = create_project("project", base_path)
    # project.add_paths("group1")
    # project.add_paths("group2")
    # project.add_paths("group3")
    project.process_images(settings=Settings(selected_file_extensions="all"))

    # print(project.images_df.get_columns())

    # Launch the interactive report
    logger.info("Opening report in your browser...")
    show_report(project, port=8052)

