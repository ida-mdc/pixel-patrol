#!/usr/bin/env python
import logging
from pathlib import Path

from pixel_patrol_base.api import (
    create_project,
    add_paths,
    process_paths,
    set_settings,
    process_images,
    get_images_df,
    show_report, import_project,
)
from pixel_patrol_base.core.project_settings import Settings

# — Configure Logging —
logging.basicConfig(
    level=logging.INFO,
    format="%((asctime)s)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # Base folder containing multiple test-image subdirectories
    exported_project_path = Path(__file__).parent / "exported_projects" / "zarr_exported.zip"

    project = import_project(exported_project_path)

    # print(project.images_df.get_columns())

    # Launch the interactive report
    logger.info("Opening report in your browser...")
    show_report(project, port=8051)
