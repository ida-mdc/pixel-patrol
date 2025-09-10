import logging
from pathlib import Path

from pixel_patrol_base.api import (
    create_project,
    set_settings,
    process_artifacts,
    export_project, show_report, add_paths,
)
from pixel_patrol_base.core.project_settings import Settings

logger = logging.getLogger(__name__)

if __name__ == "__main__":

    # Base folder containing multiple test-image subdirectories
    base_path = Path(__file__).resolve().parent / "data"

    # Define the output directory for the exported project
    output_directory = Path(__file__).parent / "exported_projects"
    output_directory.mkdir(parents=True, exist_ok=True)

    # Define the path for the exported zip file
    exported_project_path = output_directory / "report_data.zip"

    # Initialize project with the base directory
    project = create_project("Markdown Diary", base_path, loader="markdown-diary")

    project = add_paths(project, ["2024", "2025"])

    # Configure image-processing settings
    settings = Settings(
        cmap="viridis",
        selected_file_extensions="all",
        pixel_patrol_flavor="moody edition"
    )
    project = set_settings(project, settings)

    project = process_artifacts(project)

    export_project(project, exported_project_path)

    show_report(project)