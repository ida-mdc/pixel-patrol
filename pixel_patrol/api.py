from pathlib import Path
from typing import Union, Iterable, List, Optional, Dict
import polars as pl
import logging

from pixel_patrol.core.project import Project
from pixel_patrol.core.project_settings import Settings
from pixel_patrol.io.project_io import export_project as _io_export_project
from pixel_patrol.io.project_io import import_project as _io_import_project
from pixel_patrol.core import processing
from pixel_patrol.report.dashboard_app import create_app


logger = logging.getLogger(__name__)

def create_project(name: str, base_dir: Union[str, Path]) -> Project:
    logger.info(f"API Call: Creating new project '{name}' with base directory '{base_dir}'.")
    return Project(name, base_dir)

def add_paths(project: Project, paths: Union[str, Path, Iterable[Union[str, Path]]]) -> Project:
    logger.info(f"API Call: Adding paths to project '{project.name}'.")
    return project.add_paths(paths)

def delete_path(project: Project, path: str) -> Project:
    logger.info(f"API Call: deleting paths from project '{project.name}'.")
    return project.delete_path(path)

def process_paths(project: Project) -> Project:
    logger.info(f"API Call: Processing paths and building paths DataFrame for project '{project.name}'.")
    return project.process_paths()

def set_settings(project: Project, settings: Settings) -> Project:
    """
    Sets the project-specific settings by replacing the entire Settings object.
    Detailed validation for individual settings is performed within the Project class itself.
    Args:
        project: The Project instance to update.
        settings: An instance of the Settings dataclass containing the desired settings.
    """
    logger.info(f"API Call: Attempting to set project settings for '{project.name}'.")
    return project.set_settings(settings)

def process_images(project: Project) -> Project:
    logger.info(f"API Call: Processing images and building images DataFrame for project '{project.name}'.")
    return project.process_images()

def show_report(project: Project, host: str = "127.0.0.1", port: int = None) -> None:
    logger.info(f"API Call: Showing report for project '{project.name}'.")
    app = create_app(project)
    app.run(debug=True, host=host, port=port)

def export_project(project: Project, dest: Path) -> None: # TODO: think about when project can be saved
    logger.info(f"API Call: Exporting project '{project.name}' to '{dest}'.")
    _io_export_project(project, dest)
    logger.info(f"API Call: Project '{project.name}' exported successfully.")

def import_project(src: Path) -> Project: # TODO: think about when project can be saved
    logger.info(f"API Call: Importing project from '{src}'.")
    project = _io_import_project(src)
    logger.info(f"API Call: Project '{project.name}' imported successfully from '{src}'.")
    return project

def get_name(project: Project) -> str:
    return project.get_name()

def get_base_dir(project: Project) -> Optional[Path]:
    return project.get_base_dir()

def get_paths(project: Project) -> List[Path]:
    return project.get_paths()

def get_paths_df(project: Project) -> Optional[pl.DataFrame]: # TODO: can we get rid of polars import?
    return project.get_paths_df()

def get_settings(project: Project) -> Settings:
    return project.get_settings()

def get_available_extensions(project: Project) -> Dict[str, int]:
    logger.info(f"API Call: Getting all file extensions for all file paths in all paths of project '{project.name}'.")
    return processing.count_file_extensions(project.paths_df)

def get_images_df(project: Project) -> Optional[pl.DataFrame]:
    return project.get_images_df()
