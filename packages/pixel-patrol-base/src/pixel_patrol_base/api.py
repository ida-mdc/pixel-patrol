import logging
from pathlib import Path
from typing import Union, Iterable, List, Optional, Callable

import polars as pl

from pixel_patrol_base.core.project import Project
from pixel_patrol_base.core.project_settings import Settings
from pixel_patrol_base.core.processing_config import ProcessingConfig
from pixel_patrol_base.core.report_config import ReportConfig
from pixel_patrol_base.io.project_io import export_project as _io_export_project
from pixel_patrol_base.io.project_io import import_project as _io_import_project
from pixel_patrol_base.report.dashboard_app import create_app
from pixel_patrol_base.report.global_controls import init_global_config
from pixel_patrol_base.report.html_export import export_html_from_dashboard

logger = logging.getLogger(__name__)

def create_project(name: str, base_dir: Union[str, Path], loader: str = None) -> Project:
    logger.info(f"API Call: Creating new project '{name}' with base directory '{base_dir}'.")
    return Project(name, base_dir, loader)

def add_paths(project: Project, paths: Union[str, Path, Iterable[Union[str, Path]]]) -> Project:
    logger.info(f"API Call: Adding paths to project '{project.name}'.")
    return project.add_paths(paths)

def delete_path(project: Project, path: str) -> Project:
    logger.info(f"API Call: deleting paths from project '{project.name}'.")
    return project.delete_path(path)

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

def process_files(
    project: Project, 
    progress_callback: Optional[Callable[[int, int, Path], None]] = None,
    processing_config: Optional[ProcessingConfig] = None
) -> Project:
    """
    Process files in the project.
    
    Args:
        project: The project to process
        progress_callback: Optional callback function(current: int, total: int, current_file: Path) -> None
                          Called for each file processed. Useful for progress tracking in UI.
        processing_config: Optional ProcessingConfig for slicing and processor selection.
                          If None, uses default behavior (all processors, slice all dimensions except X, Y).
    
    Returns:
        The project with processed records_df
    """
    logger.info(f"API Call: Processing files and building DataFrame for project '{project.name}'.")
    return project.process_records(progress_callback=progress_callback, processing_config=processing_config)

def show_report(
    project: Project,
    host: str = "127.0.0.1",
    port: int = None,
    debug: bool = False,
    report_config: Optional[ReportConfig] = None,
) -> None:
    """
    Run without the Flask debug reloader by default. The debug reloader
    spawns a second process and will re-import/run the script, which causes
    the example to execute twice (scan/process files two times). When a
    developer needs the interactive debugger they can pass `debug=True`,
    but should also set `use_reloader=False` if they do not want the script
    re-executed.
    
    Args:
        project: The project to show report for
        host: Host address for the server
        port: Port number for the server
        debug: Enable debug mode
        report_config: Optional ReportConfig for widget selection and global filters/grouping.
                     If None, all widgets are shown and default filters/grouping are used.
    """
    logger.info(f"API Call: Showing report for project '{project.name}'.")
    global_config_dict = report_config.to_dict() if report_config else None
    sanitized = init_global_config(project.records_df, global_config_dict)
    app = create_app(project, initial_global_config=sanitized, report_config=report_config)
    app.run(debug=debug, host=host, port=port, use_reloader=False)

def export_html_report(
    project: Project,
    output_path: Union[str, Path],
    host: str = "127.0.0.1",
    port: int = None,
    timeout: int = 120,
    report_config: Optional[ReportConfig] = None,
) -> None:
    """
    Export the report dashboard as a static HTML file using headless browser automation.
    
    This function creates a Dash app from the project, launches it in a background server,
    uses Playwright to render the page, and saves the result as a static HTML file.
    
    Args:
        project: The Project instance to export
        output_path: Path where the HTML file should be saved (str or Path)
        host: Host address for the server (default: "127.0.0.1")
        port: Port number for the server (default: None, auto-assigned)
        timeout: Maximum time to wait for the export in seconds (default: 120)
        report_config: Optional ReportConfig for widget selection and global filters/grouping.
                     If None, all widgets are shown and default filters/grouping are used.
    
    Raises:
        ImportError: If Playwright is not installed
        RuntimeError: If the export fails
    
    Example:
        >>> from pixel_patrol_base import api
        >>> from pixel_patrol_base.core.report_config import ReportConfig
        >>> project = api.import_project("my_project.zip")
        >>> api.export_html_report(
        ...     project,
        ...     "report.html",
        ...     report_config=ReportConfig(
        ...         widgets_excluded={"EmbeddingProjectorWidget"},
        ...         group_col="size_readable",
        ...         filter={"file_extension": {"op": "in", "value": "tif, png"}},
        ...     )
        ... )
    """
    output_path = Path(output_path)
    logger.info(f"API Call: Exporting HTML report for project '{project.name}' to '{output_path}'.")
    global_config_dict = report_config.to_dict() if report_config else None
    sanitized = init_global_config(project.records_df, global_config_dict)
    app = create_app(project, initial_global_config=sanitized, report_config=report_config)
    export_html_from_dashboard(app, output_path, host=host, port=port, timeout=timeout)
    logger.info(f"API Call: HTML export completed successfully: '{output_path}'.")

def export_project(project: Project, dest: Path) -> None: # TODO: think about when project can be saved
    logger.info(f"API Call: Exporting project '{project.name}' to '{dest}'.")
    # If a records_flush_dir was not set explicitly, default to a batches directory
    # next to the destination ZIP. This matches CLI behavior and ensures
    # intermediate chunks (for large runs) are placed where the final ZIP will live.
    if getattr(project.settings, "records_flush_dir", None) is None:
        inferred = dest.parent / f"{project.name}_batches"
        try:
            project.settings.records_flush_dir = inferred
            logger.info("API Call: inferred records_flush_dir=%s", inferred)
        except Exception:
            logger.debug("API Call: could not set inferred records_flush_dir=%s", inferred)

    _io_export_project(project, dest)
    logger.info(f"API Call: Project '{project.name}' exported successfully.")

def import_project(src: Path) -> Project:
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

def get_settings(project: Project) -> Settings:
    return project.get_settings()

def get_records_df(project: Project) -> Optional[pl.DataFrame]:
    return project.get_records_df()
