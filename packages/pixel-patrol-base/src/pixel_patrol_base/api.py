import logging
from pathlib import Path
from typing import Union, Iterable, List, Optional, Callable
import polars as pl

from pixel_patrol_base.core.project import Project
from pixel_patrol_base.core.processing_config import ProcessingConfig
from pixel_patrol_base.core.report_config import ReportConfig
from pixel_patrol_base.io.parquet_io import load_parquet
from pixel_patrol_base.report.dashboard_app import prepare_app
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

def process_files(
    project: Project,
    processing_config: Optional[ProcessingConfig] = None,
    progress_callback: Optional[Callable[[int, int, Path], None]] = None,
) -> Project:
    """
    Process files in the project.
    
    Args:
        project: The project to process
        processing_config: Optional ProcessingConfig for slicing and processor selection.
                          If None, uses default behavior (all processors, slice all dimensions except X, Y).
        progress_callback: Optional callback function(current: int, total: int, current_file: Path) -> None
                          Called for each file processed. Useful for progress tracking in UI.
    
    Returns:
        The project with processed records_df
    """
    logger.info(f"API Call: Processing files and building DataFrame for project '{project.name}'.")
    return project.process_records(progress_callback=progress_callback, processing_config=processing_config)


def show_report(
    source: Union[Project, Path],
    host: str = "127.0.0.1",
    port: int = None,
    debug: bool = False,
    report_config: Optional[ReportConfig] = None,
) -> None:
    """
    Launch the interactive report dashboard.

    Args:
        source:         A processed Project or path to a saved .parquet file.
        host:           Host address for the server.
        port:           Port number for the server.
        debug:          Enable Flask debug mode. Note: use_reloader is always False to
                        prevent the script being executed twice.
        report_config:  Display options: colormap, widgets, initial filters/grouping.
                        If None, all widgets are shown with default settings.
    """
    app = prepare_app(source, report_config)
    logger.info(f"API Call: Showing report'.")
    app.run(debug=debug, host=host, port=port, use_reloader=False)


def export_html_report(
    source: Union[Project, Path],
    output_path: Union[str, Path],
    host: str = "127.0.0.1",
    port: int = None,
    timeout: int = 120,
    report_config: Optional[ReportConfig] = None,
) -> None:
    """
    Export the report dashboard as a static HTML file via headless browser automation.

    Args:
        source:         A processed Project or path to a saved .parquet file.
        output_path:    Path where the HTML file should be saved.
        host:           Host address for the temporary server.
        port:           Port for the temporary server (None = auto-assign).
        timeout:        Maximum seconds to wait for export.
        report_config:  Display options: colormap, widgets, initial filters/grouping.
                        If None, all widgets are shown with default settings.
    Raises:
        ImportError: If Playwright is not installed
        RuntimeError: If the export fails
    
    Example:
        >>> from pixel_patrol_base import api
        >>> from pixel_patrol_base.core.report_config import ReportConfig
        >>> api.export_html_report(
        ...     "pathA/my_project.parquet",
        ...     "pathB/report.html",
        ...     report_config=ReportConfig(
        ...         widgets_excluded={"DatasetHistogram"},
        ...         group_col="size_readable",
        ...         filter={"file_extension": {"op": "in", "value": "tif, png"}},
        ...     )
        ... )
    """
    output_path = Path(output_path)
    app = prepare_app(source, report_config)
    logger.info(f"API Call: Exporting HTML report to '{output_path}'.")
    export_html_from_dashboard(app, output_path, host=host, port=port, timeout=timeout)
    logger.info(f"API Call: HTML export completed successfully: '{output_path}'.")


def load(src: Path) -> tuple:
    """
    Load a saved project parquet file.
    Returns (records_df, metadata, project_name).

    Args:
        src: Path to the .parquet file saved by process_files.
    """
    logger.info(f"API Call: Loading project from '{src}'.")
    records_df, metadata = load_parquet(src)
    logger.info(f"API Call: Loaded '{metadata.project_name}' from '{src}'.")
    return records_df, metadata


def get_name(project: Project) -> str:
    return project.get_name()

def get_base_dir(project: Project) -> Optional[Path]:
    return project.get_base_dir()

def get_paths(project: Project) -> List[Path]:
    return project.get_paths()

def get_records_df(project: Project) -> Optional[pl.DataFrame]:
    return project.get_records_df()
