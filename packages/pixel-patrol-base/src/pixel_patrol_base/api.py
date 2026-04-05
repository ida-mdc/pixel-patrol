import logging
from pathlib import Path
from typing import Union, Iterable, List, Optional, Callable, Set, Dict
import polars as pl

from pixel_patrol_base.core.project import Project
from pixel_patrol_base.core.processing_config import ProcessingConfig
from pixel_patrol_base.core.project_metadata import ProjectMetadata
from pixel_patrol_base.core.report_config import ReportConfig
from pixel_patrol_base.io.parquet_io import load_parquet
from pixel_patrol_base.report.dashboard_app import prepare_app
from pixel_patrol_base.report.html_export import export_html_from_dashboard


logger = logging.getLogger(__name__)

def create_project(name: str,
                   base_dir: Union[str, Path],
                   loader: str = None,
                   output_path: Optional[Union[str, Path]] = None
                   ) -> Project:
    logger.info(f"API Call: Creating new project '{name}' with base directory '{base_dir}'.")
    return Project(name, base_dir, loader, output_path)

def add_paths(project: Project, paths: Union[str, Path, Iterable[Union[str, Path]]]) -> Project:
    logger.info(f"API Call: Adding paths to project '{project.name}'.")
    return project.add_paths(paths)

def delete_path(project: Project, path: str) -> Project:
    logger.info(f"API Call: deleting paths from project '{project.name}'.")
    return project.delete_path(path)


def process_files(
        project: Project,
        progress_callback: Optional[Callable[[int, int, Path], None]] = None,
        # --- Processor selection ---
        processors_included: Optional[Set[str]] = None,
        processors_excluded: Optional[Set[str]] = None,
        # --- File selection ---
        selected_file_extensions: Union[Set[str], str, None] = None,
        # --- Run behaviour ---
        processing_max_workers: Optional[int] = None,
        records_flush_every_n: Optional[int] = None,
        # --- Metadata ---
        flavor: Optional[str] = None,
        authors: Optional[str] = None,
) -> Project:
    """
    Process files in the project.

    Args:
        project:                    The project to process.
        progress_callback:          Optional callback(current, total, current_file) -> None,
                                    called for each file processed.
        processors_included:        Only run these processors (e.g. {"basic-stats"}).
                                    If set, processors_excluded is ignored.
        processors_excluded:        Exclude these processors (e.g. {"histogram"}).
        selected_file_extensions:   Extensions to process, e.g. {"tif", "png"}, or "all".
                                    Defaults to "all".
        processing_max_workers:     Thread-pool size. None = default.
        records_flush_every_n:      Flush intermediate results to disk every N records.
        flavor:                     Config flavour label embedded in the parquet metadata.
        authors:                    Free-form authors string embedded in the parquet metadata.

    Returns:
        The project with processed records_df.
    """
    processing_config = ProcessingConfig(
        processors_included=processors_included or set(),
        processors_excluded=processors_excluded or set(),
        selected_file_extensions=selected_file_extensions or "all",
        processing_max_workers=processing_max_workers,
        records_flush_every_n=records_flush_every_n,
        metadata=ProjectMetadata(
            flavor=flavor or "",
            authors=authors or "",
        ),
    )
    logger.info(f"API Call: Processing files and building DataFrame for project '{project.name}'.")
    return project.process_records(progress_callback=progress_callback, processing_config=processing_config)


def show_report(
    source: Union[Project, Path],
    host: str = "127.0.0.1",
    port: int = None,
    debug: bool = False,
    cmap: Optional[str] = None,
    widgets_included: Optional[Set[str]] = None,
    widgets_excluded: Optional[Set[str]] = None,
    group_col: Optional[str] = None,
    filter_by: Optional[Dict] = None,
    dimensions: Optional[Dict[str, str]] = None,
    is_show_significance: bool = False,
) -> None:
    """
    Launch the interactive report dashboard.

    Args:
        source:                 A processed Project or path to a saved .parquet file.
        host:                   Host address for the server.
        port:                   Port number for the server.
        debug:                  Enable Flask debug mode. Note: use_reloader is always False to
                                prevent the script being executed twice.
        cmap:                   Colormap name for visualizations.
        widgets_included:       Only show these widgets (by NAME). If set, widgets_excluded is ignored.
        widgets_excluded:       Exclude these widgets (by NAME).
        group_col:              Column name to group by.
        filter_by:              Filter dict, e.g. {"file_extension": {"op": "in", "value": "tif, png"}}.
        dimensions:             Dimension filters, e.g. {"T": "0", "Z": "1"}.
        is_show_significance:   Whether to show statistical significance annotations.
    """
    report_config = ReportConfig.from_kwargs(
        cmap=cmap, widgets_included=widgets_included, widgets_excluded=widgets_excluded,
        group_col=group_col, filter_by=filter_by, dimensions=dimensions,
        is_show_significance=is_show_significance,
    )
    app = prepare_app(source, report_config)
    logger.info("API Call: Showing report.")
    app.run(debug=debug, host=host, port=port, use_reloader=False)


def export_html_report(
    source: Union[Project, Path],
    output_path: Union[str, Path],
    host: str = "127.0.0.1",
    port: int = None,
    timeout: int = 120,
    cmap: Optional[str] = None,
    widgets_included: Optional[Set[str]] = None,
    widgets_excluded: Optional[Set[str]] = None,
    group_col: Optional[str] = None,
    filter_by: Optional[Dict] = None,
    dimensions: Optional[Dict[str, str]] = None,
    is_show_significance: bool = False,
) -> None:
    """
    Export the report dashboard as a static HTML file via headless browser automation.

    Args:
        source:                 A processed Project or path to a saved .parquet file.
        output_path:            Path where the HTML file should be saved.
        host:                   Host address for the temporary server.
        port:                   Port for the temporary server (None = auto-assign).
        timeout:                Maximum seconds to wait for export.
        cmap:                   Colormap name for visualizations.
        widgets_included:       Only show these widgets (by NAME). If set, widgets_excluded is ignored.
        widgets_excluded:       Exclude these widgets (by NAME).
        group_col:              Column name to group by.
        filter_by:              Filter dict, e.g. {"file_extension": {"op": "in", "value": "tif, png"}}.
        dimensions:             Dimension filters, e.g. {"T": "0", "Z": "1"}.
        is_show_significance:   Whether to show statistical significance annotations.
    Raises:
        ImportError: If Playwright is not installed
        RuntimeError: If the export fails

    Example:
        >>> from pixel_patrol_base import api
        >>> api.export_html_report(
        ...     "pathA/my_project.parquet",
        ...     "pathB/report.html",
        ...     widgets_excluded={"DatasetHistogram"},
        ...     group_col="size_readable",
        ...     filter_by={"file_extension": {"op": "in", "value": "tif, png"}},
        ... )
    """
    output_path = Path(output_path)
    report_config = ReportConfig.from_kwargs(
        cmap=cmap, widgets_included=widgets_included, widgets_excluded=widgets_excluded,
        group_col=group_col, filter_by=filter_by, dimensions=dimensions,
        is_show_significance=is_show_significance,
    )
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
