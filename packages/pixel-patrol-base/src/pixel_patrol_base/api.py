import logging
from datetime import datetime
from pathlib import Path
from typing import Union, Iterable, List, Optional, Callable, Set, Dict
import polars as pl

from pixel_patrol_base.core.project import Project
from pixel_patrol_base.core.processing_config import ProcessingConfig
from pixel_patrol_base.core.project_metadata import ProjectMetadata
from pixel_patrol_base.io.parquet_io import load_parquet
from pixel_patrol_base.viewer_pages import build_single_file_viewer_html, build_github_pages_site
from pixel_patrol_base.viewer_server import serve_viewer


logger = logging.getLogger(__name__)

def create_project(name: str,
                   base_dir: Union[str, Path],
                   loader: Optional[str] = None,
                   output_path: Optional[Union[str, Path]] = None
                   ) -> Project:
    logger.debug(f"API Call: Creating new project '{name}' with base directory '{base_dir}'.")
    return Project(name, base_dir, loader, output_path)

def add_paths(project: Project, paths: Union[str, Path, Iterable[Union[str, Path]]]) -> Project:
    logger.debug(f"API Call: Adding paths to project '{project.name}'.")
    return project.add_paths(paths)

def delete_path(project: Project, path: str) -> Project:
    logger.debug(f"API Call: deleting paths from project '{project.name}'.")
    return project.delete_path(path)


def process_files(
        project: Project,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        # --- Processor selection ---
        processors_included: Optional[Set[str]] = None,
        processors_excluded: Optional[Set[str]] = None,
        # --- File selection ---
        selected_file_extensions: Union[Set[str], str, None] = None,
        # --- Run behaviour ---
        max_workers: Optional[int] = None,
        mb_per_task: Optional[float] = None,
        max_images_per_task: Optional[int] = None,
        slice_size: Optional[Dict[str, int]] = None,
        rows_per_part: Optional[int] = None,
        parquet_row_group_size: Optional[int] = None,
        # --- Metadata ---
        flavor: Optional[str] = None,
        description: Optional[str] = None,
        # --- Logging ---
        log_file: bool = False,
) -> Project:
    """
    Process files in the project and write the .parquet report.

    Args:
        project:                    The project to process.
        progress_callback:          Optional callback(done: int, total: int) called per
                                    completed record. total is -1 until the full count is known.
        processors_included:        Run only these processor IDs, e.g. {"raster-basic", "thumbnail"}.
                                    If set, processors_excluded is ignored.
        processors_excluded:        Skip these processor IDs, e.g. {"raster-quality"}.
        selected_file_extensions:   Extensions to process, e.g. {"tif", "nd2"}, or "all".
                                    Defaults to "all".
        max_workers:                Dask worker count. None = auto-detect from CPUs and RAM.
                                    Use 1 to disable parallelism.
        mb_per_task:                Work budget per Dask task in MB (default: 512). Controls
                                    batch sizes for small files, spatial splitting for large files,
                                    and sub-image batching for container files.
        max_images_per_task:        Max files per batch task or sub-images per container task
                                    (default: 200).
        slice_size:                 Leaf block size per dimension for leaf processors,
                                    e.g. {"Z": 1, "Y": 256}. Controls the granularity of
                                    per-dimension statistics in the output. By default X and Y
                                    are full extent and all other dims step by 1. None = default.
        rows_per_part:              Rows buffered before flushing to a temporary file (default: 10000).
        parquet_row_group_size:     Rows per row group in the final parquet (default: 2048).
                                    Smaller values speed up thumbnail sampling in the viewer.
        flavor:                     Label shown next to the title in the viewer.
        description:                Free-form description embedded in the report metadata.
        log_file:                   Write a DEBUG-level log file alongside the output parquet.
                                    INFO and WARNING still appear in the terminal as usual.

    Returns:
        The project with processed records_df.
    """
    if log_file:
        _setup_file_logging(project)
    config_kwargs = {}
    if rows_per_part is not None:
        config_kwargs["rows_per_part"] = rows_per_part
    if mb_per_task is not None:
        config_kwargs["mb_per_task"] = mb_per_task
    if max_images_per_task is not None:
        config_kwargs["max_images_per_task"] = max_images_per_task
    if slice_size is not None:
        config_kwargs["slice_size"] = slice_size
    processing_config = ProcessingConfig(
        processors_included=processors_included or set(),
        processors_excluded=processors_excluded or set(),
        selected_file_extensions=selected_file_extensions or "all",
        max_workers=max_workers,
        parquet_row_group_size=parquet_row_group_size,
        metadata=ProjectMetadata(
            flavor=flavor or "",
            description=description or "",
        ),
        **config_kwargs,
    )
    logger.debug(f"API Call: Processing files and building DataFrame for project '{project.name}'.")
    return project.process_records(progress_callback=progress_callback, processing_config=processing_config)


def view(
        source: Union[Project, Path, str],
        port: int = 8052,
        open_browser: bool = True,
        group_col: Optional[str] = None,
        filter_by: Optional[Dict] = None,
        dimensions: Optional[Dict[str, str]] = None,
        widgets_excluded: Optional[Set[str]] = None,
        is_show_significance: bool = False,
        palette: Optional[str] = None,
) -> None:
    """
    Open a parquet file in the Pixel Patrol viewer backed by a local Python server.

    Args:
        source:               A processed Project or path to a .parquet file.
        port:                 Port for the local server (default 8052).
        open_browser:         Open the viewer in the default browser (default True).
        group_col:            Column to group by on first load.
        filter_by:            Initial filter, e.g. {"file_extension": {"op": "in", "value": "tif,png"}}.
        dimensions:           Dimension selections, e.g. {"t": "0", "z": "1"}.
        widgets_excluded:     Set of plugin IDs to hide, e.g. {"histogram", "summary"}.
        is_show_significance: Show statistical significance brackets.
        palette:              Color palette name (default "tab10").

    Example:
        >>> from pixel_patrol_base import api
        >>> api.view(
        ...     "my_project.parquet",
        ...     group_col="file_extension",
        ...     filter_by={"file_extension": {"op": "in", "value": "tif,png"}},
        ...     dimensions={"t": "0"},
        ...     widgets_excluded={"histogram"},
        ... )
    """

    path = _resolve_parquet_path(source)
    serve_viewer(
        path,
        port=port,
        open_browser=open_browser,
        group_col=group_col,
        filter_by=filter_by,
        dimensions=dimensions,
        widgets_excluded=widgets_excluded,
        is_show_significance=is_show_significance,
        palette=palette,
    )


def build_viewer(output: Union[str, Path]) -> Path:
    """
    Build a static viewer from the installed web viewer bundle.

    If OUTPUT ends in .html or .htm, writes a single self-contained HTML file
    with all JS/CSS/extensions inlined — share it alongside your .parquet file.

    Otherwise OUTPUT is treated as a directory and a GitHub Pages-style site
    is written there (index.html + assets + extensions folders) — deploy to
    any static host and open with a ?data= URL pointing to your parquet.

    Args:
        output: Path to a .html file or a directory.

    Returns:
        Path to the written file or directory.

    Examples:
        >>> from pixel_patrol_base import api
        >>> api.build_viewer("viewer.html")        # single file
        # OR
        >>> api.build_viewer("gh-pages-out/")      # site folder
    """

    output = Path(output)
    if output.suffix.lower() in {".html", ".htm"}:
        out = build_single_file_viewer_html(output)
        logger.info("Single-file viewer written to: '%s'", out)
    else:
        out = build_github_pages_site(output)
        logger.info("Static viewer site written to: '%s'", out)
    return out


def load(src: Path) -> tuple:
    """
    Load a saved project parquet file.
    Returns (records_df, metadata).

    Args:
        src: Path to the .parquet file saved by process_files.
    """
    logger.debug(f"API Call: Loading project from '{src}'.")
    records_df, metadata = load_parquet(src)
    logger.debug(f"API Call: Loaded '{metadata.project_name}' from '{src}'.")
    return records_df, metadata


def get_name(project: Project) -> str:
    return project.get_name()

def get_base_dir(project: Project) -> Optional[Path]:
    return project.get_base_dir()

def get_paths(project: Project) -> List[Path]:
    return project.get_paths()

def get_records_df(project: Project) -> Optional[pl.DataFrame]:
    return project.get_records_df()


def _setup_file_logging(project: Project) -> None:
    """Attach a DEBUG-level FileHandler to the root logger, adjacent to the output parquet.

    Existing handlers (e.g. the terminal StreamHandler) are kept at INFO so
    terminal output is unchanged — only the file receives DEBUG messages.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = project.output_path.with_name(f"{project.output_path.stem}_{ts}.log")
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    for h in root.handlers:
        if not isinstance(h, logging.FileHandler):
            h.setLevel(logging.INFO)
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(name)s: %(message)s", datefmt="%H:%M:%S"))
    root.addHandler(fh)
    logger.info("Debug log → '%s'", log_path)


def _resolve_parquet_path(source: Union[Project, Path, str]) -> Path:
    if isinstance(source, Project):
        p = source.get_output_path()
        if p is None:
            raise ValueError(
                "Project has no output path. "
                "Pass an explicit parquet path or call process_files() first."
            )
        path = Path(p).resolve()
    else:
        path = Path(source).resolve()

    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")
    if path.suffix.lower() != ".parquet":
        raise ValueError(f"Expected a .parquet file, got: {path}")
    return path