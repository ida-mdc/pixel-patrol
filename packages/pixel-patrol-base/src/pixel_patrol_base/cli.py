import os
import shutil
import threading
import time
import webbrowser
from pathlib import Path
from threading import Timer

import click

from pixel_patrol_base.api import (
    create_project,
    add_paths,
    set_settings,
    process_files,
    export_project,
    import_project,
    export_html_report,
)
from pixel_patrol_base.core.project_settings import Settings
from pixel_patrol_base.core.processing import _cleanup_partial_chunks_dir
from pixel_patrol_base.report.constants import NO_GROUPING_COL


def _serve_in_window(
    serve_fn,
    url: str,
    title: str = "Pixel Patrol",
    width: int = 1440,
    height: int = 900,
) -> None:
    """Start *serve_fn* in a background thread and open a native webview window.

    Falls back to the system browser when pywebview is not installed or when
    running in a headless / remote-host context (host != 127.0.0.1).
    """
    try:
        import os as _os
        _os.environ.setdefault("PYWEBVIEW_GUI", "qt")
        _os.environ.setdefault("QT_API", "pyside6")
        import webview  # type: ignore
        _has_webview = True
    except ImportError:
        _has_webview = False

    if _has_webview:
        t = threading.Thread(target=serve_fn, daemon=True)
        t.start()
        time.sleep(0.8)  # Give Flask/Dash a moment to bind the port

        _icon_path = Path(__file__).parent / "report" / "assets" / "icon.png"

        def _set_icon():
            try:
                from PySide6.QtGui import QIcon
                from PySide6.QtWidgets import QApplication
                app = QApplication.instance()
                if app and _icon_path.exists():
                    app.setWindowIcon(QIcon(str(_icon_path)))
            except Exception:
                pass

        window = webview.create_window(title, url, width=width, height=height)
        webview.start(func=_set_icon)
    else:
        # Graceful degradation: open in system browser, block on server
        Timer(1.0, lambda: webbrowser.open(url)).start()
        serve_fn()


@click.group()
def cli():
    """
    A command-line tool for processing image reports with Pixel Patrol.

    This tool facilitates a two-step process:
    1. Exporting a processed project to a ZIP file.
    2. Displaying a report from an exported ZIP file.
    """
    pass

@cli.command()
@click.argument('base_directory', type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True, path_type=Path))
@click.option('--output-zip', '-o', type=click.Path(exists=False, dir_okay=False, writable=True, path_type=Path),
              help='Required: Name of the output ZIP file for the exported project (e.g., my_project.zip).',
              required=True)
@click.option('--name', type=str, required=False,
              help='Optional: Name of the project. If not provided, derived from BASE_DIRECTORY.')
@click.option('--paths', '-p', multiple=True, type=str,
              help='Optional: Paths (subdirectories) to treat as **experimental conditions**, relative to BASE_DIRECTORY. '
                   'Can be specified multiple times. If omitted, all immediate subdirectories '
                   'of BASE_DIRECTORY will be included, or if BASE_DIRECTORY has no subdirectories, '
                   'it is treated as a single condition.')
@click.option('--loader', '-l', type=str, show_default=True,
              help='Recommended: Pixel Patrol file loader (e.g., bioio, zarr). If omitted, only basic file info is collected.')
@click.option('--cmap', type=str, default="rainbow", show_default=True,
              help='Colormap for report visualization (e.g., viridis, plasma, rainbow).')
@click.option('--n-example-files', type=int, default=9, show_default=True,
              help='Number of example files to display in the report.')
@click.option('--file-extension', '-e', multiple=True,
              help='Optional: File extensions to include (e.g., png, jpg). Can be specified multiple times. '
                   'If not specified, all supported extensions will be used.')
@click.option('--flavor', type=str, default="", show_default=True,
              help='Name of pixel patrol configuration, will be displayed next to the tool name.')
@click.option('--resume', 'resume', is_flag=True, default=False,
              help='If set, resume from previous partial chunk files and skip already-processed images. Default: perform a fresh run (clear previous chunks).')
@click.option('--chunk-dir', type=click.Path(exists=False, file_okay=False, dir_okay=True, path_type=Path),
                  default=None,
                  help='Optional: Directory to store intermediate parquet chunk files. Defaults to <output_zip_parent>/<project_name>_batches.')
def export(base_directory: Path, output_zip: Path, name: str | None, paths: tuple[str, ...],
              loader: str, cmap: str, n_example_files: int, file_extension: tuple[str, ...], flavor: str,
              resume: bool, chunk_dir: Path | None):
    """
    Exports a Pixel Patrol project to a ZIP file.
    Processes images from the BASE_DIRECTORY and specified --paths.
    """
    # Always operate on an absolute base directory so downstream path resolution is stable.
    base_directory = base_directory.resolve()

    # Derive project_name if not provided
    if name is None:
        name = base_directory.name # Use the name of the base directory
        click.echo(f"Project name not provided, deriving from base directory: '{name}'")

    click.echo(f"Creating project: '{name}' from base directory '{base_directory}'")
    my_project = create_project(name, str(base_directory), loader=loader) # Assuming create_project takes string path

    if paths:
        click.echo(f"Adding explicitly specified paths: {', '.join(paths)}. Resolution will be relative to '{base_directory}'")
        add_paths(my_project, paths)
    else:
        # If no paths, we want to add the base directory itself.
        click.echo(f"No --paths specified. Processing all images in '{base_directory}'.")
        add_paths(my_project, base_directory)

    selected_extensions = set(file_extension) if file_extension else "all"

    # Determine chunk dir: CLI option overrides default inferred location next to ZIP
    if chunk_dir is not None:
        chosen_chunk_dir = Path(chunk_dir).resolve()
    else:
        # Support implementations of create_project that return simple mappings (e.g. tests use a dict)
        # Prefer project name attribute, fallback to CLI 'name' that was already derived from the base directory.
        project_name_or_name = getattr(my_project, "name", None) or name
        chosen_chunk_dir = (output_zip.parent / f"{project_name_or_name}_batches").resolve()
    chunk_dir_was_inferred = chunk_dir is None

    # By default clear existing chunk dir to ensure a fresh run.
    # If --resume is passed, resume from existing partial chunks and skip already-processed images.
    if chosen_chunk_dir.exists() and not resume:
        click.echo(f"No --resume passed: clearing previous partial chunk files in: '{chosen_chunk_dir}'")
        _cleanup_partial_chunks_dir(chosen_chunk_dir, cleanup_combined_parquet=False)
    elif chosen_chunk_dir.exists() and resume:
        click.echo(f"--resume passed: resuming and skipping already-processed images in '{chosen_chunk_dir}'")

    initial_settings = Settings(
        cmap=cmap,
        n_example_files=n_example_files,
        selected_file_extensions=selected_extensions,
        pixel_patrol_flavor=flavor,
        records_flush_dir=chosen_chunk_dir,
        resume=resume,
    )
    click.echo(f"Setting project settings: {initial_settings}")
    set_settings(my_project, initial_settings)

    click.echo("Processing images...")
    process_files(my_project)

    click.echo(f"Exporting project to: '{output_zip}'")
    # Export. Project IO will include either the combined parquet (if present) or
    # any partial chunks and will perform final tidy-up by default.
    export_project(my_project, Path(output_zip))
    click.echo("Export complete.")



@cli.command()
@click.option('--port', type=int, default=8051, show_default=True,
              help='Port number for the Pixel Patrol launcher server.')
@click.option('--host', type=str, default=lambda: os.environ.get("PIXEL_PATROL_HOST", "127.0.0.1"),
              help='Host to bind to. Use 0.0.0.0 for Docker/remote access. Default: 127.0.0.1 (or PIXEL_PATROL_HOST env var).')
def launch(port: int, host: str):
    """
    Launch the Pixel Patrol web interface.

    Lists existing reports, lets you add new ones (with background processing),
    and opens them in a built-in report viewer — all on a single port.
    Reports are stored in PIXEL_PATROL_REPORTS_DIR (default: ~/pixel-patrol-reports).
    """
    from pixel_patrol_base.processing_dashboard import create_processing_app

    app = create_processing_app()
    display_host = "localhost" if host == "0.0.0.0" else host
    dashboard_url = f"http://{display_host}:{port}"
    click.echo(f"Starting Pixel Patrol at {dashboard_url}/")

    if host not in ("127.0.0.1", "localhost"):
        # Remote / Docker mode — can't open a local window; use browser.
        Timer(1, lambda: webbrowser.open_new_tab(dashboard_url)).start()
        app.run(debug=False, host=host, port=port, use_reloader=False)
    else:
        _serve_in_window(
            lambda: app.run(debug=False, host=host, port=port, use_reloader=False),
            url=dashboard_url,
        )


@cli.command()
@click.argument('input_zip', type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True, path_type=Path))
@click.option('--port', type=int, default=8050, show_default=True)
@click.option('--host', type=str, default=lambda: os.environ.get("PIXEL_PATROL_HOST", "127.0.0.1"),
              help='Host to bind to. Use 0.0.0.0 for Docker/remote access. Default: 127.0.0.1 (or PIXEL_PATROL_HOST env var).')
@click.option('--group-by', type=str, default=None)
@click.option('--filter-col', 'filter_col', type=str, default=None)
@click.option('--filter-op', type=click.Choice(["contains","not_contains","eq","gt","ge","lt","le","in"]), default=None)
@click.option('--filter', 'filter_value', type=str, default=None)
@click.option('--dim', 'dims', multiple=True, help="Repeatable, format like: t=0  z=1  c=0")
@click.option('--export-html', type=click.Path(exists=False, file_okay=True, dir_okay=False, writable=True, path_type=Path),
              help='Export the report as a static HTML file instead of launching the interactive dashboard.')
def report(input_zip: Path, port: int, host: str, group_by: str | None, filter_col: str | None,
           filter_op: str | None, filter_value: str | None, dims: tuple[str, ...], export_html: Path | None):

    my_project = import_project(Path(input_zip))

    dim_dict: dict[str, str] = {}
    for item in dims:
        s = str(item)
        if "=" not in s:
            raise click.BadParameter("Expected format key=value (e.g. z=1)")

        k, v = s.split("=", 1)
        k = k.strip()
        v = v.strip()

        # enforce unprefixed values: z=1 OK, z=z1 NOT OK
        if v.startswith(k):
            raise click.BadParameter(f"Use {k}=<value> (e.g. {k}=1), not {k}={v}")

        dim_dict[k] = v

    filters = {}
    if filter_col and filter_op and filter_value:
        filters[filter_col] = {"op": filter_op, "value": filter_value}

    global_config = {
        "group_col": group_by or NO_GROUPING_COL,
        "filter": filters,
        "dimensions": dim_dict,
    }

    if export_html:
        # Export as static HTML
        click.echo(f"Exporting report to HTML: {export_html}")
        export_html_report(my_project, export_html, host=host, port=port, global_config=global_config)
        click.echo(f"HTML export complete: {export_html}")
    else:
        from pixel_patrol_base.report.dashboard_app import create_app
        from pixel_patrol_base.report.global_controls import init_global_config

        sanitized = init_global_config(my_project.records_df, global_config)
        app = create_app(my_project, initial_global_config=sanitized)
        url = f"http://{'localhost' if host == '0.0.0.0' else host}:{port}/"

        if host not in ("127.0.0.1", "localhost"):
            Timer(1, lambda: webbrowser.open(url)).start()
            app.run(debug=False, host=host, port=port, use_reloader=False)
        else:
            _serve_in_window(
                lambda: app.run(debug=False, host=host, port=port, use_reloader=False),
                url=url,
                title=f"Pixel Patrol – {my_project.name}",
            )


@cli.command()
@click.argument("path", type=click.Path(exists=True, file_okay=True, dir_okay=True, path_type=Path))
@click.option("--loader", "loader_name", default=None,
              help="Loader plugin name to use (auto-detected from extension if omitted).")
@click.option("--port", type=int, default=0, show_default=True,
              help="Port for the inspection report (0 = pick a free port automatically).")
@click.option("--host", type=str, default=lambda: os.environ.get("PIXEL_PATROL_HOST", "127.0.0.1"),
              help="Host to bind to. Default: 127.0.0.1 (or PIXEL_PATROL_HOST env var).")
def inspect(path: Path, loader_name: str | None, port: int, host: str):
    """Process a single file and open an interactive inspection report."""
    from pixel_patrol_base.plugin_registry import (
        detect_loaders_for_file,
        discover_loader,
        discover_processor_plugins,
    )
    from pixel_patrol_base.report.inspect_app import create_inspect_app

    # ── Resolve loader ─────────────────────────────────────────────────────────
    if loader_name:
        loader = discover_loader(loader_name)
    else:
        candidates = detect_loaders_for_file(path)
        if not candidates:
            raise click.ClickException(
                f"No installed loader supports '{path.suffix}'. "
                "Try installing a compatible loader package (e.g. pixel-patrol-loader-bio)."
            )
        loader = candidates[0]
        click.echo(f"Using loader: {loader.NAME}")

    processors = discover_processor_plugins()

    # ── Pick a free port if none given ─────────────────────────────────────────
    if port == 0:
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as _s:
            _s.bind((host, 0))
            port = _s.getsockname()[1]

    # ── Build app (processing happens in background inside the app) ────────────
    app = create_inspect_app(path, loader, processors)
    url = f"http://{host}:{port}/"
    click.echo(f"Opening inspection report at {url}")

    if host not in ("127.0.0.1", "localhost"):
        Timer(1.0, lambda: webbrowser.open(url)).start()
        app.run(debug=False, host=host, port=port, use_reloader=False)
    else:
        _serve_in_window(
            lambda: app.run(debug=False, host=host, port=port, use_reloader=False),
            url=url,
            title=f"Pixel Patrol – {path.name}",
            width=1100,
            height=780,
        )


if __name__ == '__main__':
    cli()
