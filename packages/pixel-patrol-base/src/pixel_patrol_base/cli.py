import logging
import os
import webbrowser
from pathlib import Path
from threading import Timer

import click

from pixel_patrol_base.api import (
    create_project,
    add_paths,
    process_files,
    view as api_view,
    build_viewer as api_build_viewer,
)

from pixel_patrol_base.processing_dashboard import create_processing_app


def _configure_cli_logging() -> None:
    """Suppress noisy third-party INFO logs; keep warnings and errors."""
    logging.getLogger().setLevel(logging.WARNING)
    for noisy in ("numexpr", "numexpr.utils", "botocore", "boto3", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # Inherited by worker subprocesses — prevents numexpr from logging "detected N cores"
    os.environ.setdefault("NUMEXPR_MAX_THREADS", str(os.cpu_count() or 1))


@click.group()
def cli():
    """
    A command-line tool for processing image reports with Pixel Patrol.

    Two-step workflow:
      1. pixel-patrol process  — scan images, write a .parquet report file
      2. pixel-patrol view     — open the report in the interactive viewer
    """
    pass

@cli.command()
@click.argument('base_directory', type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True, path_type=Path))
@click.option('--output', '-o', type=click.Path(exists=False, file_okay=True,  dir_okay=False, writable=True, path_type=Path),
              help='Required: Path for the output parquet file (e.g., my_project.parquet).',
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
@click.option('--file-extensions', '-e', multiple=True,
              help='Optional: File extensions to include (e.g., png, jpg). Can be specified multiple times. '
                   'If not specified, all supported extensions will be used.')
@click.option('--flavor', type=str, default="", show_default=True,
              help='Name of pixel patrol configuration, will be displayed next to the tool name.')
@click.option('--description', type=str, default="",
              help='Optional: Description of this project (free-form, e.g., "Authors: Annona Buddha and Banana Java").')
@click.option('--processors-include', multiple=True, type=str,
              help='Only use these processors (e.g., basic-stats). Can be specified multiple times. If specified, --processors-exclude is ignored.')
@click.option('--processors-exclude', multiple=True, type=str,
              help='Exclude these processors (e.g., histogram). Can be specified multiple times.')
@click.option('--parquet-row-group-size', type=int, default=None, show_default=True,
              help='Number of rows per parquet row group (default: 2048). Smaller values reduce I/O when the viewer samples thumbnails.')
@click.option('--max-workers', type=int, default=None, show_default=True,
              help='Maximum number of processing workers. Use 1 to disable multiprocessing.')
@click.option('--scheduler', type=str, default=None, show_default=True,
              help='Dask distributed scheduler. Use "local" to start a LocalCluster in-process '
                   '(with dashboard), or a scheduler address for HPC (e.g. tcp://host:8786).')
def process(base_directory: Path, output: Path, name: str | None, paths: tuple[str, ...],
              loader: str, file_extensions: tuple[str, ...],
              flavor: str, description: str,
              processors_include: tuple[str, ...], processors_exclude: tuple[str, ...],
              parquet_row_group_size: int | None,
              max_workers: int | None,
              scheduler: str | None):
    """
    Processes images from the BASE_DIRECTORY and specified --paths and saves a parquet file.

    Runs locally by default (parallel processes, RAM-based worker count).
    Use --scheduler to enable dask.distributed:

    \b
      --scheduler local              start a LocalCluster in-process (shows dashboard URL)
      --scheduler tcp://host:8786    connect to an existing HPC cluster
    """
    _configure_cli_logging()

    if scheduler:
        # dask.config.set() is read by distributed when it initialises its logging
        # during cluster/client creation — higher priority than distributed's defaults.
        # The env vars are inherited by worker subprocesses (which ignore Python-level config).
        import dask
        dask.config.set({
            "distributed.logging.distributed": "warning",
            "distributed.logging.tornado": "critical",
        })
        os.environ.setdefault("DASK_DISTRIBUTED__LOGGING__DISTRIBUTED", "warning")
        os.environ.setdefault("DASK_DISTRIBUTED__LOGGING__TORNADO", "critical")
        try:
            from dask.distributed import Client as _Client
            if scheduler.lower() == 'local':
                from dask.distributed import LocalCluster as _LocalCluster
                from pixel_patrol_base.core.processing import _resolve_worker_count
                from pixel_patrol_base.core.processing_config import ProcessingConfig
                _target_mb = float(os.environ.get('PIXEL_PATROL_MAX_BLOCK_MB', '1024'))
                _n = _resolve_worker_count(ProcessingConfig(processing_max_workers=max_workers), target_mb=_target_mb)
                _cluster = _LocalCluster(n_workers=_n, threads_per_worker=1, memory_limit=0)
                _dist_client = _Client(_cluster)
            else:
                _dist_client = _Client(scheduler)
        except Exception as exc:
            click.echo(f"Could not connect to scheduler '{scheduler}': {exc}", err=True)
            raise SystemExit(1)

    base_directory = base_directory.resolve()
    if name is None:
        name = base_directory.name

    output_path = Path(output).resolve()
    if output_path.suffix.lower() != ".parquet":
        output_path = output_path.with_suffix(".parquet")

    # Resolve processors for the summary line before creating the project
    from pixel_patrol_base.plugin_registry import discover_loader, discover_processor_plugins
    if loader:
        try:
            discover_loader(loader)
        except RuntimeError:
            available = [p.NAME for p in discover_processor_plugins()]
            # discover available loaders via entry points
            import importlib.metadata
            loader_eps = importlib.metadata.entry_points(group="pixel_patrol.loader_plugins")
            loader_names: list[str] = []
            for ep in loader_eps:
                try:
                    reg = ep.load()
                    loader_names.extend(c.NAME for c in reg() if hasattr(c, "NAME"))
                except Exception:
                    pass
            click.echo(f"Loader '{loader}' not found. Available loaders: {', '.join(sorted(loader_names)) or 'none'}")
            raise SystemExit(1)

    processors = discover_processor_plugins()
    processor_names = ", ".join(p.NAME for p in processors) or "none"

    click.echo(f"Processing '{name}'")
    click.echo(f"  dir:        {base_directory}")
    click.echo(f"  loader:     {loader or '(none — basic file info only)'}")
    click.echo(f"  processors: {processor_names}")
    from pixel_patrol_base.core.processing import _resolve_worker_count, _ram_worker_limit
    from pixel_patrol_base.core.processing_config import ProcessingConfig
    target_mb = float(os.environ.get('PIXEL_PATROL_MAX_BLOCK_MB', '1024'))
    resolved_workers = _resolve_worker_count(
        ProcessingConfig(processing_max_workers=max_workers), target_mb=target_mb
    )
    try:
        from dask.distributed import get_client as _get_client
        _dist_client = _get_client()
        click.echo(f"  scheduler:  distributed ({_dist_client.scheduler_info().get('address', '?')})  →  {output_path}")
        if _dist_client.dashboard_link:
            click.echo(f"  dashboard:  {_dist_client.dashboard_link}")
    except (ImportError, ValueError):
        if max_workers:
            click.echo(f"  workers:    {resolved_workers} (explicit)  →  {output_path}")
        else:
            ram_limit = _ram_worker_limit(target_mb)
            click.echo(f"  workers:    {resolved_workers} (RAM-based, limit={ram_limit} @ {target_mb:.0f} MB/worker)  →  {output_path}")

    my_project = create_project(name, str(base_directory), loader=loader, output_path=output_path)
    add_paths(my_project, paths if paths else base_directory)

    selected_extensions = set(file_extensions) if file_extensions else "all"

    try:
        process_files(
            my_project,
            selected_file_extensions=selected_extensions,
            processors_included=set(processors_include) if processors_include else None,
            processors_excluded=set(processors_exclude) if processors_exclude else None,
            processing_max_workers=max_workers,
            flavor=flavor or None,
            description=description or None,
            parquet_row_group_size=parquet_row_group_size,
        )
    except KeyboardInterrupt:
        click.echo("\nInterrupted.")
        raise SystemExit(1)

    if output_path.exists():
        size_mb = output_path.stat().st_size / (1024 ** 2)
        click.echo(f"Saved: {output_path}  ({size_mb:.1f} MB)")

    summary_path = output_path.with_suffix(".summary.txt")
    if summary_path.exists():
        click.echo("")
        click.echo(summary_path.read_text(encoding="utf-8"))


@cli.command()
@click.option('--port', type=int, default=8051, show_default=True,
              help='Port number for the Dash processing dashboard server.')
def launch(port: int):
    """
    Launches the web-based processing dashboard for configuring and monitoring Pixel Patrol processing.
    """
    
    app = create_processing_app()
    dashboard_url = f"http://127.0.0.1:{port}"
    click.echo(f"Processing dashboard will run on {dashboard_url}/")
    click.echo("Attempting to open dashboard in your default browser...")

    def _open_browser():
        if not os.environ.get("WERKZEUG_RUN_MAIN"):
            webbrowser.open_new_tab(dashboard_url)

    Timer(1, _open_browser).start()

    click.echo("Starting processing dashboard...")
    app.run(debug=False, host="127.0.0.1", port=port, use_reloader=False)


def _filter_options(fn):
    """Decorator that adds --group-by, --filter-*, --dim, --widgets-exclude, --sig flags."""
    fn = click.option('--group-by', type=str, default=None,
                      help='Column name to group by.')(fn)
    fn = click.option('--filter-col', 'filter_col', type=str, default=None,
                      help='Column name to filter on.')(fn)
    fn = click.option('--filter-op', type=click.Choice(
                          ["contains", "not_contains", "eq", "gt", "ge", "lt", "le", "in"]),
                      default=None, help='Filter operation.')(fn)
    fn = click.option('--filter-val', 'filter_value', type=str, default=None,
                      help='Filter value.')(fn)
    fn = click.option('--dim', 'dims', multiple=True,
                      help='Dimension selection, repeatable. Format: t=0  z=1  c=0')(fn)
    fn = click.option('--widgets-exclude', multiple=True, type=str,
                      help='Plugin IDs to hide (e.g. histogram). Repeatable.')(fn)
    fn = click.option('--significance', 'is_show_significance', is_flag=True, default=False,
                      help='Show statistical significance brackets.')(fn)
    fn = click.option('--palette', type=str, default=None,
                      help='Color palette name (default: tab10).')(fn)
    return fn


@cli.command()
@click.argument('parquet_file',
                type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True, path_type=Path))
@click.option('--port', type=int, default=8052, show_default=True,
              help='Port for the local viewer server.')
@click.option('--no-browser', is_flag=True, default=False,
              help='Do not open the browser automatically.')
@_filter_options
def view(parquet_file, port, no_browser,
         group_by, filter_col, filter_op, filter_value, dims,
         widgets_exclude, is_show_significance, palette):
    """
    Open a parquet file in the Pixel Patrol static viewer.

    PARQUET_FILE is the path to a .parquet file produced by the 'process' command.
    Starts a local HTTP server backed by native DuckDB and opens the viewer in
    the browser.

    Examples:

    \b
      pixel-patrol view report.parquet
      pixel-patrol view report.parquet --group-by file_extension
      pixel-patrol view report.parquet --filter-col dtype --filter-op eq --filter-val uint8
      pixel-patrol view report.parquet --dim t=0 --dim c=1
      pixel-patrol view report.parquet --widgets-exclude histogram --widgets-exclude summary
    """
    api_view(
        parquet_file,
        port=port,
        open_browser=not no_browser,
        group_col=group_by,
        filter_by=_parse_filter(filter_col, filter_op, filter_value),
        dimensions=_parse_dims(dims),
        widgets_excluded=set(widgets_exclude) if widgets_exclude else None,
        is_show_significance=is_show_significance,
        palette=palette,
    )


@cli.command("build-viewer-html")
@click.option(
    "--output",
    "-o",
    type=click.Path(exists=False, file_okay=True, dir_okay=True, writable=True, path_type=Path),
    required=True,
    help='Output path. Use .html/.htm for a single self-contained file, '
         'or a directory for a GitHub Pages-style site folder.')
def build_viewer_html(output: Path):
    """
    Build static viewer output from the installed web viewer bundle.

    If OUTPUT ends in .html/.htm, writes a single self-contained HTML file.
    Otherwise, writes a GitHub Pages-style site folder (index.html + assets + extensions).
    """
    out = api_build_viewer(output)
    if Path(out).is_dir():
        click.echo(f"Static viewer site written to: {out}")
    else:
        click.echo(f"Single-file viewer written to: {out}")


def _parse_dims(dims: tuple) -> dict:
    """Parse ('t=0', 'z=1') → {'t': '0', 'z': '1'}."""
    result = {}
    for item in dims:
        s = str(item)
        if "=" not in s:
            raise click.BadParameter(f"Expected format key=value (e.g. z=1), got: {s!r}")
        k, v = s.split("=", 1)
        k, v = k.strip(), v.strip()
        if v.startswith(k):
            raise click.BadParameter(f"Use {k}=<value> (e.g. {k}=1), not {k}={v}")
        result[k] = v
    return result


def _parse_filter(filter_col, filter_op, filter_value) -> dict | None:
    """Build filter_by dict from CLI args, or None if incomplete."""
    if filter_col and filter_op and filter_value:
        return {filter_col: {"op": filter_op, "value": filter_value}}
    return None

if __name__ == '__main__':
    cli()
