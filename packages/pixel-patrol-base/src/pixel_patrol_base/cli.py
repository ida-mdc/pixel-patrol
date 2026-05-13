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
def process(base_directory: Path, output: Path, name: str | None, paths: tuple[str, ...],
              loader: str, file_extensions: tuple[str, ...],
              flavor: str, description: str,
              processors_include: tuple[str, ...], processors_exclude: tuple[str, ...],
              parquet_row_group_size: int | None):
    """
    Processes images from the BASE_DIRECTORY and specified --paths and saves a parquet file
    """
    base_directory = base_directory.resolve()

    if name is None:
        name = base_directory.name # Use the name of the base directory
        click.echo(f"Project name not provided, deriving from base directory: '{name}'")

    click.echo(f"Creating project: '{name}' from base directory '{base_directory}'")
    my_project = create_project(name, str(base_directory), loader=loader, output_path=output)

    if paths:
        click.echo(f"Adding explicitly specified paths: {', '.join(paths)}. Resolution will be relative to '{base_directory}'")
        add_paths(my_project, paths)
    else:
        # If no paths, we want to add the base directory itself.
        click.echo(f"No --paths specified. Processing all images in '{base_directory}'.")
        add_paths(my_project, base_directory)

    selected_extensions = set(file_extensions) if file_extensions else "all"

    output_path = Path(output).resolve()
    if output_path.suffix.lower() != ".parquet":
        output_path = output_path.with_suffix(".parquet")
        click.echo(f"Output path has no .parquet extension, using: '{output_path}'")

    click.echo("Processing images...")
    process_files(
        my_project,
        selected_file_extensions=selected_extensions,
        processors_included=set(processors_include) if processors_include else None,
        processors_excluded=set(processors_exclude) if processors_exclude else None,
        flavor=flavor or None,
        description=description or None,
        parquet_row_group_size=parquet_row_group_size,
    )

    if Path(output).exists():
        click.echo(f"Output saved to: '{output}'")
    else:
        click.echo(f"Processing complete. Expected output: '{output}'")


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
