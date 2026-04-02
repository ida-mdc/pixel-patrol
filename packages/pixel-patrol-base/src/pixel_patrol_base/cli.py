import os
import webbrowser
from pathlib import Path
from threading import Timer

import click

from pixel_patrol_base.api import (
    create_project,
    add_paths,
    process_files,
    show_report,
    export_html_report,
)
from pixel_patrol_base.core.processing_config import ProcessingConfig
from pixel_patrol_base.core.project_metadata import ProjectMetadata
from pixel_patrol_base.core.report_config import ReportConfig
from pixel_patrol_base.report.constants import NO_GROUPING_COL, DEFAULT_CMAP


@click.group()
def cli():
    """
    A command-line tool for processing image reports with Pixel Patrol.

    This tool facilitates a two-step process:
    1. Processing images from a specified base directory and saving the results as a parquet file.
    2. Displaying a report from an exported parquet file.
    """
    pass

@cli.command()
@click.argument('base_directory', type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True, path_type=Path))
@click.option('--output', '-o', type=click.Path(exists=False, file_okay=False, writable=True, path_type=Path),
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
@click.option('--authors', type=str, default="",
              help='Optional: Authors of this project (free-form, e.g., "ella, deborah").')
@click.option('--processors-include', multiple=True, type=str,
              help='Only use these processors (e.g., basic-stats). Can be specified multiple times. If specified, --processors-exclude is ignored.')
@click.option('--processors-exclude', multiple=True, type=str,
              help='Exclude these processors (e.g., histogram). Can be specified multiple times.')
def process(base_directory: Path, output: Path, name: str | None, paths: tuple[str, ...],
              loader: str, file_extensions: tuple[str, ...],
              flavor: str, authors: str,
              processors_include: tuple[str, ...], processors_exclude: tuple[str, ...]):
    """
    Processes images from the BASE_DIRECTORY and specified --paths and saves a parquet file
    """
    base_directory = base_directory.resolve()

    if name is None:
        name = base_directory.name # Use the name of the base directory
        click.echo(f"Project name not provided, deriving from base directory: '{name}'")

    click.echo(f"Creating project: '{name}' from base directory '{base_directory}'")
    my_project = create_project(name, str(base_directory), loader=loader)

    if paths:
        click.echo(f"Adding explicitly specified paths: {', '.join(paths)}. Resolution will be relative to '{base_directory}'")
        add_paths(my_project, paths)
    else:
        # If no paths, we want to add the base directory itself.
        click.echo(f"No --paths specified. Processing all images in '{base_directory}'.")
        add_paths(my_project, base_directory)

    selected_extensions = set(file_extensions) if file_extensions else "all"

    output_dir = Path(output).resolve()

    metadata = ProjectMetadata(
        flavor=flavor,
        authors=authors,
    )

    processing_config = ProcessingConfig(
        selected_file_extensions=selected_extensions,
        output_dir=output_dir,
        processors_included=set(processors_include) if processors_include else set(),
        processors_excluded=set(processors_exclude) if processors_exclude else set(),
        metadata=metadata,
    )
    click.echo(f"Processing configuration: {processing_config}")

    click.echo("Processing images...")
    process_files(my_project, processing_config=processing_config)

    # process_files -> project.process_records saves the final parquet
    # (with metadata in footer) to output_dir/<name>.parquet.
    final_parquet = output_dir / f"{name}.parquet"
    if final_parquet.exists():
        click.echo(f"Output saved to: '{final_parquet}'")
    else:
        click.echo(f"Processing complete. Output directory: '{output_dir}'")



@cli.command()
@click.option('--port', type=int, default=8051, show_default=True,
              help='Port number for the Dash processing dashboard server.')
def launch(port: int):
    """
    Launches the web-based processing dashboard for configuring and monitoring Pixel Patrol processing.
    """
    from pixel_patrol_base.processing_dashboard import create_processing_app
    
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


@cli.command()
@click.argument('input_parquet',
                type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True, path_type=Path))
@click.option('--port', type=int, default=8050, show_default=True)
@click.option('--group-by', type=str, default=None,
              help='Column name to group by in the report.')
@click.option('--filter-col', 'filter_col', type=str, default=None,
              help='Column name to filter on.')
@click.option('--filter-op', type=click.Choice(["contains","not_contains","eq","gt","ge","lt","le","in"]), default=None,
              help='Filter operation.')
@click.option('--filter-val', 'filter_value', type=str, default=None,
              help='Filter value.')
@click.option('--dim', 'dims', multiple=True, help="Repeatable, format like: t=0  z=1  c=0")
@click.option('--export-html', type=click.Path(exists=False, file_okay=True, dir_okay=False, writable=True, path_type=Path),
              help='Export the report as a static HTML file instead of launching the interactive dashboard.')
@click.option('--widgets-include', multiple=True, type=str,
              help='Only show these widgets in the report. Can be specified multiple times. If specified, --widgets-exclude is ignored.')
@click.option('--widgets-exclude', multiple=True, type=str,
              help='Exclude these widgets from the report (e.g., "TensorBoard Embedding Projector"). Can be specified multiple times.')
@click.option('--cmap', type=str, default=DEFAULT_CMAP, show_default=True,
              help='Colormap for report visualization (e.g., viridis, plasma, rainbow).')
def report(input_parquet: Path, port: int,
           group_by: str | None,
           filter_col: str | None,
           filter_op: str | None, filter_value: str | None, dims: tuple[str, ...],
           widgets_include: tuple[str, ...], widgets_exclude: tuple[str, ...],
           cmap: str,
           export_html: Path | None,):
    """
    Display or export a report from a processed parquet file.

    INPUT_PARQUET is the path to a .parquet file produced by the 'process' command.
    By default, launches an interactive dashboard. Use --export-html to save a static HTML file instead.
    """
    dim_dict: dict[str, str] = {}
    for item in dims:
        s = str(item)
        if "=" not in s:
            raise click.BadParameter("Expected format key=value (e.g. z=1)")

        k, v = s.split("=", 1)
        k = k.strip()
        v = v.strip()

        # enforce un-prefixed values: z=1 OK, z=z1 NOT OK
        if v.startswith(k):
            raise click.BadParameter(f"Use {k}=<value> (e.g. {k}=1), not {k}={v}")

        dim_dict[k] = v

    filters = {}
    if filter_col and filter_op and filter_value:
        filters[filter_col] = {"op": filter_op, "value": filter_value}

    # Build ReportConfig from CLI options (combines widgets and global config)
    report_config = None
    if widgets_include or widgets_exclude or group_by or filter_col or dims:
        report_config = ReportConfig(
            cmap=cmap,
            widgets_included=set(widgets_include) if widgets_include else set(),
            widgets_excluded=set(widgets_exclude) if widgets_exclude else set(),
            group_col=group_by or NO_GROUPING_COL if group_by else None,
            filter=filters if filters else None,
            dimensions=dim_dict if dim_dict else None,
        )
        click.echo(f"Report configuration: widgets={len(report_config.widgets_included)} included, {len(report_config.widgets_excluded)} excluded")

    parquet_path = Path(input_parquet)

    if export_html:
        # Export as static HTML
        click.echo(f"Exporting report to HTML: {export_html}")
        export_html_report(parquet_path, export_html, port=port, report_config=report_config)
        click.echo(f"HTML export complete: {export_html}")
    else:
        # Launch interactive dashboard
        show_report(parquet_path, port=port, report_config=report_config)


if __name__ == '__main__':
    cli()
