import os
import shutil
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
    show_report,
)
from pixel_patrol_base.core.project_settings import Settings


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
@click.option('--rerun-incomplete', is_flag=True, default=False,
              help='If set, clear previous partial chunk files and re-run processing from scratch. Default: resume where left off.')
@click.option('--chunk-dir', type=click.Path(exists=False, file_okay=False, dir_okay=True, path_type=Path),
                  default=None,
                  help='Optional: Directory to store intermediate parquet chunk files. Defaults to <output_zip_parent>/<output_stem>_records_chunks.')
def export(base_directory: Path, output_zip: Path, name: str | None, paths: tuple[str, ...],
              loader: str, cmap: str, n_example_files: int, file_extension: tuple[str, ...], flavor: str,
              rerun_incomplete: bool, chunk_dir: Path | None):
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
        chosen_chunk_dir = (output_zip.parent / f"{output_zip.stem}_records_chunks").resolve()

    # If user requested a clean rerun, clear existing partials in the chosen dir
    if chosen_chunk_dir.exists() and rerun_incomplete:
        try:
            click.echo(f"--rerun-incomplete passed: clearing previous records chunk directory: '{chosen_chunk_dir}'")
            shutil.rmtree(chosen_chunk_dir)
        except OSError as exc:
            click.echo(f"Warning: Could not clear '{chosen_chunk_dir}': {exc}")

    initial_settings = Settings(
        cmap=cmap,
        n_example_files=n_example_files,
        selected_file_extensions=selected_extensions,
        pixel_patrol_flavor=flavor,
        records_flush_dir=chosen_chunk_dir,
    )
    # Note: Do not clear existing chunk dir by default so runs may be resumed.
    # Use --rerun-incomplete flag to force clearing of partial results before processing.
    # Clear existing partial chunks only when explicitly requested by the user.
    if chunk_dir.exists() and rerun_incomplete:
        try:
            click.echo(f"--rerun-incomplete passed: clearing previous records chunk directory: '{chunk_dir}'")
            shutil.rmtree(chunk_dir)
        except OSError as exc:
            click.echo(f"Warning: Could not clear '{chunk_dir}': {exc}")
    click.echo(f"Setting project settings: {initial_settings}")
    set_settings(my_project, initial_settings)

    click.echo("Processing images...")
    process_files(my_project)

    click.echo(f"Exporting project to: '{output_zip}'")
    export_project(my_project, Path(output_zip)) # Assuming export_project takes string path
    click.echo("Export complete.")


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
@click.argument('input_zip', type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True, path_type=Path))
@click.option('--port', type=int, default=8050, show_default=True)
@click.option('--group-by', type=str, default=None)
@click.option('--filter-col', type=str, default=None)
@click.option('--filter-op', type=click.Choice(["contains","not_contains","eq","gt","ge","lt","le","in"]), default=None)
@click.option('--filter', 'filter_value', type=str, default=None)
@click.option('--dim', 'dims', multiple=True, help="Repeatable, format like: t=0  z=1  c=0")
def report(input_zip: Path, port: int, group_by: str | None, filter_col: str | None,
           filter_op: str | None, filter_value: str | None, dims: tuple[str, ...]):

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
        "group_col": [group_by] if group_by else ["report_group"],
        "filter": filters,
        "dimensions": dim_dict,
    }

    show_report(my_project, port=port, global_config=global_config)


if __name__ == '__main__':
    cli()
