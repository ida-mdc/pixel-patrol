import os
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path
from threading import Timer
from typing import Set, Union, Literal

import click

from pixel_patrol.api import (
    create_project,
    add_paths,
    process_paths,
    set_settings,
    process_images,
    export_project,
    import_project,
    show_report,
)


# Your provided Settings class
@dataclass
class Settings:
    cmap: str = "rainbow"
    n_example_images: int = 9
    selected_file_extensions: Union[Set[str], Literal["all"]] = field(default_factory=set)

# --- CLI Tool ---

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
@click.option('--paths', '-p', multiple=True, type=str, # Changed to str, will be resolved as Path relative to base_directory
              help='Optional: Paths to include in the project, relative to BASE_DIRECTORY. '
                   'Can be specified multiple times. If omitted, all immediate subdirectories '
                   'of BASE_DIRECTORY will be included.')
@click.option('--cmap', type=str, default="rainbow", show_default=True,
              help='Colormap for report visualization (e.g., viridis, plasma, rainbow).')
@click.option('--n-example-images', type=int, default=9, show_default=True,
              help='Number of example images to display in the report.')
@click.option('--file-extension', '-e', multiple=True,
              help='Optional: File extensions to include (e.g., png, jpg). Can be specified multiple times. '
                   'If not specified, all supported extensions will be used.')
def export(base_directory: Path, output_zip: Path, name: str | None, paths: tuple[str, ...],
           cmap: str, n_example_images: int, file_extension: tuple[str, ...]):
    """
    Exports a Pixel Patrol project to a ZIP file.
    Processes images from the BASE_DIRECTORY and specified --paths.
    """
    # Derive project_name if not provided
    if name is None:
        name = base_directory.name # Use the name of the base directory
        click.echo(f"Project name not provided, deriving from base directory: '{name}'")

    click.echo(f"Creating project: '{name}' from base directory '{base_directory}'")
    my_project = create_project(name, str(base_directory)) # Assuming create_project takes string path

    # Handle paths: explicit --paths or auto-discover subdirectories
    resolved_paths = []
    if paths:
        for rel_path_str in paths:
            abs_path = base_directory / rel_path_str
            if not abs_path.is_dir():
                click.echo(f"Warning: Specified path '{abs_path}' is not a valid directory. Skipping.", err=True)
                continue
            resolved_paths.append(Path(abs_path))
            click.echo(f"Adding explicitly specified path: '{abs_path}'")
    else:
        click.echo(f"No --paths specified. Processing all images in '{base_directory}'.")
        resolved_paths.append(Path(base_directory))

    if not resolved_paths:
        click.echo("No valid paths found to process. Please check your base directory or --paths.", err=True)
        # Optionally, raise an error or exit here if no paths is considered an invalid state
        return

    for path in resolved_paths:
        add_paths(my_project, path)

    process_paths(my_project)
    selected_extensions = set(file_extension) if file_extension else "all"
    initial_settings = Settings(
        cmap=cmap,
        n_example_images=n_example_images,
        selected_file_extensions=selected_extensions
    )
    click.echo(f"Setting project settings: {initial_settings}")
    set_settings(my_project, initial_settings)

    click.echo("Processing images...")
    process_images(my_project)

    click.echo(f"Exporting project to: '{output_zip}'")
    export_project(my_project, Path(output_zip)) # Assuming export_project takes string path
    click.echo("Export complete.")


@cli.command()
@click.argument('input_zip', type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True, path_type=Path))
@click.option('--port', type=int, default=8050, show_default=True,
              help='Port number for the Dash report server.')
def report(input_zip: Path, port: int):
    """
    Displays the report of an exported Pixel Patrol project from a ZIP file.
    """
    click.echo(f"Importing project from: '{input_zip}'")
    my_project = import_project(Path(input_zip))
    click.echo("Project imported.")

    report_url = f"http://127.0.0.1:{port}"
    click.echo(f"Dash report will run on {report_url}/")
    click.echo("Attempting to open report in your default browser...")

    # We don't need a Timer here if show_report is blocking and we open BEFORE
    # However, in some systems, opening too fast can fail. A small delay is safer.
    def _open_browser():
        # This check is still useful if Werkzeug debug mode spawns a second process
        # and you want to ensure it only opens once from the parent.
        if not os.environ.get("WERKZEUG_RUN_MAIN"):
            webbrowser.open_new_tab(report_url)

    # Schedule the browser open for slightly in the future to give the OS a moment
    # and to potentially allow Dash's own startup messages to appear first.
    Timer(1, _open_browser).start() # 0.5 seconds delay

    click.echo("Showing report...")
    # Pass the port to show_report. You must ensure show_report accepts this.
    show_report(my_project, port=port)

if __name__ == '__main__':
    cli()