"""
Minimal extension example.

Shows how a Pixel Patrol extension adds a custom loader and processor (Python)
and viewer plugins (JavaScript) to the browser-based viewer.

The "Pixel HAI Watch" dataset is a handful of tiny `.parquet` tables, each
holding a 16x16 grid of `uint8` values plus fake "image metadata" (depth_zone)
— read by SharkCamLoader as if they were deep-sea camera snapshots.

Steps
-----
1. Generate the tiny dataset (skipped if it already exists).
2. Process the parquet "dive patches" with the custom loader and processor.
   The processor adds ``glow_count`` and other columns to the report.
3. Serve the parquet with the viewer. The server auto-discovers viewer
   extensions from installed packages (``pixel_patrol.viewer_extensions``
   entry-point group) and loads the JS plugins bundled with this package.

Usage
-----
Install the package with uv, then run:

    uv run python create_and_show_report.py
"""

from pathlib import Path

from pixel_patrol_base.api import create_project, add_paths, process_files
from pixel_patrol_base.viewer_server import serve_viewer

from make_dataset import main as make_dataset, DESCRIPTION

HERE = Path(__file__).resolve().parent

if __name__ == "__main__":
    data_dir = HERE / "data"
    if not data_dir.exists():
        make_dataset()

    output = HERE / "out" / "report.parquet"
    output.parent.mkdir(parents=True, exist_ok=True)

    project = create_project(
        "Pixel HAI Watch",
        data_dir,
        loader="shark-cam",
        output_path=output,
    )
    project = add_paths(project, ["azores_log", "kermadec_log"])
    # `description` is shown right below the title in the viewer and stored
    # in the report's own metadata - the natural home for a project blurb
    # (as opposed to per-file metadata, which SharkCamLoader reads instead).
    project = process_files(project, description=DESCRIPTION)

    serve_viewer(output)
