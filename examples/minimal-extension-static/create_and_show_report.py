"""
Minimal extension — static viewer example.

Shows how a Pixel Patrol extension adds a custom loader and processor (Python)
and viewer plugins (JavaScript) to the static, browser-based viewer.

Steps
-----
1. Process the diary Markdown files with the custom loader and processor.
   The processor adds ``positivity_factor`` and other columns to the parquet.
2. Serve the parquet with the static viewer, loading this folder as the
   extension.  The server reads ``extension.json`` to discover the plugin JS
   files and serves them alongside the viewer.

This folder can also be hosted remotely (e.g. GitHub Pages) and linked via
``?extension=<url>`` without any Python server involved.

Usage
-----
Install the package with uv, then run:

    uv run python create_and_show_report.py
"""

from pathlib import Path

from pixel_patrol_base.api import create_project, add_paths, process_files
from pixel_patrol_base.viewer_server import serve_viewer

HERE = Path(__file__).resolve().parent

if __name__ == "__main__":
    output = HERE / "out" / "report.parquet"
    output.parent.mkdir(parents=True, exist_ok=True)

    project = create_project(
        "Markdown Diary",
        HERE / "data",
        loader="markdown-diary",
        output_path=output,
    )
    project = add_paths(project, ["2024", "2025"])
    project = process_files(project)

    serve_viewer(output, extension=HERE / "viewer")
