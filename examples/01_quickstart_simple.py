from pathlib import Path
from pixel_patrol_base import api
import logging
logging.basicConfig(level=logging.INFO)


def main():
    # --- Step 1: choose a base directory with files to be processed, sub-paths, and loader ---
    base_path = Path("datasets/bioio")
    # output path where the project table (a .parquet file) will be saved
    output_path = Path("out/quickstart_simple.parquet")

    # Optional: Define sub-dirs that will be processed.
    paths = [p.name for p in base_path.iterdir() if p.is_dir() and not p.name.startswith('.')]

    # Load data using the 'bioio' loader plugin
    loader = 'bioio'  # for image files (e.g. png, jpg, tiff, etc.); requires pixel-patrol-loader-bio package

    # --- Step 2: create a project ---
    project = api.create_project("quickstart_simple", base_dir=base_path, loader=loader, output_path=output_path)
    # Optional: set project settings and the distinguished file extensions and paths to process
    api.add_paths(project, paths)

    # --- Step 3: process files ---
    # This step creates a dataframe with file information, and if available metadata and quality control results.
    api.process_files(project)

    # --- Step 5: open the dashport report ---
    # Open http://127.0.0.1:8050/ in your browser
    api.show_report(project)


if __name__ == "__main__":
    main()
