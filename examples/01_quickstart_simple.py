from pathlib import Path
from pixel_patrol_base import api
from pixel_patrol_base.core.project_settings import Settings
import logging
logging.basicConfig(level=logging.INFO)

def main():
    # --- Step 1: choose a base directory with files to be processed, sub-paths, and loader ---
    base_path = Path("datasets/bioio")
    # output path of the project zip file to share your results
    zip_path = Path("out_parallel/quickstart_project.zip")

    # Optional: Define sub-dirs that in the report are compared as experimental conditions
    # If you don't specify any paths, all files in base_path and its subfolders are processed as one condition.
    # You can specify the paths either as absolute or relevant to base_path.
    # This line will add all sub-directories as separate conditions to compare in the report
    paths = [p.name for p in base_path.iterdir() if p.is_dir() and not p.name.startswith('.')]

    # Load data using the 'bioio' loader plugin
    loader = 'bioio'  # for image files (e.g. png, jpg, tiff, etc.); requires pixel-patrol-loader-bio package

    # Set your preferred file extensions to process (alternatively set to `{"tif", "png", "jpeg", ...}`, etc.)
    # if loader is None all file types are processed, otherwise all file types supported by the loader
    selected_file_extensions = "all" 

    # --- Step 2: create a project ---
    project = api.create_project("Quickstart Project", base_dir=base_path, loader=loader)
    # Optional: set project settings and the distinguished file extensions and paths to process
    api.set_settings(project, Settings(selected_file_extensions=selected_file_extensions))
    api.add_paths(project, paths)

    # --- Step 3: process files ---
    # This step creates a dataframe with file information, and if available metadata and quality control results.
    api.process_files(project)

    # --- Step 4: export project for sharing ---
    api.export_project(project, zip_path) # project exports to zip_path 'out/'

    # --- Step 5: open the dashport report ---
    # Open http://127.0.0.1:8050/ in your browser
    api.show_report(project,)


if __name__ == "__main__":
    main()
