#!/usr/bin/env python
import logging
from pathlib import Path

from pixel_patrol.api import (
    create_project,
    process_paths,
    set_settings,
    process_images,
    get_images_df,
    export_project, add_paths,
)
from pixel_patrol.core.project_settings import Settings

# — Configure Logging —
logging.basicConfig(
    level=logging.INFO,
    format="%((asctime)s)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":

    # Base folder containing multiple test-image subdirectories
    base_path = Path(__file__).resolve().parent / "data" / "geff"
    logger.info(f"Scanning for immediate subdirectories under: {base_path}")

    # Define the output directory for the exported project
    output_directory = Path(__file__).parent / "exported_projects"
    output_directory.mkdir(parents=True, exist_ok=True)  # Ensure output directory exists

    # Define the path for the exported zip file
    exported_project_path = output_directory / "geff_exported.zip"

    # Initialize project with the base directory
    project = create_project("Local Test Image Collections", base_path)

    project = add_paths(project, ["anniek", "georgeos", "morgan", "teun"])

    # Discover and process all image paths
    project = process_paths(project)

    # Configure image-processing settings
    settings = Settings(
        selected_file_extensions={"zarr", "tiff", "json"},
        cmap="viridis",
        n_example_images=5,
    )
    project = set_settings(project, settings)

    # Compute all image metrics
    project = process_images(project)

    # Display head of the images DataFrame
    df = get_images_df(project)
    if df is not None and not df.is_empty():
        print("\n--- Images DataFrame Head ---")
        print(df.head())
        print(f"\nTotal images: {df.height}")
    else:
        logger.warning("No images were found or the DataFrame is empty.")

    export_project(project, exported_project_path)
