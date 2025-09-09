#!/usr/bin/env python
import logging
from pathlib import Path

from pixel_patrol_base.api import (
    create_project,
    add_paths,
    set_settings,
    process_images,
    get_images_df,
    show_report,
)
from pixel_patrol_base.core.project_settings import Settings

# — Configure Logging —
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # Base folder containing multiple test-image subdirectories
    base_path = Path(__file__).resolve().parent / "data" / "basic_image_data"
    logger.info(f"Scanning for immediate subdirectories under: {base_path}")

    # Collect all immediate child directories as separate data sources
    subdirs = [d for d in base_path.iterdir() if d.is_dir()]
    if not subdirs:
        logger.error(f"No subdirectories found in {base_path}")
        exit(1)
    logger.info(f"Found subdirectories: {[d.name for d in subdirs]}")

    # Initialize project with the base directory
    project = create_project("Local Test Image Collections", base_path)

    # Add each subdirectory as a path in the project
    project = add_paths(project, subdirs)

    # Configure image-processing settings
    settings = Settings(
        selected_file_extensions={"png", "jpg", "jpeg", "bmp", "tif", "tiff"},
        cmap="viridis",
        n_example_files=5,
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

    # Launch the interactive report
    logger.info("Opening report in your browser...")
    show_report(project)
