#!/usr/bin/env python
import logging
import os
from pathlib import Path

from distributed import Client, LocalCluster

from pixel_patrol.api import (
    create_project,
    add_paths,
    process_paths,
    set_settings,
    process_images,
    get_images_df,
    show_report, export_project,
)
from pixel_patrol.core.project_settings import Settings

# — Configure Logging —
logging.basicConfig(
    level=logging.INFO,
    format="%((asctime)s)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    #
    # _dask_client = None  # Global variable to store client, or pass it around
    #
    #
    # def get_dask_client():
    #     global _dask_client
    #     if _dask_client is None or _dask_client.status == 'closed':
    #         try:
    #             cluster = LocalCluster(n_workers=os.cpu_count(),
    #                                    threads_per_worker=1,
    #                                    memory_limit='32GB',  # Example: Adjust based on your available RAM
    #                                    local_directory=os.environ.get('DASK_LOCAL_DIR',
    #                                                                   '/tmp/dask-worker-dir'))  # Use env var for custom dir
    #             _dask_client = Client(cluster)
    #             logger.info(f"Dask Distributed Client started. Dashboard: {_dask_client.dashboard_link}")
    #         except Exception as e:
    #             logger.error(
    #                 f"Failed to start Dask LocalCluster: {e}. Falling back to default Dask scheduler (which may be less efficient).")
    #             _dask_client = None  # Indicate fallback
    #     return _dask_client
    #
    #
    # # Call this at the start of your main script/application
    # client = get_dask_client()

    # Base folder containing multiple test-image subdirectories
    base_path = Path(__file__).resolve().parent / "data" / "zarr_image_datasets"
    logger.info(f"Scanning for immediate subdirectories under: {base_path}")

    # Define the output directory for the exported project
    output_directory = Path(__file__).parent / "exported_projects"
    output_directory.mkdir(parents=True, exist_ok=True)  # Ensure output directory exists

    # Define the path for the exported zip file
    exported_project_path = output_directory / "zarr_exported.zip"

    # Initialize project with the base directory
    project = create_project("Local Test Image Collections", base_path)

    project = add_paths(project, ["group1", "group2"])

    # Discover and process all image paths
    project = process_paths(project)

    # Configure image-processing settings
    settings = Settings(
        selected_file_extensions={"zarr", "tiff"},
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
