import logging
import requests
import tarfile
import os
from pathlib import Path

from pixel_patrol_base.api import (
    create_project,
    add_paths,
    set_settings,
    process_artifacts,
    get_artifacts_df,
    show_report,
)
from pixel_patrol_base.core.project_settings import Settings

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- USER CONFIGURATION START ---
# Define your data's local base directory here.
# This path should be the parent directory containing 'condition1_org', 'condition2_bl', etc.
# e.g., if your conditions are in /my_data/plankton_processed/, set this to Path("/my_data/plankton_processed")
# If set to None, the script will attempt to download the example plankton data from DESY Sync & Share.
local_data_base_path = None
# Example: local_data_base_path = Path("/Users/youruser/my_plankton_data/plankton_filtered_processed")
# Example: local_data_base_path = Path(__file__).parent.parent / "my_existing_data" / "plankton_filtered_processed"

# The specific subdirectories within the chosen base path that contain the images you want to compare.
# These paths are relative to the 'local_data_base_path' or 'downloaded_processed_data_path'.
data_subdirectories_to_process = [
    "condition1_org",
    "condition2_bl",
    "condition3_comp",
    "condition4_nois"
]
# --- USER CONFIGURATION END ---


# --- Configuration for plankton download (Example Data) ---
plankton_sync_share_folder_link = "https://syncandshare.desy.de/index.php/s/wsNgyjpXJQMJpML"
plankton_file_name = "plankton_filtered_processed.tar.gz"
plankton_direct_download_url = f"{plankton_sync_share_folder_link}/download?path=/&files={plankton_file_name}"
download_filename = plankton_file_name

# Define intermediate path for where the tar.gz will be downloaded
downloaded_tar_path = Path(__file__).parent / download_filename

# Define the base directory for extraction. The tar.gz is expected to contain
# a top-level folder 'plankton_filtered_processed'.
# So, the final data will be in 'downloaded_plankton_data/plankton_filtered_processed'
extraction_base_dir = Path(__file__).parent / "downloaded_plankton_data"
downloaded_processed_data_path = extraction_base_dir / "plankton_filtered_processed"


if __name__ == "__main__":
    base_path_for_processing = None # This will be the parent path for add_paths

    # --- Determine Data Source ---
    # 1. Check if user-defined local data path is valid and contains data
    if local_data_base_path and local_data_base_path.is_dir() and any(local_data_base_path.iterdir()):
        logger.info(f"Using local data from: {local_data_base_path}")
        base_path_for_processing = local_data_base_path
    # 2. Else, check if example data has been previously downloaded and extracted
    elif downloaded_processed_data_path.is_dir() and any(downloaded_processed_data_path.iterdir()):
        logger.info(f"Using pre-downloaded example data from: {downloaded_processed_data_path}")
        base_path_for_processing = downloaded_processed_data_path
    # 3. Else, download and extract the example data
    else:
        logger.info(f"Neither local data nor pre-downloaded example data found. Initiating download...")

        # Download the plankton .tar.gz file
        logger.info(f"Attempting to download from: {plankton_direct_download_url}")
        try:
            response = requests.get(plankton_direct_download_url, stream=True)
            response.raise_for_status()

            with open(downloaded_tar_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            logger.info(f"Successfully downloaded '{download_filename}' to '{downloaded_tar_path}'.")

        except requests.exceptions.RequestException as e:
            logger.error(f"Error downloading file plankton data: {e}")
            logger.error("Please ensure the link is correct and the file name matches exactly.")
            exit(1)

        # Extract the .tar.gz file
        logger.info(f"Attempting to extract '{download_filename}' to '{extraction_base_dir}'.")
        try:
            extraction_base_dir.mkdir(parents=True, exist_ok=True)

            with tarfile.open(downloaded_tar_path, 'r:gz') as tar:
                tar.extractall(path=extraction_base_dir)
            logger.info(f"Successfully extracted '{download_filename}' to '{extraction_base_dir}'.")

            os.remove(downloaded_tar_path) # Clean up downloaded tar.gz
            logger.info(f"Removed downloaded file: {downloaded_tar_path}")

            base_path_for_processing = downloaded_processed_data_path

        except tarfile.ReadError as e:
            logger.error(f"Error reading or extracting tar.gz file: {e}")
            logger.error("This might happen if the downloaded file is corrupted or not a valid tar.gz archive.")
            exit(1)
        except Exception as e:
            logger.error(f"An unexpected error occurred during extraction: {e}")
            exit(1)

    # --- Final validation of data base path ---
    if not base_path_for_processing or not base_path_for_processing.is_dir() or not any(base_path_for_processing.iterdir()):
        logger.error("No valid data base directory could be determined or found. Please check your configuration.")
        exit(1)

    logger.info(f"Base path for processing data: {base_path_for_processing}")

    # --- Prepare specific paths to add to the project ---
    # These are the absolute paths to your condition directories
    paths_to_add = []
    for subdir in data_subdirectories_to_process:
        full_path = base_path_for_processing / subdir
        if full_path.is_dir():
            paths_to_add.append(full_path)
        else:
            logger.warning(f"Warning: Data subdirectory not found: {full_path}. It will be skipped.")

    if not paths_to_add:
        logger.error("No valid data subdirectories found to process. Please check 'data_subdirectories_to_process' and your data structure.")
        exit(1)

    logger.info(f"Processing the following directories: {[str(p) for p in paths_to_add]}")

    # 1. Create a new project (using API)
    my_project = create_project("Test Project for API Run", Path(__file__).parent.parent)

    # 2. Add the specific data directory paths (using API)
    # This loop ensures each specified subdirectory is added
    for path in paths_to_add:
        my_project = add_paths(my_project, path)

    # 4. Set relevant settings (e.g., image extensions) (using API)
    initial_settings = Settings(
        selected_file_extensions={"png", "tif", "tiff", "jpg", "jpeg"},
        cmap="viridis",
        n_example_files=5
    )
    my_project = set_settings(my_project, initial_settings)

    # 5. Process images to build the im-ages_df (using API)
    my_project = process_artifacts(my_project)

    # 6. Get and print the head of the images_df (using API)
    images_dataframe = get_artifacts_df(my_project)

    if images_dataframe is not None and not images_dataframe.is_empty():
        print("\n--- Images DataFrame Head ---")
        print(images_dataframe.head())
        print(f"\nTotal image files processed: {images_dataframe.height}")
    else:
        print("\nNo images DataFrame was generated or it is empty.")
        print("Please check your input data, file extensions settings, or the processing logic.")

    # 7. Show the report (using API) - Called without an output_path, as per your working script
    logger.info("API Call: Showing report for project 'Test Project for API Run'.")
    show_report(my_project)
    print("\nIf the report does not open automatically, try navigating to http://127.0.0.1:8050/ in your browser.")