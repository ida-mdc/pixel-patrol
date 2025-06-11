import logging
from pathlib import Path
import polars as pl

# Import necessary API functions
from pixel_patrol.api import (
    import_project,
    get_images_df,
    get_paths_df,  # Also useful for verification
    get_name  # To confirm the project name
)

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Main Execution ---
if __name__ == "__main__":
    # Define the path to the previously exported project file
    # This should match the 'exported_project_path' from temp.py
    exported_project_path = Path(__file__).parent / "exported_projects" / "my_exported_project.zip"

    # Ensure the exported project file exists
    if not exported_project_path.exists():
        logger.error(f"Error: Exported project file not found at {exported_project_path}.")
        logger.error("Please run temp.py first to create the exported project zip file.")
        exit(1)

    logger.info(f"Attempting to import project from '{exported_project_path}'.")
    try:
        # 1. Import the project
        imported_project = import_project(exported_project_path)
        logger.info(f"Project '{get_name(imported_project)}' successfully imported.")

        # 2. Get the images_df from the imported project
        images_dataframe = get_images_df(imported_project)
        paths_dataframe = get_paths_df(imported_project)

        # 3. Print the head of the images_df
        if images_dataframe is not None and not images_dataframe.is_empty():
            print("\n--- Imported Images DataFrame Head ---")
            print(images_dataframe.head())
            print(f"\nTotal image files in imported project: {images_dataframe.height}")
        else:
            print("\nNo images DataFrame found in the imported project or it is empty.")

        # Optionally, print paths_df info to verify
        if paths_dataframe is not None and not paths_dataframe.is_empty():
            print("\n--- Imported Paths DataFrame Head ---")
            print(paths_dataframe.head())
            print(f"\nTotal paths in imported project: {paths_dataframe.height}")
        else:
            print("\nNo paths DataFrame found in the imported project or it is empty.")

    except Exception as e:
        logger.error(f"An error occurred during project import or data retrieval: {e}", exc_info=True)
        exit(1)

