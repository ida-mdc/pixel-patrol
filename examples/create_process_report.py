import logging
from pathlib import Path

from pixel_patrol_base.api import (
    create_project,
    add_paths,
    set_settings,
    process_files,
    get_artifacts_df, show_report,
)
from pixel_patrol_base.core.project_settings import Settings

# --- Configure Logging ---
logger = logging.getLogger(__name__)

# --- Main Execution ---
if __name__ == "__main__":
    # Define the path to your test data directory relative to this script
    test_data_directory = Path(__file__).parent.parent / "ella_extras" / "data"

    # Define the output directory for the exported project
    output_directory = Path(__file__).parent / "exported_projects"
    output_directory.mkdir(parents=True, exist_ok=True)  # Ensure output directory exists

    # Define the path for the exported zip file
    exported_project_path = output_directory / "my_exported_project1.zip"

    # Ensure the test data directory exists (optional, for robustness)
    if not test_data_directory.is_dir():
        logger.error(f"Error: Test data directory not found at {test_data_directory}. Please ensure it exists.")
        exit(1)

    # 1. Create a new project (using API)
    my_project = create_project("Test Project for API Run", Path(__file__).parent.parent)

    # 2. Add the test data directory paths (using API)
    my_project = add_paths(my_project, test_data_directory)

    # 4. Set relevant settings (e.g., image extensions) (using API)
    initial_settings = Settings(
        selected_file_extensions={"png", "tif", "tiff", "jpg", "jpeg"},
        cmap="viridis",
        n_example_files=5
    )
    my_project = set_settings(my_project, initial_settings)

    # 5. Process images to build the images_df (using API)
    my_project = process_files(my_project)

    # 6. Get and print the head of the images_df (using API)
    images_dataframe = get_artifacts_df(my_project)

    if images_dataframe is not None and not images_dataframe.is_empty():
        print("\n--- Images DataFrame Head ---")
        print(images_dataframe.head())
        print(f"\nTotal image files processed: {images_dataframe.height}")
    else:
        print("\nNo images DataFrame was generated or it is empty.")
        print("Please check your input data, file extensions settings, or the processing logic.")

    show_report(my_project)
