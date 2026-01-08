from pathlib import Path
from pixel_patrol_base import api
from pixel_patrol_base.core.project_settings import Settings
import logging
logging.basicConfig(level=logging.INFO)

def main():

    # --- Step 1: choose a base directory with files to be processed, sub-paths, and loader ---
    base_path = Path("datasets/bioio")

    # Optional: Define sub-dirs that in the report are compared as experimental conditions
    # If you don't specify any paths, all files in base_path and its subfolders are processed as one condition.
    # You can specify the paths either as absolute or relevant to base_path.
    paths = [p.name for p in base_path.iterdir() if p.is_dir() and not p.name.startswith('.')]
    # OR e.g.
    # paths = ['/home/ella/work/pixel-patrol/examples/datasets/bioio/pngs', 'tifs', 'jpgs'] # those are relative paths inside the base_path
    # OR e.g.
    # paths = []

    loader = 'bioio'  # for image files (e.g. png, jpg, tiff, etc.); requires pixel-patrol-loader-bio package
    # OR e.g.
    # loader = None    # for basic file info only (no image data/metadata); only pixel-patrol-base package needed
    # loader = 'zarr'   # for zarr files; requires pixel-patrol-loader-zarr package

    selected_file_extensions = "all" # if loader is None all file types are processed, otherwise all file types supported by the loader
    # OR e.g.
    # selected_file_extensions = {"tif", "png"} # only those file types are processed

    # --- Step 2: create a project ---
    project = api.create_project("Quickstart Project", base_dir=base_path, loader=loader)

    # --- Step 3: (optional) add sub-paths (inside base_path) ---
    api.add_paths(project, paths)

    # --- Step 4: set basic settings (e.g. file types to process) ---
    api.set_settings(project, Settings(selected_file_extensions=selected_file_extensions))

    # --- Step 5: process files ---
    # This step creates a dataframe with file information, and if available metadata and data (e.g. the image itself) metrics.
    api.process_files(project)
    records_df = api.get_records_df(project)
    print(records_df.head())

    # --- Step 6: export project ---
    zip_path = Path("out/quickstart_project.zip")
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    api.export_project(project, zip_path) # project exports to zip_path 'out/'

    # --- Step 7: open the dashport report ---
    # Open http://127.0.0.1:8050/ in your browser
    api.show_report(project)

    ## OR start report already with applied filters.
    # api.show_report(
    #     project,
    #     global_config={
    #         "group_cols": ["size_readable"],
    #         "filter": {
    #             "file_extension": {
    #                 "op": "in",
    #                 "value": "tif, png",
    #             }
    #         },
    #         "dimensions": {
    #             "c":"0"
    #         },
    #     },
    # )

    # --- Step 8: (optional) import project ---
    imported = api.import_project(zip_path)
    print("Imported:", imported.name)

if __name__ == "__main__":
    main()
