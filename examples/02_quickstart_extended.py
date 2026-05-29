from pathlib import Path
from pixel_patrol_base import api

import logging
logging.basicConfig(level=logging.INFO)

def main():

    # Choose a base directory with files to be processed, sub-paths, and loader ---
    base_path = Path("datasets/bioio")
    # output path for parquet file
    output_path = Path("out/quickstart_extended.parquet")

    # Optional: Define sub-dirs that in the report are compared as experimental conditions
    # If you don't specify any paths, all files in base_path and its subfolders are processed as one condition.
    # You can specify the paths either as absolute or relevant to base_path.
    paths = [p.name for p in base_path.iterdir() if p.is_dir() and not p.name.startswith('.')]
    # OR e.g.
    # paths = ['/home/yourusername/work/pixel-patrol/examples/datasets/bioio/pngs', 'tifs', 'jpgs'] # those are relative paths inside the base_path

    loader = 'bioio'  # for image files (e.g. png, jpg, tiff, etc.); requires pixel-patrol-loader-bio package
    # OR e.g.
    # loader = None    # for basic file info only (no image data/metadata); only pixel-patrol-base package needed
    # loader = 'zarr'   # for zarr files; requires pixel-patrol-loader-zarr package

    project = api.create_project("Quickstart Extended", base_dir=base_path, loader=loader, output_path=output_path)
    api.add_paths(project, paths)
    api.process_files(
        project,
        # --- Parallelism ---
        max_workers=4,          # Dask worker count; None = auto (all CPUs)
                                # To use an external Dask cluster (e.g. SLURM):
                                #   connect before calling: with Client("tcp://host:8786"): api.process_files(...)
        # --- Task sizing ---
        mb_per_task=512,        # MB budget per Dask task.
                                # Default (512 MB) works well for small files.
                                # For containers with large images (e.g. 6 MP), use 50 MB or less
                                # to keep individual task durations under ~2 minutes.
        # --- File selection ---
        selected_file_extensions={"tif", "png", "jpeg"},
        # --- Report metadata ---
        flavor="Example Datasets",
        description="Authors: Annona Buddha and Banana Java",
    )
    # Tip: run with --log-file from the CLI to write a debug log alongside the parquet:
    #   pixel-patrol process datasets/bioio --output out/quickstart_extended.parquet --log-file

    # get the results dataframe
    records_df = api.get_records_df(project)
    print(records_df.head())

    # --- View with filters and grouping ---
    api.view(
        project,
        group_col="size_readable",
        filter_by={"file_extension": {"op": "in", "value": "tif, png"}},
        dimensions={"c": "0"},   # show channel 0 by default
    )

    # --- Load a previously saved project ---
    records_df, metadata = api.load(output_path)
    print(f"Loaded: {metadata.project_name}  (v{metadata.version}, {metadata.description})")

if __name__ == "__main__":
    main()
