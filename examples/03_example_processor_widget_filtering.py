"""
Example demonstrating configuration options for processors and widgets.
"""

from pathlib import Path
from pixel_patrol_base import api

def main():
    base_path = Path("datasets/bioio")
    output_path = Path("out/configured_project.parquet")
    loader = 'bioio'
    paths = [p.name for p in base_path.iterdir() if p.is_dir() and not p.name.startswith('.')]
    
    project = api.create_project("Configured Project", base_dir=base_path, loader=loader, output_path=output_path)
    api.add_paths(project, paths)

    api.process_files(project, processors_included={"basic-stats"})

    # --- Report: exclude specific widgets ---
    # Other options:
    #   widgets_included={"FileSummaryWidget", "DataFrameWidget"}  — include instead of exclude
    #   group_col="imported_path_short"                            — group by column
    #   filter_by={"file_extension": {"op": "in", "value": "tif, png"}}
    #   dimensions={"T": "0", "Z": "1"}                           — filter by dimensions
    api.show_report(
        project,
        cmap="viridis",
        widgets_excluded={"image-mosaic-widget"},
    )


if __name__ == "__main__":
    main()
