"""
Example demonstrating configuration options for processors and widgets.
"""

from pathlib import Path
from pixel_patrol_base import api
from pixel_patrol_base.core.processing_config import ProcessingConfig
from pixel_patrol_base.core.report_config import ReportConfig

def main():
    base_path = Path("datasets/bioio")
    output_path = Path("out/configured_project.parquet")
    loader = 'bioio'
    paths = [p.name for p in base_path.iterdir() if p.is_dir() and not p.name.startswith('.')]
    
    project = api.create_project("Configured Project", base_dir=base_path, loader=loader, output_path=output_path)
    api.add_paths(project, paths)
    
    # Processing configuration: include only specific processors
    # Other options:
    #   processors_excluded={"HistogramProcessor"}  # exclude instead of include
    processing_config = ProcessingConfig(
        processors_included={"basic-stats"},
    )
    
    api.process_files(project, processing_config=processing_config)

    # --- Report: exclude specific widgets ---
    # Other options:
    #   widgets_included={"FileSummaryWidget", "DataFrameWidget"}  — include instead of exclude
    #   group_col="imported_path_short"                            — group by column
    #   filter={"file_extension": {"op": "in", "value": "tif, png"}}
    #   dimensions={"T": "0", "Z": "1"}                           — filter by dimensions
    report_config = ReportConfig(
        cmap="viridis",
        widgets_excluded={"image-mosaic-widget"},
    )

    api.show_report(project, report_config=report_config)


if __name__ == "__main__":
    main()
