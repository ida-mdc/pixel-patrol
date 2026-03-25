"""
Example demonstrating configuration options for processors and widgets.
"""

from pathlib import Path
from pixel_patrol_base import api
from pixel_patrol_base.core.processing_config import ProcessingConfig
from pixel_patrol_base.core.report_config import ReportConfig

def main():
    base_path = Path("datasets/bioio")
    zip_path = Path("out/configured_project.zip")
    loader = 'bioio'
    paths = [p.name for p in base_path.iterdir() if p.is_dir() and not p.name.startswith('.')]
    
    project = api.create_project("Configured Project", base_dir=base_path, loader=loader)
    api.add_paths(project, paths)
    
    # Processing configuration: include only specific processors
    # Other options:
    #   processors_excluded={"HistogramProcessor"}  # exclude instead of include
    #   slicing_enabled=False  # disable slicing entirely
    #   slicing_dimensions_included={"T", "C"}  # slice only specific dimensions
    #   slicing_dimensions_excluded={"Z"}  # exclude specific dimensions from slicing
    processing_config = ProcessingConfig(
        processors_included={"BasicStatsProcessor"}  # only run this processor
    )
    
    api.process_files(project, processing_config=processing_config)
    api.export_project(project, zip_path)
    
    # Report configuration: exclude specific widgets
    # Other options:
    #   widgets_included={"FileSummaryWidget", "DataFrameWidget"}  # include instead of exclude
    #   group_col="imported_path_short"  # group by column
    #   filter={"file_extension": {"op": "in", "value": "tif, png"}}  # filter rows
    #   dimensions={"T": "0", "Z": "1"}  # filter by dimensions
    report_config = ReportConfig(
        cmap='viridis',
        widgets_excluded={"ImageMosaikWidget"}
    )
    
    api.show_report(project, report_config=report_config)


if __name__ == "__main__":
    main()
