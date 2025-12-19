from typing import List, Set

from pixel_patrol_base.plugins.widgets.metrics_across_dims_abstract_widget import MetricsAcrossDimensionsWidget
from pixel_patrol_base.plugins.processors.basic_stats_processor import BasicStatsProcessor
from pixel_patrol_base.core.feature_schema import patterns_from_processor
from pixel_patrol_base.report.widget_categories import WidgetCategories


class DatasetStatsAcrossDimensionsWidget(MetricsAcrossDimensionsWidget):
    NAME: str = "Basic Statistics Across Dimensions"
    TAB: str = WidgetCategories.DATASET_STATS.value

    REQUIRES: Set[str] = set()
    REQUIRES_PATTERNS: List[str] = patterns_from_processor(BasicStatsProcessor)

    def __init__(self):
        super().__init__(widget_id="basic-stats")

    @property
    def help_text(self) -> str:
        return (
            "Shows how image statistics (e.g., mean, std, min, max) change **across different dimension slices**.\n\n"
            "Useful for identifying drift, artifacts, or unexpected variation within (e.g.) T/C/Z/S dimensions.\n\n"
            "You can select slices in the dropdowns to filter the tables.\n"
        )

    def get_supported_metrics(self) -> List[str]:
        return list(getattr(BasicStatsProcessor, "OUTPUT_SCHEMA", {}).keys())
