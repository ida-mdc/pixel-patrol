# pixel_patrol/plugins/widgets/dynamic_stats_widget.py
from typing import List, Set

from pixel_patrol_base.plugins.widgets.base_dynamic_table_widget import BaseDynamicTableWidget
from pixel_patrol_base.plugins.processors.basic_stats_processor import BasicStatsProcessor
from pixel_patrol_base.core.feature_schema import patterns_from_processor
from pixel_patrol_base.report.widget_categories import WidgetCategories


class DynamicStatsWidget(BaseDynamicTableWidget):
    NAME: str = "Basic Dynamic Statistics"
    TAB: str = WidgetCategories.DATASET_STATS.value

    # No fixed columns; we require whatever dynamic columns the processor emits.
    REQUIRES: Set[str] = set()
    REQUIRES_PATTERNS: List[str] = patterns_from_processor(BasicStatsProcessor)

    def __init__(self):
        super().__init__(widget_id="basic-stats")

    def get_supported_metrics(self) -> List[str]:
        return list(getattr(BasicStatsProcessor, "OUTPUT_SCHEMA", {}).keys())
