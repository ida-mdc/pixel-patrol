from typing import List, Set

from pixel_patrol_base.plugins.widgets.base_dynamic_table_widget import BaseDynamicTableWidget
from pixel_patrol_base.core.feature_schema import patterns_from_processor
from pixel_patrol_base.report.widget_categories import WidgetCategories

from pixel_patrol_image.plugins.processors.quality_metrics_processor import QualityMetricsProcessor


class DynamicQualityMetricsWidget(BaseDynamicTableWidget):
    NAME: str = "Quality metrics across dimensions"
    TAB: str = WidgetCategories.DATASET_STATS.value

    # No fixed columns; rely on the processor's dynamic outputs
    REQUIRES: Set[str] = set()
    REQUIRES_PATTERNS: List[str] = patterns_from_processor(QualityMetricsProcessor)

    def __init__(self):
        super().__init__(widget_id="quality-stats")

    def get_supported_metrics(self) -> List[str]:
        # Base metric names expected in dynamic columns (e.g., "snr", "focus", …)
        return list(getattr(QualityMetricsProcessor, "OUTPUT_SCHEMA", {}).keys())
