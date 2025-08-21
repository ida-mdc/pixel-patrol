from typing import List, Set

from pixel_patrol.plugins.base_dynamic_table_widget import BaseDynamicTableWidget
from pixel_patrol.plugins.processors.quality_metrics_processor import QualityMetricsProcessor
from pixel_patrol_base.core.feature_schema import patterns_from_processor, get_requirements_as_patterns
from pixel_patrol_base.report.widget_categories import WidgetCategories
from traxel_patrol.loaders.geff_loader import GeffLoader


class DynamicGeffMetricsWidget(BaseDynamicTableWidget):
    NAME: str = "GEFF metrics across dimensions"
    TAB: str = "Tracking"

    REQUIRES: Set[str] = set()
    REQUIRES_PATTERNS: List[str] = get_requirements_as_patterns(GeffLoader)

    def __init__(self):
        super().__init__(widget_id="geff-stats")

    def get_supported_metrics(self) -> List[str]:
        # Base metric names expected in dynamic columns (e.g., "snr", "focus", …)
        return list(getattr(GeffLoader, "OUTPUT_SCHEMA", {}).keys())
