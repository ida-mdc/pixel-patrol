from typing import List

from pixel_patrol_base.core.processors.quality_metrics_processor import QualityMetricsProcessor
from pixel_patrol_base.core.spec_provider import get_requirements_as_patterns
from pixel_patrol_base.report.widget_categories import WidgetCategories
from pixel_patrol_base.report.widgets.base_dynamic_table_widget import BaseDynamicTableWidget


class DynamicQualityMetricsWidget(BaseDynamicTableWidget):
    def __init__(self):
        super().__init__(widget_id='quality-stats')

    @property
    def tab(self) -> str:
        return WidgetCategories.DATASET_STATS.value

    @property
    def name(self) -> str:
        return "Quality metrics across dimensions"

    def required_columns(self) -> List[str]:
        return get_requirements_as_patterns(QualityMetricsProcessor())

    def get_supported_metrics(self) -> List[str]:
        """
        Specifies that this widget uses metrics from the BasicStatsProcessor.
        """
        return list(QualityMetricsProcessor().get_specification().keys())
