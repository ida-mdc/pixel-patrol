from typing import List

from pixel_patrol.core.processors.basic_stats_processor import BasicStatsProcessor
from pixel_patrol.report.widget_categories import WidgetCategories
from pixel_patrol.report.widgets.base_dynamic_table_widget import BaseDynamicTableWidget
from pixel_patrol.core.spec_provider import get_requirements_as_patterns


class DynamicStatsWidget(BaseDynamicTableWidget):
    def __init__(self):
        super().__init__(widget_id='basic-stats')

    @property
    def tab(self) -> str:
        return WidgetCategories.DATASET_STATS.value

    @property
    def name(self) -> str:
        return "Basic Dynamic Statistics"

    def required_columns(self) -> List[str]:
        return get_requirements_as_patterns(BasicStatsProcessor())

    def get_supported_metrics(self) -> List[str]:
        return list(BasicStatsProcessor().get_specification().keys())
