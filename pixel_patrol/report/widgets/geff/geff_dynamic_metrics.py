from typing import List

from pixel_patrol.core.loaders.geff_loader import GeffLoader
from pixel_patrol.core.spec_provider import get_requirements_as_patterns
from pixel_patrol.report.widgets.base_dynamic_table_widget import BaseDynamicTableWidget


class GeffDynamicMetricsWidget(BaseDynamicTableWidget):
    def __init__(self):
        super().__init__(widget_id='geff-stats')

    @property
    def tab(self) -> str:
        return "Tracking"

    @property
    def name(self) -> str:
        return "GEFF metrics across dimensions"

    def required_columns(self) -> List[str]:
        return get_requirements_as_patterns(GeffLoader())

    def get_supported_metrics(self) -> List[str]:
        """
        Specifies that this widget uses metrics from the BasicStatsProcessor.
        """
        metrics = list(GeffLoader().get_specification().keys())
        metrics.append("geff_node_attr")
        metrics.append("geff_edge_attr")
        return metrics
