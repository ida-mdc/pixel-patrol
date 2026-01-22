from typing import List, Set

from pixel_patrol_base.plugins.processors.basic_stats_processor import BasicStatsProcessor
from pixel_patrol_base.report.widget_categories import WidgetCategories
from pixel_patrol_base.plugins.widgets.multi_metric_violin_abstract_widget import MultiMetricViolinGridWidget
from pixel_patrol_base.report.constants import SIGNIFICANCE_HELP_TEXT


class DatasetStatsWidget(MultiMetricViolinGridWidget):
    NAME: str = "Pixel Value Statistics"
    TAB: str = WidgetCategories.DATASET_STATS.value
    REQUIRES: Set[str] = {"name"}
    REQUIRES_PATTERNS = None
    CONTENT_ID = "stats-content-container"
    BASE_METRIC_NAMES: List[str] = list(getattr(BasicStatsProcessor, "OUTPUT_SCHEMA", {}).keys())

    @property
    def help_text(self) -> str:
        return (
            "Shows **per-image intensity statistics** across groups.\n\n"
            "You can choose which statistic to plot and filter by image dimensions.\n\n"
            "In the plot each point is one image; the box shows the distribution per group.\n\n"
            + SIGNIFICANCE_HELP_TEXT
        )
