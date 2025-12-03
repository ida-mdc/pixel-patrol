from typing import List, Set

from pixel_patrol_base.plugins.widgets.metrics_across_dims_abstract_widget import MetricsAcrossDimensionsWidget
from pixel_patrol_base.core.feature_schema import patterns_from_processor
from pixel_patrol_base.report.widget_categories import WidgetCategories

from pixel_patrol_image.plugins.processors.quality_metrics_processor import QualityMetricsProcessor


class QualityMetricsAcrossDimensionsWidget(MetricsAcrossDimensionsWidget):
    NAME: str = "Quality Metrics Across Dimensions"
    TAB: str = WidgetCategories.DATASET_STATS.value

    # No fixed columns; rely on the processor's dynamic outputs
    REQUIRES: Set[str] = set()
    REQUIRES_PATTERNS: List[str] = patterns_from_processor(QualityMetricsProcessor)

    def __init__(self):
        super().__init__(widget_id="quality-stats")

    @property
    def help_text(self) -> str:
        return (
            "Shows how **image quality metrics** change across (e.g. T, C, Z, S) slices.\n\n"
            "Use this view to detect:\n"
            "- drift in focus or noise over time (T)\n"
            "- channel-specific artifacts (C)\n"
            "- depth-dependent quality changes (Z)\n"
        )

    def get_supported_metrics(self) -> List[str]:
        # Base metric names expected in dynamic columns (e.g., "snr", "focus", …)
        return list(getattr(QualityMetricsProcessor, "OUTPUT_SCHEMA", {}).keys())
