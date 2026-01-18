from typing import List, Set

from pixel_patrol_image.plugins.processors.quality_metrics_processor import QualityMetricsProcessor
from pixel_patrol_base.core.feature_schema import patterns_from_processor
from pixel_patrol_base.report.widget_categories import WidgetCategories
from pixel_patrol_base.plugins.widgets.multi_metric_violin_abstract_widget import MultiMetricViolinGridWidget


class QualityMetricsWidget(MultiMetricViolinGridWidget):
    # ---- Declarative spec ----
    NAME: str = "Image Quality Metrics"
    TAB: str = WidgetCategories.DATASET_STATS.value
    REQUIRES: Set[str] = QualityMetricsProcessor.OUTPUT_SCHEMA.keys()

    CONTENT_ID = "image-quality-container"

    BASE_METRIC_NAMES: List[str] = [
        "laplacian_variance",
        "tenengrad",
        "brenner",
        "noise_std",
        "blocking_records",
        "ringing_records",
    ]

    @property
    def help_text(self) -> str:
        return (
            "Visualizes **image quality metrics** as violin plots across groups.\n\n"
            "Use these plots to quickly spot outliers, compare image sets, and detect quality differences.\n\n"
            "**Metrics**\n"
            "- **Laplacian variance** – Edge-based sharpness estimate. Higher values indicate a sharper image.\n"
            "- **Tenengrad** – Focus measure based on Sobel gradients; captures overall edge strength.\n"
            "- **Brenner** – Measures fine structural detail using pixel intensity differences.\n"
            "- **Noise std** – Estimated pixel-level noise standard deviation; higher noise reduces clarity.\n"
            "- **Blocking records** – Strength of blocky compression artifacts (e.g. JPEG blocking).\n"
            "- **Ringing records** – Edge oscillation artifacts around sharp boundaries, often due to compression.\n\n"
        )
