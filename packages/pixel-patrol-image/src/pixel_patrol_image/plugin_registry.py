from pixel_patrol_image.plugins.processors.quality_metrics_processor import QualityMetricsProcessor
from pixel_patrol_image.plugins.widgets.quality_metrics import QualityMetricsWidget
from pixel_patrol_image.plugins.widgets.quality_metrics_across_dims import QualityMetricsAcrossDimensionsWidget

def register_processor_plugins():
    return [
        QualityMetricsProcessor,
    ]

def register_widget_plugins():
    return [
        QualityMetricsWidget,
        QualityMetricsAcrossDimensionsWidget,
    ]