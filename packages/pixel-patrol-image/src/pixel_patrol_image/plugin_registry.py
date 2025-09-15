from pixel_patrol_image.plugins.processors.quality_metrics_processor import QualityMetricsProcessor
from pixel_patrol_image.plugins.widgets.image_quality import ImageQualityWidget, DynamicQualityMetricsWidget

def register_processor_plugins():
    return [
        QualityMetricsProcessor,
    ]

def register_widget_plugins():
    return [
        ImageQualityWidget,
        DynamicQualityMetricsWidget,
    ]