from pixel_patrol_image.plugins.processors.quality_metrics_processor import QualityMetricsProcessor

def register_processor_plugins():
    return [
        QualityMetricsProcessor,
    ]