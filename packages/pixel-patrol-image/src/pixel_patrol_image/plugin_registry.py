def register_processor_plugins():
    from pixel_patrol_image.plugins.processors.raster_image_processor import (
        QualityMetricsProcessor,
        CompressionMetricsProcessor,
    )
    return [
        QualityMetricsProcessor,
        CompressionMetricsProcessor,
    ]


def register_widget_plugins():
    return []
