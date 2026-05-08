from pixel_patrol_image.plugins.processors.channel_colocalization_processor import ChannelColocalizationProcessor
from pixel_patrol_image.plugins.processors.raster_image_dask_processor import RasterImageDaskProcessor
from pixel_patrol_image.plugins.processors.thumbnail_processor import ThumbnailProcessor

def register_processor_plugins():
    return [
        RasterImageDaskProcessor,
        ChannelColocalizationProcessor,
        ThumbnailProcessor,
    ]

def register_widget_plugins():
    return [
    ]