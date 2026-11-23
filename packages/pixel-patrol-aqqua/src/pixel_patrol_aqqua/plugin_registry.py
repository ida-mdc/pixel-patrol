from pixel_patrol_aqqua.plugins.loaders.lmdb_loader import LmdbLoader
from pixel_patrol_aqqua.plugins.processors.scale_bar_processor import ScaleBarProcessor

def register_loader_plugins():
    return [
        LmdbLoader
    ]

def register_processor_plugins():
    return [
        ScaleBarProcessor,
    ]
