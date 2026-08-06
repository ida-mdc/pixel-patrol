from pathlib import Path
from pixel_patrol_geospatial.geospatial_loader import GeospatialLoader
from pixel_patrol_geospatial.nodata_processor import NoDataCountProcessor

def register_loader_plugins():
    return [GeospatialLoader]


def get_viewer_extension_dir():
    return Path(__file__).parent / "viewer"

def register_processor_plugins():
    return [NoDataCountProcessor]