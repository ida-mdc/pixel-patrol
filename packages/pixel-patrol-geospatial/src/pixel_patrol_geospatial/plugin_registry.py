from pathlib import Path
from pixel_patrol_geospatial.geospatial_loader import GeospatialLoader

def register_loader_plugins():
    return [GeospatialLoader]


def get_viewer_extension_dir():
    return Path(__file__).parent / "viewer"
