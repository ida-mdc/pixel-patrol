from pixel_patrol_geospatial.geospatial_loader import GeoImageLoader
from pixel_patrol_geospatial.map_centroid_widget import CentroidGeoMapWidget
from pixel_patrol_geospatial.map_bbox_widget import BBoxGeoMapWidget

def register_loader_plugins():
    return [
        GeoImageLoader
    ]

def register_processor_plugins():
    return list()

def register_widget_plugins():
    return [
        CentroidGeoMapWidget,
        BBoxGeoMapWidget,
    ]