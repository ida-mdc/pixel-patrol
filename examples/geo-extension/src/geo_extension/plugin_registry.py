from geo_extension.geoimageloader import GeoImageLoader

def register_loader_plugins():
    return [
        GeoImageLoader
    ]

def register_processor_plugins():
    return list()

def register_widget_plugins():
    return list()