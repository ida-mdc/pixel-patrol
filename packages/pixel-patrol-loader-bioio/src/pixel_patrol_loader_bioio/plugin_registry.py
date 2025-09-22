from pixel_patrol_loader_bioio.plugins.loaders.bioio_loader import BioIoLoader

def register_loader_plugins():
    return [
        BioIoLoader,
    ]