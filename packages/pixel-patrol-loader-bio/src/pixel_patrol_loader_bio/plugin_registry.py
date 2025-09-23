from pixel_patrol_loader_bio.plugins.loaders.bioio_loader import BioIoLoader
from pixel_patrol_loader_bio.plugins.loaders.zarr_loader import ZarrLoader

def register_loader_plugins():
    return [
        BioIoLoader,
        ZarrLoader,
    ]