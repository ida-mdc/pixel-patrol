from pixel_patrol_loader_bio.plugins.loaders.bioio_loader import BioIoLoader
from pixel_patrol_loader_bio.plugins.loaders.h5_loader import H5Loader
from pixel_patrol_loader_bio.plugins.loaders.tifffile_loader import TifffileLoader
from pixel_patrol_loader_bio.plugins.loaders.zarr_loader import ZarrLoader


def register_loader_plugins():
    return [
        BioIoLoader,
        H5Loader,
        TifffileLoader,
        ZarrLoader,
    ]
