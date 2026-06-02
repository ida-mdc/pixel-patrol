from pixel_patrol_loader_tifffile.plugins.loaders.tifffile_loader import TifffileLoader


def register_loader_plugins():
    return [
        TifffileLoader,
    ]
