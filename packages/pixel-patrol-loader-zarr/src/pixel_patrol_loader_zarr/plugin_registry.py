from pixel_patrol_loader_zarr.plugins.loaders.zarr_loader import ZarrLoader

def register_loader_plugins():
    return [
        ZarrLoader,
    ]