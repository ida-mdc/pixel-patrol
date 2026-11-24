from pixel_patrol_tomography.mrc_loader import MrcLoader


def register_loader_plugins() -> list[type]:
    return [MrcLoader]
