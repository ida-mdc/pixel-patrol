from pixel_patrol_tomography.axis_profile_processor import AxisProfileProcessor
from pixel_patrol_tomography.mrc_loader import MrcLoader


def register_loader_plugins() -> list[type]:
    return [MrcLoader]


def register_processor_plugins() -> list[type]:
    return [AxisProfileProcessor]
