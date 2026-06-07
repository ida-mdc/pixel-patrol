from pathlib import Path

from pixel_patrol_hai_watch.my_loader import SharkCamLoader
from pixel_patrol_hai_watch.my_processor import GlowSpotterProcessor


def register_loader_plugins():
    return [SharkCamLoader]


def register_processor_plugins():
    return [GlowSpotterProcessor]


def get_viewer_extension_dir():
    """Return the path to the bundled viewer extension directory."""
    return Path(__file__).parent / "viewer"
