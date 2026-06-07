from pathlib import Path

from pixel_sky_watch.my_loader import SkyPatchLoader
from pixel_sky_watch.my_processor import StarSpotterProcessor


def register_loader_plugins():
    return [SkyPatchLoader]


def register_processor_plugins():
    return [StarSpotterProcessor]


def get_viewer_extension_dir():
    """Return the path to the bundled viewer extension directory."""
    return Path(__file__).parent / "viewer"
