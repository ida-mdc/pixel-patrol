from pathlib import Path

from pixel_patrol_labels.plugins.processors.label_processor import LabelProcessor


def register_processor_plugins():
    return [LabelProcessor]


def get_viewer_extension_dir():
    """Return the path to the bundled viewer extension directory."""
    return Path(__file__).parent / "viewer"