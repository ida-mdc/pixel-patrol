from pathlib import Path


def get_viewer_extension_dir():
    """Return the path to the bundled viewer extension directory."""
    return Path(__file__).parent / "viewer"
