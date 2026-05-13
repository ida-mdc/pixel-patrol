from pathlib import Path

from markdown_diary_tracker_static.my_loader import MarkdownDiaryLoader
from markdown_diary_tracker_static.my_processor import MarkdownMoodProcessor


def register_loader_plugins():
    return [MarkdownDiaryLoader]


def register_processor_plugins():
    return [MarkdownMoodProcessor]


def get_viewer_extension_dir():
    """Return the path to the bundled viewer extension directory."""
    return Path(__file__).parent / "viewer"
