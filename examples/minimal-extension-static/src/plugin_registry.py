from my_loader import MarkdownDiaryLoader
from my_processor import MarkdownMoodProcessor


def register_loader_plugins():
    return [MarkdownDiaryLoader]


def register_processor_plugins():
    return [MarkdownMoodProcessor]
