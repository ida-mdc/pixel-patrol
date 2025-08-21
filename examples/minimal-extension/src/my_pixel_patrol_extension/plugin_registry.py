from my_pixel_patrol_extension.my_loader import MarkdownDiaryLoader
from my_pixel_patrol_extension.my_processor import MarkdownMoodProcessor
from my_pixel_patrol_extension.my_widget1 import DiaryWordCloudWidget
from my_pixel_patrol_extension.my_widget2 import MoodTrendWidget

def register_loader_plugins():
    return [
        MarkdownDiaryLoader
    ]

def register_processor_plugins():
    return [
        MarkdownMoodProcessor
    ]

def register_widget_plugins():
    return [
        DiaryWordCloudWidget,
        MoodTrendWidget
    ]
