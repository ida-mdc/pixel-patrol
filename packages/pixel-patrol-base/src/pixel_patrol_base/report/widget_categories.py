from enum import Enum, auto

class WidgetCategories(Enum):
    """
    Defines categories for organizing widgets in the application.
    These are logical groupings for different types of widget functionalities.
    """
    SUMMARY = "Summary"
    FILE_STATS = "File Stats"
    METADATA = "Metadata"
    VISUALIZATION = "Visualization"
    NOISE = "Noise"
    OTHER = "Other Widgets"
    DATASET_STATS = "Dataset Stats"
