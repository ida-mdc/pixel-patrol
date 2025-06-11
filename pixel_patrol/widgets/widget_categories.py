from enum import Enum, auto

class WidgetCategories(Enum):
    """
    Defines categories for organizing widgets in the application.
    These are logical groupings for different types of widget functionalities.
    """
    SUMMARY = auto()
    METADATA = auto()
