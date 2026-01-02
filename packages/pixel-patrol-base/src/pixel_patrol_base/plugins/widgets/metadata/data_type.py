from typing import Set

from pixel_patrol_base.report.widget_categories import WidgetCategories
from pixel_patrol_base.plugins.widgets.column_count_with_grouping_abstract_widget import (
    ColumnCountWithGroupingBarWidget,
)


class DataTypeWidget(ColumnCountWithGroupingBarWidget):
    NAME: str = "Data Type Distribution"
    TAB: str = WidgetCategories.METADATA.value
    REQUIRES: Set[str] = {"dtype", "name"}
    REQUIRES_PATTERNS = None

    CATEGORY_COLUMN: str = "dtype"
    CATEGORY_LABEL: str = "Data Type"
    CONTENT_ID: str = "data-type-content"

    @property
    def help_text(self) -> str:
        return (
            "Shows the distribution of **pixel data types** "
            "(e.g., `uint8`, `uint16`, `float32`) across groupings.\n\n"
            "**Use this to check**\n"
            "- whether different images/groupings use different numeric formats\n"
            "- potential range differences (e.g., 0–255 vs. arbitrary floats)\n"
        )
