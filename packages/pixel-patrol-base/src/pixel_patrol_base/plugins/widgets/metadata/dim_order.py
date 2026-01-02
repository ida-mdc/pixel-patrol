from typing import Set

from pixel_patrol_base.report.widget_categories import WidgetCategories
from pixel_patrol_base.plugins.widgets.column_count_with_grouping_abstract_widget import (
    ColumnCountWithGroupingBarWidget,
)


class DimOrderWidget(ColumnCountWithGroupingBarWidget):
    NAME: str = "Dimension Order Distribution"
    TAB: str = WidgetCategories.METADATA.value
    REQUIRES: Set[str] = {"dim_order", "name"}
    REQUIRES_PATTERNS = None

    CATEGORY_COLUMN: str = "dim_order"
    CATEGORY_LABEL: str = "Dim Order"
    CONTENT_ID: str = "dim-order-content"

    @property
    def help_text(self) -> str:
        return (
            "Shows how often each **dimension ordering** (e.g., `TZYX`, `ZYX`, `CTZYX`) appears in the dataset.\n\n"
            "**Use this to detect**\n"
            "- inconsistent dimension layouts between/within groupings\n"
            "- files that may need reordering before analysis\n"
        )
