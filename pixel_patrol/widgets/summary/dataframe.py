from typing import List

import polars as pl

from pixel_patrol.widgets.widget_categories import WidgetCategories
from pixel_patrol.widgets.widget_interface import PixelPatrolWidget


class DataFrameWidget(PixelPatrolWidget): # Updated inheritance

    @property
    def category(self) -> str:
        return WidgetCategories.SUMMARY.value # Updated category

    @property
    def name(self) -> str:
        return "Dataframe Viewer"

    def required_columns(self) -> List[str]:
        return []

    def summary(self, data_frame: pl.DataFrame):
        """
        Provides a summary of the DataFrame content. (GUI-specific)
        """
        # print(f"DataFrame summary: {data_frame.describe()}") # Example for API-first logging
        pass

    # def render(self, data_frame: pl.DataFrame, *nargs):
    #     pass