from typing import List

import polars as pl

from pixel_patrol.widgets.widget_categories import WidgetCategories
from pixel_patrol.widgets.widget_interface import PixelPatrolWidget


class DimOrderWidget(PixelPatrolWidget):  # Updated inheritance

    @property
    def category(self) -> str:
        return WidgetCategories.METADATA.value  # Updated category

    def required_columns(self) -> List[str]:
        return ["dim_order"]

    def summary(self, data_frame: pl.DataFrame): # TODO: Implement summary
        """
        Provides a summary of dimension order distribution. (GUI-specific)
        """
        print(f"Dim order distribution: {data_frame['dim_order'].value_counts()}")
        pass

    def render(self, selected_files_df: pl.DataFrame, *nargs):
        pass
