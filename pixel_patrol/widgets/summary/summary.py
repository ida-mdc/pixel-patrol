from typing import List

import polars as pl

from pixel_patrol.widgets.widget_categories import WidgetCategories
from pixel_patrol.widgets.widget_interface import PixelPatrolWidget


class SummaryWidget(PixelPatrolWidget):

    @property
    def category(self) -> str:
        return WidgetCategories.SUMMARY.value

    @property
    def name(self) -> str:
        return "Summary"

    def required_columns(self) -> List[str]:
        return [
            "n_images", "mean_intensity", "std_intensity", "median_intensity",
            "x_size", "dtype", "file_extension", "min_intensity", "max_intensity"
        ]

    def summary(self, data_frame: pl.DataFrame): #TODO: Implement summary
        print("Testing summary method in SummaryWidget")
        pass

    # def render(self, df: pl.DataFrame, *nargs):
    #     pass