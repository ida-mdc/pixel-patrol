from typing import List, Dict

import polars as pl
from dash import html, Input, Output

from pixel_patrol.core.processors.basic_stats_processor import BasicStatsProcessor
from pixel_patrol.core.spec_provider import get_requirements_as_patterns
from pixel_patrol.report.utils import generate_column_violin_plots
from pixel_patrol.report.widget_categories import WidgetCategories
from pixel_patrol.report.widget_interface import PixelPatrolWidget


class DatasetStatsWidget(PixelPatrolWidget):
    @property
    def tab(self) -> str:
        return WidgetCategories.DATASET_STATS.value

    @property
    def name(self) -> str:
        return "Pixel Value Statistics"

    def required_columns(self) -> List[str]:
        return get_requirements_as_patterns(BasicStatsProcessor())

    def layout(self) -> List:
        """Defines the new layout with a container for all plots and tables."""
        return [
            # This single container will be populated by the callback
            html.Div(id="dataset-stats-container"),
            #
            # # The static markdown content remains unchanged
            # html.Div(className="markdown-content", children=[
            #     html.H4("Description of the plots"),
            #     html.P([
            #         html.Strong("Selectable values to plot: "),
            #         "The selected representation of intensities within an image is plotted on the y-axis, while the x-axis shows the different groups (folders) selected. This is calculated on each individual image in the selected folders."
            #     ]),
            #     html.P([
            #         "Each image is represented by a dot, and the boxplot shows the distribution of the selected value for each group."
            #     ]),
            #     html.P([
            #         html.Strong("Images with more than 2 dimensions: "),
            #         "As images can contain multiple time points (t), channels (c), and z-slices (z), the statistics are calculated across all dimensions. To e.g. visualize the distribution of mean intensities across all z-slices and channels at time point t0, please select e.g. ",
            #         html.Code("mean_intensity_t0"), "."
            #     ]),
            #     html.P([
            #         "If you want to display the mean intensity across the whole image, select ",
            #         html.Code("mean_intensity"),
            #         " (without any suffix)."
            #     ]),
            #     html.P([
            #         html.Strong("Higher dimensional images that include RGB data: "),
            #         "When an image with Z-slices or even time points contains RGB data, the S-dimension is added. Therefore, the RGB color is indicated by the suffix ",
            #         html.Code("s0"), ", ", html.Code("s1"), ", and ", html.Code("s2"),
            #         " for red, green, and blue channels, respectively. This allows for images with multiple channels, where each channels consists of an RGB image itself, while still being able to select the color channel."
            #     ]),
            #     html.P([
            #         "The suffixes are as follows:", html.Br(),
            #         html.Ul([
            #             html.Li(html.Code("t: time point")),
            #             html.Li(html.Code("c: channel")),
            #             html.Li(html.Code("z: z-slice")),
            #             html.Li(html.Code("s: color in RGB images (red, green, blue)"))
            #         ])
            #     ]),
            #     html.H4("Statistical hints:"),
            #     html.P([
            #         "The symbols (", html.Code("*"), " or ", html.Code("ns"),
            #         ") shown above indicate the significance of the differences between two groups, with more astersisk indicating a more significant difference. The Mann-Whitney U test is applied to compare the distributions of the selected value between pairs of groups. This non-parametric test is used as a first step to assess whether the distributions of two independent samples. The results are adjusted with a Bonferroni correction to account for multiple comparisons, reducing the risk of false positives."
            #     ]),
            #     html.P([
            #         "Significance levels:", html.Br(),
            #         html.Ul([
            #             html.Li(html.Code("ns: not significant")),
            #             html.Li(html.Code("*: p < 0.05")),
            #             html.Li(html.Code("**: p < 0.01")),
            #             html.Li(html.Code("***: p < 0.001"))
            #         ])
            #     ]),
            #     html.H5("Disclaimer:"),
            #     html.P(
            #         "Please do not interpret the results as a final conclusion, but rather as a first step to assess the differences between groups. This may not be the appropriate test for your data, and you should always consult a statistician for a more detailed analysis.")
            # ])
        ]


    def register_callbacks(self, app, df_global: pl.DataFrame):
        """Registers a single callback to generate all plots and tables."""

        @app.callback(
            Output("dataset-stats-container", "children"),
            Input("color-map-store", "data")
        )
        def update_dataset_stats_layout(color_map: Dict[str, str]):

            numeric_cols = BasicStatsProcessor().get_specification().keys()

            return generate_column_violin_plots(df_global, color_map, numeric_cols)
