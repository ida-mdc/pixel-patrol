from typing import List, Dict, Set

import polars as pl
from dash import html, dcc, Input, Output, no_update
from pixel_patrol_base.plugins.processors.basic_stats_processor import BasicStatsProcessor

from pixel_patrol_base.report.widget_categories import WidgetCategories
from pixel_patrol_base.report.global_controls import (
    apply_global_config,
    GLOBAL_CONFIG_STORE_ID,
)
from pixel_patrol_base.report.base_widget import BaseReportWidget
from pixel_patrol_base.report.factory import create_labeled_dropdown, plot_violin
from pixel_patrol_base.report.data_utils import find_best_matching_column


class DatasetStatsWidget(BaseReportWidget):
    NAME: str = "Pixel Value Statistics"
    TAB: str = WidgetCategories.DATASET_STATS.value
    REQUIRES: Set[str] = {"name"}
    REQUIRES_PATTERNS = None

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._df: pl.DataFrame | None = None

    @property
    def help_text(self) -> str:
        return (
            "Shows **per-image intensity statistics** across groups.\n\n"
            "You can choose which statistic to plot and filter by image dimensions.\n\n"
            "In the plot each point is one image; the box shows the distribution per group.\n\n"
            "**Statistics**\n"
            "Pairwise group comparisons use a Mann–Whitney U test with Bonferroni correction:\n"
            "- `ns`: not significant\n"
            "- `*: p < 0.05`, `**: p < 0.01`, `***: p < 0.001`"
        )

    def get_content_layout(self) -> List:
        return [
            html.P(id="dataset-stats-warning", className="text-warning", style={"marginBottom": "15px"}),

            # --- Controls Section ---
            html.Div([
                create_labeled_dropdown(
                    label="Select metric to plot:",
                    component_id="stats-value-to-plot-dropdown",
                    options=[],
                    value=None,
                    width="100%"  # Utilize full width
                ),
            ], style={"marginBottom": "20px"}),

            # --- Plot Section ---
            dcc.Graph(id="stats-violin-chart", style={"height": "600px"}),
        ]

    def register(self, app, df: pl.DataFrame):

        self._df = df

        app.callback(
            Output("stats-value-to-plot-dropdown", "options"),
            Output("stats-value-to-plot-dropdown", "value"),
            Input("color-map-store", "data"),
            prevent_initial_call=False,
        )(self._set_control_options)

        app.callback(
            Output("stats-violin-chart", "figure"),
            Output("dataset-stats-warning", "children"),
            Input("color-map-store", "data"),
            Input("stats-value-to-plot-dropdown", "value"),
            Input(GLOBAL_CONFIG_STORE_ID, "data"),
        )(self._update_plot)

    def _set_control_options(self, _color_map: Dict[str, str]):
        df = self._df

        processor_metrics = (
            list(BasicStatsProcessor.OUTPUT_SCHEMA.keys())
            if hasattr(BasicStatsProcessor, 'OUTPUT_SCHEMA')
            else []
        )
        numeric_candidates = [
            col for col in df.columns
            if df[col].dtype.is_numeric()
        ]
        dropdown_options = [{'label': m, 'value': m} for m in processor_metrics] if processor_metrics else [
            {'label': col, 'value': col} for col in numeric_candidates]

        if processor_metrics:
            default_value_to_plot = next(iter(processor_metrics))
        elif numeric_candidates:
            default_value_to_plot = next(iter(numeric_candidates))
        else:
            default_value_to_plot = None

        return dropdown_options, default_value_to_plot

    def _update_plot(
            self,
            color_map: Dict[str, str],
            value_to_plot: str,
            global_config: Dict,
    ):
        df = self._df

        if not value_to_plot:
            return no_update, "Please select a value to plot."

        # Get dimensions from global store
        selections = global_config.get("dimensions", {})

        df_processed, group_col = apply_global_config(df, global_config)

        chosen_col = (
                find_best_matching_column(df_processed.columns, value_to_plot, selections)
                or value_to_plot
        )

        plot_data = df_processed.filter(pl.col(chosen_col).is_not_null())

        if plot_data.is_empty():
            return (
                no_update,
                html.P(
                    f"No valid data found for '{value_to_plot}'.",
                    className="text-warning",
                ),
            )

        warning_message = ""

        chart = plot_violin(
            df=plot_data,
            y=chosen_col,
            group_col=group_col,
            color_map=color_map or {},
            custom_data_cols=["name"],
            title=None,
            height=600,
        )

        return chart, warning_message
