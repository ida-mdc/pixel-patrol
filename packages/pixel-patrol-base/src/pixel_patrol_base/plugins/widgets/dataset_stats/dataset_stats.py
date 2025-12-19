from typing import List, Dict, Set

import polars as pl
from dash import html, dcc, Input, Output
from pixel_patrol_base.plugins.processors.basic_stats_processor import BasicStatsProcessor

from pixel_patrol_base.report.widget_categories import WidgetCategories
from pixel_patrol_base.report.global_controls import (
    prepare_widget_data,
    GLOBAL_CONFIG_STORE_ID,
    FILTERED_INDICES_STORE_ID,
)
from pixel_patrol_base.report.base_widget import BaseReportWidget
from pixel_patrol_base.report.factory import (
    create_labeled_dropdown,
    plot_violin,
    show_no_data_message,
)
from pixel_patrol_base.report.data_utils import format_selection_title


class DatasetStatsWidget(BaseReportWidget):
    NAME: str = "Pixel Value Statistics"
    TAB: str = WidgetCategories.DATASET_STATS.value
    REQUIRES: Set[str] = {"name"}
    REQUIRES_PATTERNS = None
    CONTENT_ID = "stats-content-container"

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
            # --- Controls Section ---
            html.Div([
                create_labeled_dropdown(
                    label="Select metric to plot:",
                    component_id="stats-value-to-plot-dropdown",
                    options=[],
                    value=None,
                    width="100%"
                ),
            ], style={"marginBottom": "20px"}),

            # --- Plot Section (Dynamic Container) ---
            html.Div(id=self.CONTENT_ID, style={"minHeight": "400px"}),
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
            Output(self.CONTENT_ID, "children"),
            Input("color-map-store", "data"),
            Input("stats-value-to-plot-dropdown", "value"),
            Input(FILTERED_INDICES_STORE_ID, "data"),  # NEW: filtered rows
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
            default_col_to_plot = next(iter(processor_metrics))
        elif numeric_candidates:
            default_col_to_plot = next(iter(numeric_candidates))
        else:
            default_col_to_plot = None

        return dropdown_options, default_col_to_plot


    def _update_plot(
        self,
        color_map: Dict[str, str],
        col_to_plot: str,
        subset_indices: List[int] | None,
        global_config: Dict,
    ):

        df_filtered, group_col, resolved_col, warning_msg = prepare_widget_data(
            self._df,
            subset_indices,
            global_config or {},
            metric_base=col_to_plot,
        )

        if resolved_col is None or df_filtered.is_empty():
            return show_no_data_message()

        dims_selection = (global_config or {}).get("dimensions", {}) or {}

        chart = plot_violin(
            df=df_filtered,
            y=resolved_col,  # use the dimension-resolved column
            group_col=group_col,
            color_map=color_map or {},
            custom_data_cols=["name"],
            title=format_selection_title(dims_selection),
            height=600,
        )

        return dcc.Graph(figure=chart, style={"height": "600px"})

