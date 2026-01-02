from typing import List, Dict, Set

import polars as pl
from dash import html, dcc, Input, Output
from pixel_patrol_base.plugins.processors.basic_stats_processor import BasicStatsProcessor

from pixel_patrol_base.report.widget_categories import WidgetCategories
from pixel_patrol_base.report.global_controls import (
    prepare_widget_data,
    resolve_group_column,
    GLOBAL_CONFIG_STORE_ID,
    FILTERED_INDICES_STORE_ID,
)
from pixel_patrol_base.report.base_widget import BaseReportWidget
from pixel_patrol_base.report.factory import (
    create_labeled_dropdown,
    plot_violin,
    show_no_data_message,
)
from pixel_patrol_base.report.data_utils import format_selection_title, get_dim_aware_column


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
            Input(FILTERED_INDICES_STORE_ID, "data"),
            Input(GLOBAL_CONFIG_STORE_ID, "data"),
        )(self._update_plot)

    def _set_control_options(self, _color_map: Dict[str, str]):
        df = self._df

        schema = df.schema

        # Identify numeric columns using the schema dict
        numeric_candidates = [
            col for col, dtype in schema.items()
            if dtype in (
                pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
                pl.Float32, pl.Float64
            )
        ]

        processor_metrics = (
            list(BasicStatsProcessor.OUTPUT_SCHEMA.keys())
            if hasattr(BasicStatsProcessor, 'OUTPUT_SCHEMA')
            else []
        )

        # If processor metrics are defined, prioritize them. Otherwise fallback to all numerics.
        if processor_metrics:
            # Only include metrics that actually exist in the dataframe
            valid_metrics = [m for m in processor_metrics if m in schema]
            # If no processor metrics found (e.g. processor didn't run), fallback
            if not valid_metrics:
                options_list = numeric_candidates
            else:
                options_list = valid_metrics
        else:
            options_list = numeric_candidates

        # Build dropdown options (limit to 200 to prevent browser lag if fallback is used)
        if len(options_list) > 200 and not processor_metrics:
            options_list = options_list[:200]

        dropdown_options = [{'label': m, 'value': m} for m in options_list]

        if dropdown_options:
            default_col_to_plot = dropdown_options[0]['value']
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
        global_config = global_config or {}
        dims_selection = global_config.get("dimensions", {})

        needed_cols = {"unique_id", "name"}

        # Resolve group column manually to select it
        group_col = resolve_group_column(self._df, global_config)
        if group_col in self._df.columns:
            needed_cols.add(group_col)

        if col_to_plot:
            resolved_col = get_dim_aware_column(self._df.columns, col_to_plot, dims_selection)
            if resolved_col:
                needed_cols.add(resolved_col)

        df_slim = self._df.select([c for c in needed_cols if c in self._df.columns])

        df_filtered, final_group_col, final_resolved_col, warning_msg = prepare_widget_data(
            df_slim,
            subset_indices,
            global_config,
            metric_base=col_to_plot,
        )

        if final_resolved_col is None or df_filtered.is_empty():
            return show_no_data_message()

        chart = plot_violin(
            df=df_filtered,
            y=final_resolved_col,
            group_col=final_group_col,
            color_map=color_map or {},
            custom_data_cols=["name"],
            title=format_selection_title(dims_selection),
            height=600,
        )

        return dcc.Graph(figure=chart, style={"height": "600px"})
