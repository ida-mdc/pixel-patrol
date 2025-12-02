import itertools
from pathlib import Path
from typing import List, Dict, Set

import plotly.graph_objects as go
import polars as pl
import statsmodels.stats.multitest as smm
from dash import html, dcc, Input, Output, ALL
from pixel_patrol_base.plugins.processors.basic_stats_processor import BasicStatsProcessor
from scipy.stats import mannwhitneyu

from pixel_patrol_base.report.widget_categories import WidgetCategories
from pixel_patrol_base.report.global_controls import (
    apply_global_config,
    GLOBAL_CONFIG_STORE_ID,
)
from pixel_patrol_base.report.base_widget import BaseReportWidget
# Import UI Factory controls
from pixel_patrol_base.report.factory import create_dimension_selectors, create_labeled_dropdown


class DatasetStatsWidget(BaseReportWidget):
    NAME: str = "Pixel Value Statistics"
    TAB: str = WidgetCategories.DATASET_STATS.value
    REQUIRES: Set[str] = {"imported_path", "name"}
    REQUIRES_PATTERNS = None

    @property
    def help_text(self) -> str:
        return (
            "### Description of the test\n"
            "The selected representation of intensities within an image is plotted on the y-axis, "
            "while the x-axis shows the different groups (folders) selected.\n\n"
            "Each image is represented by a dot, and the boxplot shows the distribution of the selected value.\n\n"
            "**Statistical hints:**\n"
            "The symbols (* or ns) indicate significance (Mann-Whitney U test with Bonferroni correction).\n"
            "- ns: not significant\n"
            "- *: p < 0.05\n"
            "- **: p < 0.01\n"
            "- ***: p < 0.001"
        )

    def get_content_layout(self) -> List:
        return [
            html.P(id="dataset-stats-warning", className="text-warning", style={"marginBottom": "15px"}),
            html.Div([
                create_labeled_dropdown(
                    label="Select value to plot:",
                    id="stats-value-to-plot-dropdown",
                    options=[],
                    value=None
                ),
                html.Div(id="stats-filters-container"),
                dcc.Store(id="stats-dims-store")
            ]),
            dcc.Graph(id="stats-violin-chart", style={"height": "600px"}),
        ]

    def register(self, app, df_global: pl.DataFrame):
        @app.callback(
            Output("stats-value-to-plot-dropdown", "options"),
            Output("stats-value-to-plot-dropdown", "value"),
            Output("stats-filters-container", "children"),
            Output("stats-dims-store", "data"),
            Input("color-map-store", "data"),
            prevent_initial_call=False,
        )
        def set_stats_dropdown_options(color_map: Dict[str, str]):
            processor_metrics = list(BasicStatsProcessor.OUTPUT_SCHEMA.keys()) if hasattr(BasicStatsProcessor,
                                                                                          'OUTPUT_SCHEMA') else []
            numeric_candidates = [
                col for col in df_global.columns
                if df_global[col].dtype.is_numeric()
            ]
            dropdown_options = [{'label': m, 'value': m} for m in processor_metrics] if processor_metrics else [
                {'label': col, 'value': col} for col in numeric_candidates]
            default_value_to_plot = processor_metrics[0] if processor_metrics else (
                numeric_candidates[0] if numeric_candidates else None)

            children = []
            dims_order = []
            if default_value_to_plot:
                # Use Utils to analyze columns
                from pixel_patrol_base.report.data_utils import extract_dimension_tokens
                # Use Factory to generate UI components
                children, dims_order = create_dimension_selectors(
                    tokens=extract_dimension_tokens(df_global.columns, default_value_to_plot),
                    id_type="stats-dim-filter"
                )

            return dropdown_options, default_value_to_plot, children, dims_order

        @app.callback(
            Output("stats-violin-chart", "figure"),
            Output("dataset-stats-warning", "children"),
            Input("color-map-store", "data"),
            Input("stats-value-to-plot-dropdown", "value"),
            Input({"type": "stats-dim-filter", "dim": ALL}, "value"),
            Input("stats-dims-store", "data"),
            Input(GLOBAL_CONFIG_STORE_ID, "data"),
        )
        def update_stats_chart(
                color_map: Dict[str, str],
                value_to_plot: str,
                dim_values_list,
                dims_order,
                global_config: Dict,
        ):
            if not value_to_plot:
                return go.Figure(), "Please select a value to plot."

            selections = {}
            if dims_order and dim_values_list:
                for dim_name, val in zip(dims_order, dim_values_list):
                    selections[dim_name] = val

            df_processed, group_col = apply_global_config(df_global, global_config)

            from pixel_patrol_base.report.data_utils import find_best_matching_column
            chosen_col = (
                    find_best_matching_column(df_processed.columns, value_to_plot, selections)
                    or value_to_plot
            )

            plot_data = df_processed.filter(pl.col(chosen_col).is_not_null())

            if plot_data.is_empty():
                return (
                    go.Figure(),
                    html.P(f"No valid data found for '{value_to_plot}'.", className="text-warning"),
                )

            warning_message = ""
            chart = go.Figure()
            groups = plot_data.get_column(group_col).unique().to_list()
            groups.sort()

            for group_name in groups:
                df_group = plot_data.filter(pl.col(group_col) == group_name)
                group_color = color_map.get(group_name, "#333333") if group_col == "imported_path_short" else None

                chart.add_trace(go.Violin(
                    y=df_group.get_column(chosen_col).to_list(),
                    name=group_name,
                    customdata=df_group.get_column("name").to_list(),
                    opacity=0.9,
                    showlegend=True,
                    points="all",
                    pointpos=0,
                    box_visible=True,
                    meanline=dict(visible=True),
                    marker_color=group_color,
                    hovertemplate="<b>Group: %{x}</b><br>Value: %{y:.2f}<br>File: %{customdata}<extra></extra>"
                ))

            chart.update_traces(marker=dict(line=dict(width=1, color="black")), box=dict(line_color="black"))
            chart.update_layout(
                title_text=f"Distribution of {value_to_plot.replace('_', ' ').title()}",
                xaxis_title="Group",
                yaxis_title=value_to_plot,
                height=600,
                margin=dict(l=50, r=50, t=80, b=100),
                template="plotly_white"
            )
            return chart, warning_message