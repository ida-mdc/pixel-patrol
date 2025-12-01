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
## New:
from pixel_patrol_base.report.base_widget import BaseReportWidget


class DatasetStatsWidget(BaseReportWidget):
    # ---- Declarative spec (plugin registry expects these at class-level) ----
    NAME: str = "Pixel Value Statistics"
    TAB: str = WidgetCategories.DATASET_STATS.value
    REQUIRES: Set[str] = {"imported_path", "name"}
    REQUIRES_PATTERNS = None

    # ... existing properties ...
    @property
    def tab(self) -> str:
        return WidgetCategories.DATASET_STATS.value

    @property
    def name(self) -> str:
        return "Pixel Value Statistics"

    def required_columns(self) -> List[str]:
        return ["mean", "median", "std", "min", "max", "name", "imported_path"]

    @property
    def help_text(self) -> str:
        return (
            "### Description of the test\n"
            "The selected representation of intensities within an image is plotted on the y-axis, "
            "while the x-axis shows the different groups (folders) selected.\n\n"
            "Each image is represented by a dot, and the boxplot shows the distribution of the selected value.\n\n"
            "**Images with >2 dimensions:**\n"
            "As images can contain multiple time points (t), channels (c), and z-slices (z), "
            "statistics are calculated across all dimensions. Select specific slices (e.g. `mean_intensity_t0`) "
            "to narrow down.\n\n"
            "**Statistical hints:**\n"
            "The symbols (* or ns) indicate significance (Mann-Whitney U test with Bonferroni correction).\n"
            "- ns: not significant\n"
            "- *: p < 0.05\n"
            "- **: p < 0.01\n"
            "- ***: p < 0.001"
        )

    def get_content_layout(self) -> List:
        """Defines the layout of the Pixel Value Statistics widget."""
        return [
            html.P(id="dataset-stats-warning", className="text-warning", style={"marginBottom": "15px"}),
            html.Div([
                html.Label("Select value to plot:"),
                dcc.Dropdown(
                    id="stats-value-to-plot-dropdown",
                    options=[],
                    value=None,
                    clearable=False,
                    style={"width": "300px", "marginTop": "10px", "marginBottom": "20px"}
                ),
                html.Div(id="stats-filters-container"),
                dcc.Store(id="stats-dims-store")
            ]),
            dcc.Graph(id="stats-violin-chart", style={"height": "600px"}),
            # Markdown description removed from here (moved to help_text)
        ]

    ##

    def register(self, app, df_global: pl.DataFrame):
        # ... existing code ...
        # Populate dropdown options dynamically
        @app.callback(
            Output("stats-value-to-plot-dropdown", "options"),
            Output("stats-value-to-plot-dropdown", "value"),
            Output("stats-filters-container", "children"),
            Output("stats-dims-store", "data"),
            Input("color-map-store", "data"),
            prevent_initial_call=False,
        )
        def set_stats_dropdown_options(color_map: Dict[str, str]):
            # Determine available base metrics from the processor declaration if possible
            processor_metrics = list(BasicStatsProcessor.OUTPUT_SCHEMA.keys()) if hasattr(BasicStatsProcessor,
                                                                                          'OUTPUT_SCHEMA') else []
            # Fallback: find numeric columns resembling metrics
            numeric_candidates = [
                col for col in df_global.columns
                if df_global[col].dtype.is_numeric()
            ]
            # Build dropdown options (prefer processor-declared metrics if available)
            dropdown_options = [{'label': m, 'value': m} for m in processor_metrics] if processor_metrics else [
                {'label': col, 'value': col} for col in numeric_candidates]
            default_value_to_plot = processor_metrics[0] if processor_metrics else (
                numeric_candidates[0] if numeric_candidates else None)

            # Build dynamic filter dropdowns for the chosen base metric using shared helper
            children = []
            dims_order = []
            if default_value_to_plot:
                from pixel_patrol_base.report.utils import build_dimension_dropdown_children
                children, dims_order = build_dimension_dropdown_children(
                    df_global.columns, base=default_value_to_plot, id_type="stats-dim-filter"
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

            # Reconstruct selection dict from dims_order and values
            selections = {}
            if dims_order and dim_values_list:
                for dim_name, val in zip(dims_order, dim_values_list):
                    selections[dim_name] = val

            # Apply global filters & grouping (creates __global_group)
            df_processed, group_col = apply_global_config(df_global, global_config)

            # Find best matching column for the chosen metric on the processed df
            from pixel_patrol_base.report.utils import find_best_matching_column
            chosen_col = (
                    find_best_matching_column(df_processed.columns, value_to_plot, selections)
                    or value_to_plot
            )

            # Drop rows without the chosen metric
            plot_data = df_processed.filter(pl.col(chosen_col).is_not_null())

            if plot_data.is_empty():
                return (
                    go.Figure(),
                    html.P(
                        f"No valid data found for '{value_to_plot}' with current global settings.",
                        className="text-warning",
                    ),
                )

            warning_message = ""
            chart = go.Figure()

            # Groups come from the global grouping column
            groups = (
                plot_data.get_column(group_col)
                .unique()
                .to_list()
            )
            groups.sort()

            for group_name in groups:
                df_group = plot_data.filter(pl.col(group_col) == group_name)
                data_values = df_group.get_column(chosen_col).to_list()
                file_names = df_group.get_column("name").to_list()
                file_names_short = [
                    str(Path(x).name) if x is not None else "Unknown File"
                    for x in file_names
                ]

                # If grouping is still by imported_path_short, use color_map;
                # otherwise let Plotly handle colors.
                if group_col == "imported_path_short":
                    group_color = color_map.get(group_name, "#333333")
                else:
                    group_color = None

                violin_kwargs = dict(
                    y=data_values,
                    name=group_name,
                    customdata=file_names_short,
                    opacity=0.9,
                    showlegend=True,
                    points="all",
                    pointpos=0,
                    box_visible=True,
                    meanline=dict(visible=True),
                    hovertemplate=(
                        "<b>Group: %{x}</b><br>"
                        "Value: %{y:.2f}<br>"
                        "Filename: %{customdata}<extra></extra>"
                    ),
                )
                if group_color is not None:
                    violin_kwargs["marker_color"] = group_color

                chart.add_trace(go.Violin(**violin_kwargs))

            chart.update_traces(
                marker=dict(line=dict(width=1, color="black")),
                box=dict(line_color="black"),
            )

            # Statistical annotations (Mann-Whitney U + Bonferroni), now using group_col
            if len(groups) > 1:
                comparisons = list(itertools.combinations(groups, 2))
                p_values = []
                for group1, group2 in comparisons:
                    data1 = (
                        plot_data.filter(pl.col(group_col) == group1)
                        .get_column(chosen_col)
                        .to_list()
                    )
                    data2 = (
                        plot_data.filter(pl.col(group_col) == group2)
                        .get_column(chosen_col)
                        .to_list()
                    )
                    if len(data1) > 0 and len(data2) > 0:
                        _, p_val = mannwhitneyu(
                            data1, data2, alternative="two-sided"
                        )
                        p_values.append(p_val)
                    else:
                        p_values.append(1.0)

                if p_values:
                    _, pvals_corrected, _, _ = smm.multipletests(
                        p_values, alpha=0.05, method="bonferroni"
                    )
                else:
                    pvals_corrected = []

                chart.update_layout(
                    xaxis=dict(categoryorder="array", categoryarray=groups)
                )

                positions = {group: i for i, group in enumerate(groups)}
                overall_y_min = plot_data.get_column(chosen_col).min()
                overall_y_max = plot_data.get_column(chosen_col).max()
                y_range = overall_y_max - overall_y_min
                y_offset = y_range * 0.05 if y_range > 0 else 1.0

                annotation_y_levels = {g: overall_y_max for g in groups}
                comparisons_to_annotate = [
                    (groups[i], groups[i + 1])
                    for i in range(len(groups) - 1)
                ]

                for i, (group1, group2) in enumerate(comparisons_to_annotate):
                    try:
                        original_idx = comparisons.index((group1, group2))
                    except ValueError:
                        original_idx = comparisons.index((group2, group1))

                    p_corr = (
                        pvals_corrected[original_idx]
                        if original_idx < len(pvals_corrected)
                        else 1.0
                    )
                    if p_corr < 0.001:
                        sig = "***"
                    elif p_corr < 0.01:
                        sig = "**"
                    elif p_corr < 0.05:
                        sig = "*"
                    else:
                        sig = "ns"

                    y_max1 = (
                        plot_data.filter(pl.col(group_col) == group1)
                        .get_column(chosen_col)
                        .max()
                    )
                    y_max2 = (
                        plot_data.filter(pl.col(group_col) == group2)
                        .get_column(chosen_col)
                        .max()
                    )

                    current_y_level = max(
                        annotation_y_levels.get(group1, overall_y_max),
                        annotation_y_levels.get(group2, overall_y_max),
                    )
                    y_bracket = max(y_max1, y_max2, current_y_level) + y_offset
                    annotation_y_levels[group1] = y_bracket + y_offset
                    annotation_y_levels[group2] = y_bracket + y_offset

                    pos1 = positions[group1]
                    pos2 = positions[group2]
                    x_offset_line = 0.05

                    chart.add_shape(
                        type="line",
                        x0=pos1 + x_offset_line,
                        x1=pos2 - x_offset_line,
                        y0=y_bracket,
                        y1=y_bracket,
                        line=dict(color="black", width=1.5),
                        xref="x",
                        yref="y",
                    )
                    chart.add_shape(
                        type="line",
                        x0=pos1 + x_offset_line,
                        x1=pos1 + x_offset_line,
                        y0=y_bracket,
                        y1=y_bracket - y_offset / 2,
                        line=dict(color="black", width=1.5),
                        xref="x",
                        yref="y",
                    )
                    chart.add_shape(
                        type="line",
                        x0=pos2 - x_offset_line,
                        x1=pos2 - x_offset_line,
                        y0=y_bracket,
                        y1=y_bracket - y_offset / 2,
                        line=dict(color="black", width=1.5),
                        xref="x",
                        yref="y",
                    )
                    x_mid = (pos1 + pos2) / 2
                    chart.add_annotation(
                        x=x_mid,
                        y=y_bracket + y_offset / 4,
                        text=sig,
                        showarrow=False,
                        font=dict(color="black"),
                        xref="x",
                        yref="y",
                    )

            chart.update_layout(
                title_text=f"Distribution of {value_to_plot.replace('_', ' ').title()}",
                xaxis_title="Group",
                yaxis_title=value_to_plot.replace("_", " ").title(),
                height=600,
                margin=dict(l=50, r=50, t=80, b=100),
                hovermode="closest",
                legend=dict(
                    orientation="h",
                    yanchor="top",
                    y=-0.15,
                    xanchor="center",
                    x=0.5,
                ),
            )

            return chart, warning_message