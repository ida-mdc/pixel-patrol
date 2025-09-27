from typing import List, Dict, Set

import plotly.express as px
import polars as pl
from dash import html, dcc, Input, Output

from pixel_patrol_base.report.widget_categories import WidgetCategories


class DimSizeWidget:
    # ---- Declarative spec ----
    NAME: str = "Dimension Size Distribution"
    TAB: str = WidgetCategories.METADATA.value
    REQUIRES: Set[str] = {"imported_path_short", "name"}    # used for grouping/hover
    REQUIRES_PATTERNS: List[str] = [r"^[a-zA-Z]_size$"]     # dynamic size columns

    # Component IDs
    INFO_ID = "dim-size-info"
    XY_AREA_ID = "xy-size-plot-area"
    INDV_AREA_ID = "individual-dim-plots-area"

    def layout(self) -> List:
        """Defines the layout of the Dimension Size Distribution widget."""
        return [
            html.Div(id=self.INFO_ID, style={"marginBottom": "15px"}),
            html.Div(
                id=self.XY_AREA_ID,
                children=[html.P("No valid data to plot for X and Y dimension sizes.")],
            ),
            html.Div(id=self.INDV_AREA_ID),
        ]

    def register(self, app, df_global: pl.DataFrame):
        """Registers callbacks for the Dimension Size Distribution widget."""
        @app.callback(
            Output(self.INFO_ID, "children"),
            Output(self.XY_AREA_ID, "children"),
            Output(self.INDV_AREA_ID, "children"),
            Input("color-map-store", "data"),
        )
        def update_dim_size_charts(color_map: Dict[str, str]):
            color_map = color_map or {}

            # --- 1) Identify all *_size columns present ---
            dimension_size_cols = [col for col in df_global.columns if col.endswith("_size")]
            if not dimension_size_cols:
                return [html.P("No dimension size columns (e.g., 'X_size') found in data.")], [], []

            # Pre-filter rows where at least one size dimension is a valid number (>1)
            filtered_df = df_global.filter(
                pl.any_horizontal([pl.col(c).is_not_null() & (pl.col(c) > 1) for c in dimension_size_cols])
            )

            # --- 2) X/Y bubble chart (if both X_size and Y_size exist) ---
            x_col, y_col = "X_size", "Y_size"
            xy_plot_children = [
                html.P("No valid data for X/Y plot. Requires 'X_size' and 'Y_size' columns with data > 1.")
            ]
            if x_col in filtered_df.columns and y_col in filtered_df.columns:
                xy_plot_data = filtered_df.filter((pl.col(x_col) > 1) & (pl.col(y_col) > 1))
                if xy_plot_data.height > 0:
                    bubble_data_agg = (
                        xy_plot_data
                        .group_by([x_col, y_col, "imported_path_short"])
                        .agg(
                            pl.count().alias("bubble_size"),
                            pl.col("name").unique().alias("names_in_group"),
                        )
                        .sort([x_col, y_col, "imported_path_short"])
                    )

                    fig_bubble = px.scatter(
                        bubble_data_agg,
                        x=x_col,
                        y=y_col,
                        size="bubble_size",
                        color="imported_path_short",
                        color_discrete_map=color_map,
                        title="Distribution of X and Y Dimension Sizes",
                        labels={"bubble_size": "Count", "imported_path_short": "Folder"},
                        hover_data=["names_in_group"],
                    )
                    fig_bubble.update_layout(
                        height=500,
                        margin=dict(l=50, r=50, t=80, b=100),
                        hovermode="closest",
                        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5),
                        template="plotly",
                    )
                    xy_plot_children = [dcc.Graph(figure=fig_bubble)]

            # --- 3) Individual dimension strip plots (facet per *_size) ---
            melted_df = (
                filtered_df
                .unpivot(
                    index=["imported_path_short", "name"],
                    on=dimension_size_cols,
                    variable_name="dimension_name",
                    value_name="dimension_value",
                )
                .filter(pl.col("dimension_value") > 1)
            )

            if melted_df.height == 0:
                individual_dim_plots = [html.P("No data to plot for individual dimension sizes.")]
            else:
                fig_strip = px.strip(
                    melted_df,
                    x="imported_path_short",
                    y="dimension_value",
                    color="imported_path_short",
                    facet_col="dimension_name",
                    facet_col_wrap=3,
                    color_discrete_map=color_map,
                    facet_row_spacing=0.16,
                    title="Individual Dimension Sizes per Dataset",
                    labels={"dimension_value": "Size", "imported_path_short": "Folder"},
                    hover_data=["name"],
                )
                # Show numeric tick labels on all subplots
                fig_strip.update_xaxes(matches=None, showticklabels=True)
                fig_strip.update_yaxes(matches=None, showticklabels=True)
                fig_strip.for_each_annotation(
                    lambda a: a.update(text=a.text.replace("dimension_name=", "").replace("_size", " Size"))
                )
                fig_strip.update_layout(
                    height=200 * ((len(dimension_size_cols) + 2) // 3),
                    margin=dict(l=50, r=50, t=80, b=80),
                    showlegend=False,
                    template="plotly",
                )
                individual_dim_plots = [dcc.Graph(figure=fig_strip)]

            # --- 4) Availability ratios for each *_size column ---
            ratio_spans: List = []
            total_files = df_global.height
            for column in dimension_size_cols:
                present = filtered_df.filter(pl.col(column) > 1).height
                text = (
                    f"{column.replace('_', ' ').title()}: {present} of {total_files} files "
                    f"({(present / total_files) * 100:.2f}%)."
                    if total_files > 0 else f"{column.replace('_', ' ').title()}: No files."
                )
                ratio_spans.extend([html.Span(text), html.Br()])

            info_children = [html.P(html.B("Data Availability by Dimension:")), html.P(ratio_spans)]

            return info_children, xy_plot_children, individual_dim_plots
