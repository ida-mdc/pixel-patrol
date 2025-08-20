from typing import List, Dict

import plotly.express as px
import polars as pl
from dash import html, dcc, Input, Output

from pixel_patrol.report.widget_categories import WidgetCategories
from pixel_patrol.report.widget_interface import PixelPatrolWidget


class DimSizeWidget(PixelPatrolWidget):

    @property
    def tab(self) -> str:
        return WidgetCategories.METADATA.value

    @property
    def name(self) -> str:
        return "Dimension Size Distribution"

    def required_columns(self) -> List[str]:
        return [r'^[a-zA-Z]_size$']

    def layout(self) -> List:
        """Defines the layout of the Dimension Size Distribution widget."""
        return [
            html.Div(id="dim-size-info", style={"marginBottom": "15px"}),
            html.Div(id="xy-size-plot-area", children=[
                html.P("No valid data to plot for X and Y dimension sizes.")
            ]),
            # This container will hold the new strip plots
            html.Div(id="individual-dim-plots-area")
        ]

    def register_callbacks(self, app, df_global: pl.DataFrame):
        """Registers callbacks for the Dimension Size Distribution widget."""

        @app.callback(
            Output("dim-size-info", "children"),
            Output("xy-size-plot-area", "children"),
            Output("individual-dim-plots-area", "children"),
            Input("color-map-store", "data"),
        )
        def update_dim_size_charts(color_map: Dict[str, str]):
            # --- 1. Performance: Pre-filter the DataFrame ---
            # Identify all dimension size columns available in the global dataframe.
            dimension_size_cols = [col for col in df_global.columns if col.endswith('_size')]

            # If no size columns exist, exit early.
            if not dimension_size_cols:
                return [html.P("No dimension size columns (e.g., 'X_size') found in data.")], [], []

            # ✨ PERFORMANCE: Create one filtered df. We only need rows where at least one
            # size dimension is a valid number (>1). This avoids cloning and repeated filtering.
            filtered_df = df_global.filter(
                pl.any_horizontal([
                    pl.col(c).is_not_null() & (pl.col(c) > 1) for c in dimension_size_cols
                ])
            )

            # --- 2. X and Y Size Distribution (Bubble Chart) ---
            x_col, y_col = "X_size", "Y_size"
            xy_size_plot_children = [
                html.P("No valid data for X/Y plot. Requires 'X_size' and 'Y_size' columns with data > 1.")]

            # Use the pre-filtered dataframe, which is much smaller.
            if x_col in filtered_df.columns and y_col in filtered_df.columns:
                # Further filter for rows where *both* X and Y are valid.
                xy_plot_data = filtered_df.filter(
                    (pl.col(x_col) > 1) & (pl.col(y_col) > 1)  # .is_not_null() is redundant now
                )

                if xy_plot_data.height > 0:
                    bubble_data_agg = xy_plot_data.group_by(
                        [x_col, y_col, "imported_path_short"]
                    ).agg(
                        pl.count().alias("bubble_size"),
                        pl.col("name").unique().alias("names_in_group")
                    ).sort([x_col, y_col, "imported_path_short"])

                    fig_bubble = px.scatter(
                        bubble_data_agg, x=x_col, y=y_col, size='bubble_size',
                        color='imported_path_short', color_discrete_map=color_map,
                        title="Distribution of X and Y Dimension Sizes",
                        labels={'bubble_size': 'Count', 'imported_path_short': 'Folder'},
                    )
                    fig_bubble.update_layout(
                        height=500, margin=dict(l=50, r=50, t=80, b=100),
                        hovermode='closest',
                        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5),
                        template='plotly'
                    )
                    xy_size_plot_children = [dcc.Graph(figure=fig_bubble)]

            # --- 3. Individual Dimension Strip Plots ---
            # Melt the already filtered dataframe for maximum efficiency.
            melted_df = filtered_df.unpivot(
                index=["imported_path_short", "name"],
                on=dimension_size_cols,
                variable_name="dimension_name",
                value_name="dimension_value"
            ).filter(
                pl.col("dimension_value") > 1  # .is_not_null() is redundant here as well
            )

            if melted_df.height == 0:
                individual_dim_plots = [html.P("No data to plot for individual dimension sizes.")]
            else:
                fig_strip = px.strip(
                    melted_df, x="dimension_value", y="imported_path_short",
                    color="imported_path_short", facet_col="dimension_name",
                    facet_col_wrap=3, color_discrete_map=color_map,
                    title="Individual Dimension Sizes per Dataset",
                    labels={"dimension_value": "", "imported_path_short": "Folder"},
                    hover_data=['name']
                )

                # ✨ VISIBILITY: Show numeric tick labels on ALL subplots, not just the bottom ones.
                fig_strip.update_xaxes(matches=None, showticklabels=True)
                fig_strip.update_yaxes(matches=None)  # Keep y-axes independent
                fig_strip.for_each_annotation(
                    lambda a: a.update(text=a.text.replace("dimension_name=", "").replace("_size", " Size")))
                fig_strip.update_layout(
                    height=200 * ((len(dimension_size_cols) + 2) // 3),
                    margin=dict(l=50, r=50, t=80, b=80), showlegend=False, template='plotly'
                )
                individual_dim_plots = [dcc.Graph(figure=fig_strip)]

            # --- 4. Data Availability Ratios ---
            all_ratios_text_components = []
            col_total_files = df_global.height  # Total count comes from the original dataframe
            for column in dimension_size_cols:
                # Count comes from the already filtered dataframe. This is much faster than filtering df_global in a loop.
                col_present_count = filtered_df.filter(pl.col(column) > 1).height
                col_ratio_text = (
                    f"{column.replace('_', ' ').title()}: {col_present_count} of {col_total_files} files ({((col_present_count / col_total_files) * 100):.2f}%)."
                    if col_total_files > 0 else f"{column.replace('_', ' ').title()}: No files."
                )
                all_ratios_text_components.append(html.Span(col_ratio_text))
                all_ratios_text_components.append(html.Br())

            dim_size_info_children = [
                html.P(html.B("Data Availability by Dimension:")),
                html.P(all_ratios_text_components)
            ]

            return dim_size_info_children, xy_size_plot_children, individual_dim_plots