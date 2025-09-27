from typing import List, Dict, Set

import plotly.express as px
import polars as pl
from dash import html, dcc, Input, Output

from pixel_patrol_base.report.widget_categories import WidgetCategories


class DimOrderWidget:
    # ---- Declarative spec ----
    NAME: str = "Dim Order Distribution"
    TAB: str = WidgetCategories.METADATA.value
    # columns used below: dim_order for grouping; imported_path_short/name for hover grouping
    REQUIRES: Set[str] = {"dim_order", "imported_path_short", "name"}
    REQUIRES_PATTERNS = None

    # Component IDs
    RATIO_ID = "dim-order-present-ratio"
    GRAPH_ID = "dim-order-bar-chart"

    def layout(self) -> List:
        """Defines the layout of the Dim Order Distribution widget."""
        return [
            html.Div(id=self.RATIO_ID, style={"marginBottom": "15px"}),
            dcc.Graph(id=self.GRAPH_ID, style={"height": "500px"}),
        ]

    def register(self, app, df_global: pl.DataFrame):
        """Registers callbacks for the Dim Order Distribution widget."""
        @app.callback(
            Output(self.GRAPH_ID, "figure"),
            Output(self.RATIO_ID, "children"),
            Input("color-map-store", "data"),  # color mapping
        )
        def update_dim_order_chart(color_map: Dict[str, str]):
            color_map = color_map or {}

            # Count only rows with a dim_order value
            processed_df = (
                df_global
                .with_columns(pl.lit(1).alias("value_count"))
                .filter(pl.col("dim_order").is_not_null())
            )

            # Ratio text
            present = processed_df.height
            total = df_global.height
            ratio_text = (
                f"{present} of {total} files ({(present / total) * 100:.2f}%) have 'Dim Order' information."
                if total > 0 else "No files to display Dim Order information."
            )

            # Aggregate counts per (dim_order, folder) and collect names for hover
            plot_data_agg = (
                processed_df
                .group_by(["dim_order", "imported_path_short"])
                .agg(
                    pl.sum("value_count").alias("count"),
                    pl.col("name").unique().alias("names_in_group"),
                )
                .sort(["dim_order", "imported_path_short"])
            )

            # Plot
            fig = px.bar(
                plot_data_agg,
                x="dim_order",
                y="count",
                color="imported_path_short",
                barmode="stack",
                color_discrete_map=color_map,
                title="Dim Order Distribution",
                labels={
                    "dim_order": "Dimension Order",
                    "count": "Number of Files",
                    "imported_path_short": "Folder",
                },
                hover_data=["imported_path_short", "count", "names_in_group"],
            )

            fig.update_traces(marker_line_color="white", marker_line_width=0.5, opacity=1)
            fig.update_layout(
                height=500,
                margin=dict(l=50, r=50, t=80, b=100),
                hovermode="closest",
                bargap=0.1,
                bargroupgap=0.05,
                legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5),
            )
            n = plot_data_agg["dim_order"].n_unique()
            if n==1:
                fig.update_layout(bargap=0.7)
            if n==2:
                fig.update_layout(bargap=0.4)

            return fig, ratio_text
