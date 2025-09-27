from typing import List, Dict, Set

import plotly.express as px
import polars as pl
from dash import html, dcc, Input, Output

from pixel_patrol_base.report.widget_categories import WidgetCategories


class DataTypeWidget:
    # ---- Declarative spec ----
    NAME: str = "Data Type Distribution"
    TAB: str = WidgetCategories.METADATA.value
    REQUIRES: Set[str] = {"dtype", "imported_path_short", "name"}  # all used below
    REQUIRES_PATTERNS = None

    # Component IDs
    RATIO_ID = "dtype-present-ratio"
    GRAPH_ID = "data-type-bar-chart"

    def layout(self) -> List:
        """Defines the layout of the Data Type Distribution widget."""
        return [
            html.Div(id=self.RATIO_ID, style={"marginBottom": "15px"}),
            dcc.Graph(id=self.GRAPH_ID, style={"height": "500px"}),
        ]

    def register(self, app, df_global: pl.DataFrame):
        """Registers callbacks for the Data Type Distribution widget."""
        @app.callback(
            Output(self.GRAPH_ID, "figure"),
            Output(self.RATIO_ID, "children"),
            Input("color-map-store", "data"),
        )
        def update_data_type_chart(color_map: Dict[str, str]):
            color_map = color_map or {}

            # Prepare data: count only rows with dtype present
            processed_df = (
                df_global
                .with_columns(pl.lit(1).alias("value_count"))
                .filter(pl.col("dtype").is_not_null())
            )

            # Ratio text
            dtype_present_count = processed_df.height
            total_files = df_global.height
            if total_files > 0:
                dtype_ratio_text = (
                    f"{dtype_present_count} of {total_files} files "
                    f"({(dtype_present_count / total_files) * 100:.2f}%) have 'Data Type' information."
                )
            else:
                dtype_ratio_text = "No files to display data type information."

            # Aggregate counts per (dtype, folder) and collect file names for hover
            plot_data_agg = (
                processed_df
                .group_by(["dtype", "imported_path_short"])
                .agg(
                    pl.sum("value_count").alias("count"),
                    pl.col("name").unique().alias("names_in_group"),
                )
                .sort(["dtype", "imported_path_short"])
            )

            # Plot
            fig = px.bar(
                plot_data_agg,
                x="dtype",
                y="count",
                color="imported_path_short",
                barmode="stack",
                color_discrete_map=color_map,
                title="Data Type Distribution",
                labels={
                    "dtype": "Data Type",
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
            n = plot_data_agg["dtype"].n_unique()
            if n == 1:
                fig.update_layout(bargap=0.7)
            if n == 2:
                fig.update_layout(bargap=0.4)

            return fig, dtype_ratio_text
