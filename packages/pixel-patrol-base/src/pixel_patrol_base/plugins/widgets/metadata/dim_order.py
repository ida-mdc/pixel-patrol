from typing import List, Dict, Set

import polars as pl
from dash import html, dcc, Input, Output

from pixel_patrol_base.report.widget_categories import WidgetCategories
from pixel_patrol_base.report.base_widget import BaseReportWidget
from pixel_patrol_base.report.global_controls import apply_global_config, GLOBAL_CONFIG_STORE_ID
from pixel_patrol_base.report.factory import plot_bar

class DimOrderWidget(BaseReportWidget):
    NAME: str = "Dimension Order Distribution"
    TAB: str = WidgetCategories.METADATA.value
    REQUIRES: Set[str] = {"dim_order", "name"}
    REQUIRES_PATTERNS = None

    # Component IDs
    RATIO_ID = "dim-order-present-ratio"
    GRAPH_ID = "dim-order-bar-chart"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._df: pl.DataFrame | None = None

    @property
    def help_text(self) -> str:
        return (
            "Shows how often each **dimension ordering** (e.g., `TZYX`, `ZYX`, `CTZYX`) appears in the dataset.\n\n"
            "**Use this to detect**\n"
            "- inconsistent dimension layouts between/within groupings\n"
            "- files that may need reordering before analysis\n"
        )

    def get_content_layout(self) -> List:
        return [
            html.Div(id=self.RATIO_ID, style={"marginBottom": "15px"}),
            dcc.Graph(id=self.GRAPH_ID, style={"height": "500px"}),
        ]

    def register(self, app, df: pl.DataFrame):
        self._df = df

        app.callback(
            Output(self.GRAPH_ID, "figure"),
            Output(self.RATIO_ID, "children"),
            Input("color-map-store", "data"),
            Input(GLOBAL_CONFIG_STORE_ID, "data"),
        )(self._update_plot)

    def _update_plot(self, color_map: Dict[str, str], global_config: Dict):
        color_map = color_map or {}
        df = self._df

        # Apply global filters (rows) and get dynamic group column
        df_processed, group_col = apply_global_config(df, global_config)

        # Count only rows with a dim_order value
        df_filtered = (
            df_processed
            .with_columns(pl.lit(1).alias("value_count"))
            .filter(pl.col("dim_order").is_not_null())
        )

        # Ratio text
        present = df_filtered.height
        total = df_processed.height
        ratio_text = (
            f"{present} of {total} files ({(present / total) * 100:.2f}%) have 'Dimension Order' information."
            if total > 0 else "No files to display Dim Order information."
        )

        if present == 0:
            return {}, ratio_text

        # Aggregate counts per (dim_order, group)
        plot_data_agg = (
            df_filtered
            .group_by(["dim_order", group_col])
            .agg(
                pl.sum("value_count").alias("count"),
            )
            .sort(["dim_order", group_col])
        )

        # Use factory plot
        fig = plot_bar(
            df=plot_data_agg,
            x="dim_order",
            y="count",
            color=group_col,
            color_map=color_map,
            title="Dim Order Distribution",
            labels={"dim_order": "Dimension Order", "count": "Count", group_col: "Group"},
            barmode="stack"
        )

        return fig, ratio_text