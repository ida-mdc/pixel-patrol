from typing import List, Dict, Set
import polars as pl
from dash import html, dcc, Input, Output

from pixel_patrol_base.report.widget_categories import WidgetCategories
from pixel_patrol_base.report.base_widget import BaseReportWidget
from pixel_patrol_base.report.global_controls import GLOBAL_CONFIG_STORE_ID

class DataTypeWidget(BaseReportWidget):
    NAME: str = "Data Type Distribution"
    TAB: str = WidgetCategories.METADATA.value
    REQUIRES: Set[str] = {"dtype", "name"}
    REQUIRES_PATTERNS = None

    # Component IDs
    RATIO_ID = "dtype-present-ratio"
    GRAPH_ID = "data-type-bar-chart"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._df: pl.DataFrame | None = None

    @property
    def help_text(self) -> str:
        return (
            "Shows the distribution of **pixel data types** (e.g., `uint8`, `uint16`, `float32`) across groupings.\n\n"
            "**Use this to check**\n"
            "- whether different images/groupings use different numeric formats\n"
            "- potential range differences (e.g., 0–255 vs. arbitrary floats)\n"
        )


    def get_content_layout(self) -> List:
        """Defines the layout of the Data Type Distribution widget."""
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
        from pixel_patrol_base.report.global_controls import apply_global_config
        from pixel_patrol_base.report.factory import plot_bar

        color_map = color_map or {}
        df = self._df

        # Apply global filters (rows) and get dynamic group column
        df_processed, group_col = apply_global_config(df, global_config)

        # Prepare data: count only rows with dtype present
        df_filtered = (
            df_processed
            .with_columns(pl.lit(1).alias("value_count"))
            .filter(pl.col("dtype").is_not_null())
        )

        # Ratio text
        present = df_filtered.height
        total = df_processed.height
        ratio_text = (
            f"{present} of {total} files ({(present / total) * 100:.2f}%) have 'Data Type' information."
            if total > 0 else "No files to display data type information."
        )

        if present == 0:
            return {}, ratio_text

        # Aggregate counts per (dtype, group_col)
        plot_data_agg = (
            df_filtered
            .group_by(["dtype", group_col])
            .agg(
                pl.sum("value_count").alias("count"),
            )
            .sort(["dtype", group_col])
        )

        # Use factory plot
        fig = plot_bar(
            df=plot_data_agg,
            x="dtype",
            y="count",
            color=group_col,
            color_map=color_map,
            title="Data Type Distribution",
            labels={"dtype": "Data Type", "count": "Count", group_col: "Group"},
            barmode="stack"
        )

        return fig, ratio_text