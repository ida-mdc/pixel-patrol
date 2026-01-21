from typing import List, Dict, Set, Optional

import polars as pl
from dash import html, dcc, Input, Output

from pixel_patrol_base.report.base_widget import BaseReportWidget
from pixel_patrol_base.report.global_controls import prepare_widget_data
from pixel_patrol_base.report.constants import GLOBAL_CONFIG_STORE_ID, FILTERED_INDICES_STORE_ID
from pixel_patrol_base.report.factory import plot_bar, show_no_data_message
from pixel_patrol_base.report.data_utils import get_all_grouping_cols, select_needed_columns


class ColumnCountWithGroupingBarWidget(BaseReportWidget):
    """
    Abstract base for widgets that show counts of a categorical column on the X axis,
    optionally grouped by the global grouping column (color).
    """

    # Subclasses MUST set these:
    NAME: str = "Categorical Column Count by Group"
    TAB: str = ""
    REQUIRES: Set[str] = set()
    REQUIRES_PATTERNS = None

    # Categorical column in the dataframe used for the X axis
    CATEGORY_COLUMN: str = ""          # e.g. "dtype", "dim_order"
    CATEGORY_LABEL: str = ""           # human label for axis & messages, e.g. "Data Type"
    CONTENT_ID: str = ""               # unique Dash component id

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._df: Optional[pl.DataFrame] = None

    def get_content_layout(self) -> List:
        """Simple layout: one div that will contain ratio text + bar chart."""
        return [html.Div(id=self.CONTENT_ID)]

    def register(self, app, df: pl.DataFrame) -> None:
        self._df = df

        app.callback(
            Output(self.CONTENT_ID, "children"),
            Input("color-map-store", "data"),
            Input(FILTERED_INDICES_STORE_ID, "data"),
            Input(GLOBAL_CONFIG_STORE_ID, "data"),
        )(self._update_plot)

    def _update_plot(
        self,
        color_map: Dict[str, str] | None,
        subset_indices: List[int] | None,
        global_config: Dict | None,
    ):
        """Generic implementation: count rows by CATEGORY_COLUMN (+ optional group)."""

        df_filtered, group_col, _resolved, warning_msg, group_order = prepare_widget_data(
            self._df,
            subset_indices,
            global_config,
            metric_base=None,
        )
        n_total_rows = df_filtered.height

        # 1. Select only the columns strictly needed for grouping and counting.
        cols_needed = [self.CATEGORY_COLUMN]
        extra = [group_col] if group_col else []
        df_filtered = select_needed_columns(df_filtered, cols_needed, extra_cols=extra)

        # 2. Filter nulls from the category column
        df_filtered = df_filtered.filter(pl.col(self.CATEGORY_COLUMN).is_not_null())

        if df_filtered.is_empty():
            return show_no_data_message()

        n_filtered_rows = df_filtered.height
        ratio_text = (
            f"{n_filtered_rows} of {n_total_rows} files "
            f"({(n_filtered_rows / max(n_total_rows, 1)) * 100:.2f}%) "
            f"have '{self.CATEGORY_LABEL}' information."
        )

        all_grouping_cols = get_all_grouping_cols(
            [self.CATEGORY_COLUMN],
            group_col,
        )

        plot_data_agg = (
            df_filtered
            .group_by(all_grouping_cols)
            .agg(pl.len().alias("count"))
            .sort(all_grouping_cols)
        )

        labels = {
            self.CATEGORY_COLUMN: self.CATEGORY_LABEL,
            "count": "Count",
        }

        fig = plot_bar(
            df=plot_data_agg,
            x=self.CATEGORY_COLUMN,
            y="count",
            color=group_col,
            color_map=color_map,
            order_x=group_order,
            title=self.NAME,
            labels=labels,
            barmode="stack",
            show_legend=True,
        )

        return [
            html.Div(ratio_text, style={"marginBottom": "15px"}),
            dcc.Graph(figure=fig, style={"height": "500px"}),
        ]
