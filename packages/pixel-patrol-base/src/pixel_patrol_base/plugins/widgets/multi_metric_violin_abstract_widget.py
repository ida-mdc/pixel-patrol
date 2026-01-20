from typing import List, Dict, Set

import polars as pl
from dash import Input, Output, html

from pixel_patrol_base.report.base_widget import BaseReportWidget
from pixel_patrol_base.report.global_controls import (
    GLOBAL_CONFIG_STORE_ID,
    FILTERED_INDICES_STORE_ID,
    prepare_widget_data,
)
from pixel_patrol_base.report.factory import build_violin_grid, show_no_data_message
from pixel_patrol_base.report.data_utils import get_dim_aware_column, select_needed_columns


class MultiMetricViolinGridWidget(BaseReportWidget):
    """
    Abstract base for widgets that show a grid of violin plots for multiple numeric metrics,
    grouped by the global grouping column (color).
    """

    # Subclasses MUST set these:
    NAME: str = "Multi Metric Violin Grid"
    TAB: str = ""
    REQUIRES: Set[str] = set()
    REQUIRES_PATTERNS = None

    # Metric base names (dimension-aware resolution happens automatically)
    BASE_METRIC_NAMES: List[str] = []
    CONTENT_ID: str = ""  # unique Dash component id

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._df: pl.DataFrame | None = None

    def get_content_layout(self):
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
        df_filtered, group_col, _resolved, _warning_msg, group_order = prepare_widget_data(
            self._df,
            subset_indices,
            global_config,
            metric_base=None,
        )

        if df_filtered.is_empty() or not group_col:
            return show_no_data_message()

        global_config = global_config or {}
        dims_selection = global_config.get("dimensions", {})

        resolved_metric_cols: List[str] = []
        for base in self.BASE_METRIC_NAMES:
            col = get_dim_aware_column(df_filtered.columns, base, dims_selection)
            if col is not None:
                resolved_metric_cols.append(col)

        if not resolved_metric_cols:
            return show_no_data_message()

        cols_needed = resolved_metric_cols + ["name"]
        extra = [group_col] if group_col else []
        df_plot = select_needed_columns(df_filtered, cols_needed, extra_cols=extra)

        return build_violin_grid(
            df_plot,
            color_map or {},
            resolved_metric_cols,
            group_col=group_col,
            order_x=group_order,
        )
