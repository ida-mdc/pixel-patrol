from typing import List, Dict, Set

import dash_ag_grid as dag
import polars as pl
from dash import html, Input, Output

from pixel_patrol_base.config import MAX_ROWS_DISPLAYED, MAX_COLS_DISPLAYED
from pixel_patrol_base.report.widget_categories import WidgetCategories
from pixel_patrol_base.report.base_widget import BaseReportWidget
from pixel_patrol_base.report.global_controls import prepare_widget_data
from pixel_patrol_base.report.constants import GLOBAL_CONFIG_STORE_ID, FILTERED_INDICES_STORE_ID
from pixel_patrol_base.report.factory import show_no_data_message


class DataFrameWidget(BaseReportWidget):
    # ---- Declarative spec ----
    NAME: str = "Dataframe Viewer"
    TAB: str = WidgetCategories.SUMMARY.value
    REQUIRES: Set[str] = set()     # no required columns
    REQUIRES_PATTERNS = None

    CONTENT_ID = "dataframe-content"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._df: pl.DataFrame | None = None

    @property
    def help_text(self) -> str:
        return (
            f"Displays the first {MAX_ROWS_DISPLAYED} rows and {MAX_COLS_DISPLAYED} columns of the processed data table.\n\n"
            "Sorting is available by clicking on column headers."
        )

    def get_content_layout(self) -> List:
        return [html.Div(id=self.CONTENT_ID, style={"marginTop": "15px"})]

    def register(self, app, df: pl.DataFrame):
        self._df = df

        app.callback(
            Output(self.CONTENT_ID, "children"),
            Input("color-map-store", "data"),
            Input(FILTERED_INDICES_STORE_ID, "data"),
            Input(GLOBAL_CONFIG_STORE_ID, "data"),
        )(self._update_content)


    def _update_content(
        self,
        _color_map: Dict[str, str] | None,
        subset_indices: List[int] | None,
        global_config: Dict | None,
    ):

        df_filtered, _group_col, _resolved, _warning, _order = prepare_widget_data(
            self._df,
            subset_indices,
            global_config or {},
            metric_base=None,
        )

        if df_filtered.is_empty():
            return show_no_data_message()

        df_limited = df_filtered.limit(MAX_ROWS_DISPLAYED)
        cols_to_display = df_limited.columns[:MAX_COLS_DISPLAYED]
        df_limited = df_limited.select(cols_to_display)

        grid = dag.AgGrid(
            id="dataframe-grid",
            # Even though rechunk is expensive, it ensures that to_dicts works
            rowData=df_limited.rechunk().to_dicts(),
            columnDefs=[{"field": col} for col in cols_to_display],
            style={"maxHeight": "70vh"},
        )

        return html.Div(grid)