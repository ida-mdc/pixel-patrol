from typing import List, Dict, Set

import dash_ag_grid as dag
import polars as pl
from dash import html, Input, Output

from pixel_patrol_base.config import MAX_ROWS_DISPLAYED, MAX_COLS_DISPLAYED
from pixel_patrol_base.report.widget_categories import WidgetCategories
from pixel_patrol_base.report.base_widget import BaseReportWidget

class DataFrameWidget(BaseReportWidget):
    # ---- Declarative spec ----
    NAME: str = "Dataframe Viewer"
    TAB: str = WidgetCategories.SUMMARY.value
    REQUIRES: Set[str] = set()     # no required columns
    REQUIRES_PATTERNS = None

    # Component IDs
    INTRO_ID = "table-intro"
    TABLE_ID = "table-table"
    GRID_ID = "summary_grid"

    @property
    def help_text(self) -> str:
        return (
            f"Displays the first {MAX_ROWS_DISPLAYED} rows and {MAX_COLS_DISPLAYED} columns of the processed data table."
        )

    def get_content_layout(self) -> List:
        return [
            # Removed INTRO_ID div
            html.Div(id=self.TABLE_ID, style={"marginTop": "20px"}),
        ]

    def register(self, app, df_global: pl.DataFrame):
        @app.callback(
            Output(self.TABLE_ID, "children"),
            Input("color-map-store", "data"),  # not used; just a trigger
        )
        def update_table(_color_map: Dict[str, str]):
            # intro logic removed; static help_text used instead

            df_global_limited = df_global.limit(MAX_ROWS_DISPLAYED)
            cols_to_display = df_global_limited.columns[:MAX_COLS_DISPLAYED]
            df_global_limited = df_global_limited.select(cols_to_display)

            grid = dag.AgGrid(
                id=self.GRID_ID,
                rowData=df_global_limited.to_dicts(),
                columnDefs=[{"field": col} for col in cols_to_display],
                style={"maxHeight": "70vh"},
                # pagination=True, paginationPageSize=100
                # columnSize="sizeToFit",
                # dashGridOptions={"domLayout": "autoHeight"},
            )
            table_div = html.Div([grid])

            return table_div