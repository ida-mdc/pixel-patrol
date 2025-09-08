from typing import List, Dict, Set

import dash_ag_grid as dag
import polars as pl
from dash import html, Input, Output

from pixel_patrol_base.report.widget_categories import WidgetCategories


class DataFrameWidget:
    # ---- Declarative spec ----
    NAME: str = "Dataframe Viewer"
    TAB: str = WidgetCategories.SUMMARY.value
    REQUIRES: Set[str] = set()     # no required columns
    REQUIRES_PATTERNS = None

    # Component IDs
    INTRO_ID = "table-intro"
    TABLE_ID = "table-table"
    GRID_ID = "summary_grid"

    def layout(self) -> List:
        return [
            html.Div(id=self.INTRO_ID, style={"marginBottom": "20px"}),
            html.Div(id=self.TABLE_ID, style={"marginTop": "20px"}),
        ]

    def register(self, app, df_global: pl.DataFrame):
        @app.callback(
            Output(self.INTRO_ID, "children"),
            Output(self.TABLE_ID, "children"),
            Input("color-map-store", "data"),  # not used; just a trigger
        )
        def update_table(_color_map: Dict[str, str]):
            intro = html.P("This is the whole image collection table this report is based on.")

            max_cols = 500
            cols_to_display = df_global.columns[:max_cols]

            grid = dag.AgGrid(
                id=self.GRID_ID,
                rowData=df_global.to_dicts(),
                columnDefs=[{"field": col} for col in cols_to_display],
                # columnSize="sizeToFit",
                # dashGridOptions={"domLayout": "autoHeight"},
            )
            table_div = html.Div([grid])

            return intro, table_div
