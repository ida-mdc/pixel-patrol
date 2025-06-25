from typing import List, Dict

import dash_ag_grid as dag
import polars as pl
from dash import html, Input, Output

from pixel_patrol.report.widget_interface import PixelPatrolWidget
from pixel_patrol.report.widget_categories import WidgetCategories


class DataFrameWidget(PixelPatrolWidget):

    @property
    def tab(self) -> str:
        return WidgetCategories.SUMMARY.value

    @property
    def name(self) -> str:
        return "Dataframe Viewer"

    def required_columns(self) -> List[str]:
        return []


    def layout(self) -> List:
        intro = html.Div(id="table-intro", style={"marginBottom": "20px"})
        table = html.Div(id="table-table", style={"marginTop": "20px"})
        return [intro, table]


    def register_callbacks(self, app, df_global: pl.DataFrame):
        @app.callback(
            Output("table-intro", "children"),
            Output("table-table", "children"),
            Input("color-map-store", "data")
        )
        def update_table(color_map: Dict[str, str]):
            intro = html.P(f"This is the whole image collection table this report is based on.")
    
            # AgGrid table component
            grid = dag.AgGrid(
                rowData=df_global.to_dicts(),
                columnDefs=[{"field": col} for col in df_global.columns],
                # columnSize="autoSize",
                id="summary_grid"
            )
            table_div = html.Div([grid])
    
            return intro, table_div
