from typing import List, Dict, Set

import plotly.graph_objects as go
import polars as pl
from dash import dcc, Input, Output
from wordcloud import WordCloud

from pixel_patrol_base.report.widget_categories import WidgetCategories


class DiaryWordCloudWidget:
    NAME = "Diary Word Cloud"
    TAB = WidgetCategories.SUMMARY.value
    REQUIRES: Set[str] = {"free_text"}      # produced by the processor
    REQUIRES_PATTERNS = None

    FIG_ID = "diary-wordcloud-fig"

    def layout(self) -> List:
        return [dcc.Graph(id=self.FIG_ID)]

    def register(self, app, df_global: pl.DataFrame):
        @app.callback(
            Output(self.FIG_ID, "figure"),
            Input("color-map-store", "data"),
        )
        def render(_color_map: Dict[str, str]):
            texts = " ".join([t or "" for t in df_global["free_text"].to_list()])

            wc = WordCloud(width=900, height=500, background_color="white")
            img = wc.generate(texts).to_array()  # -> numpy array (H, W, 3)

            fig = go.Figure(go.Image(z=img))
            fig.update_xaxes(visible=False)
            fig.update_yaxes(visible=False)
            fig.update_layout(
                title="Word Cloud",
                margin=dict(l=20, r=20, t=40, b=20),
                height=520,
                showlegend=False,
            )
            return fig
