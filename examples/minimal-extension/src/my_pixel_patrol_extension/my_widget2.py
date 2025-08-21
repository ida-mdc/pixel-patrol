from typing import List, Dict, Set

import polars as pl
from dash import dcc, Input, Output, html

from pixel_patrol_base.report.widget_categories import WidgetCategories


class MoodTrendWidget:
    NAME = "Mood Trend"
    TAB = WidgetCategories.SUMMARY.value
    REQUIRES: Set[str] = {"moods", "entry_date", "positivity_factor", "imported_path_short"}
    REQUIRES_PATTERNS = None

    COUNTS_ID = "mood-trend-counts"
    TREND_ID = "mood-trend-lines"

    def layout(self) -> List:
        return [
            dcc.Graph(id=self.COUNTS_ID),   # stacked bars by folder (with color map)
            dcc.Graph(id=self.TREND_ID),    # overlaid years via projected month-day
        ]

    def register(self, app, df_global: pl.DataFrame):
        @app.callback(
            Output(self.COUNTS_ID, "figure"),
            Output(self.TREND_ID, "figure"),
            Input("color-map-store", "data"),
        )
        def render(color_map: Dict[str, str]):
            color_map = color_map or {}

            # --- Prep: parse date; explode moods for counts ---
            df = (
                df_global
                .with_columns(pl.col("entry_date").str.strptime(pl.Date, strict=False))
                .explode("moods")
                .drop_nulls(subset=["entry_date"])
            )

            # === Figure 1: Mood counts (stacked by folder, colored by folder) ===
            counts = (
                df
                .filter(pl.col("moods").is_not_null())
                .group_by(["moods", "imported_path_short"])
                .agg(pl.count().alias("count"))
                .sort(["moods", "imported_path_short"])
            )

            moods_order = counts["moods"].unique().to_list()
            folders = counts["imported_path_short"].unique().to_list()

            counts_traces = []
            for folder in folders:
                sub = counts.filter(pl.col("imported_path_short") == folder)
                mood_to_count = {m: 0 for m in moods_order}
                for m, c in zip(sub["moods"].to_list(), sub["count"].to_list()):
                    mood_to_count[m] = c
                counts_traces.append({
                    "type": "bar",
                    "x": moods_order,
                    "y": [mood_to_count[m] for m in moods_order],
                    "name": str(folder),
                    "marker": {"color": color_map.get(folder, "#888")},
                })

            counts_fig = {
                "data": counts_traces,
                "layout": {
                    "title": "Mood Occurrences (stacked by folder)",
                    "barmode": "stack",
                    "height": 420,
                    "margin": {"l": 30, "r": 30, "t": 60, "b": 40},
                    "showlegend": True,
                    "yaxis": {"title": "Count"},
                },
            }

            # === Figure 2: Mean positivity over time (years overlaid) ===
            # Project dates to fixed year 2000 so all years align on x.
            trend = (
                df_global
                .with_columns(pl.col("entry_date").str.strptime(pl.Date, strict=False))
                .drop_nulls(subset=["entry_date", "positivity_factor"])
                .with_columns([
                    pl.col("entry_date").dt.year().alias("year"),
                    pl.col("entry_date").dt.strftime("2000-%m-%d").alias("md_date_str"),
                ])
                .group_by(["md_date_str", "year", "imported_path_short"])
                .agg(pl.mean("positivity_factor").alias("mean_positivity"))
                .sort(["imported_path_short", "year", "md_date_str"])
            )

            trend_traces = []
            for folder in trend["imported_path_short"].unique().to_list():
                tf = trend.filter(pl.col("imported_path_short") == folder)
                for year in tf["year"].unique().to_list():
                    sub = tf.filter(pl.col("year") == year).sort("md_date_str")
                    trend_traces.append({
                        "type": "scatter",
                        "mode": "lines+markers",
                        "name": f"{folder}",
                        "x": sub["md_date_str"].to_list(),   # all in year 2000
                        "y": sub["mean_positivity"].to_list(),
                        "line": {"color": color_map.get(folder, "#333")},
                        "marker": {"color": color_map.get(folder, "#333")},
                        "hovertemplate": "%{x|%d/%m} · %{y:.3f}<extra>" + f"{folder}" + "</extra>",
                    })

            trend_fig = {
                "data": trend_traces,
                "layout": {
                    "title": "Mean Positivity Over Year",
                    "xaxis": {
                        "tickformat": "%d/%m",         # show day/month only
                        "range": ["2000-01-01", "2000-12-31"],  # full synthetic year
                    },
                    "yaxis": {"title": "Mean Positivity"},
                    "height": 420,
                    "margin": {"l": 30, "r": 30, "t": 60, "b": 40},
                    "showlegend": True,
                },
            }

            return counts_fig, trend_fig
