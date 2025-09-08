from pathlib import Path
from typing import List, Dict, Set

import numpy as np
import plotly.graph_objects as go
import polars as pl
from dash import html, dcc, Input, Output

from pixel_patrol_base.report.widget_categories import WidgetCategories


class DatasetHistogramsWidget:
    # ---- Declarative spec ----
    NAME: str = "Pixel Value Histograms"
    TAB: str = WidgetCategories.DATASET_STATS.value
    REQUIRES: Set[str] = {"imported_path_short", "name"}   # columns directly accessed
    REQUIRES_PATTERNS = [r"^histogram"]                    # dynamic histogram columns

    # Component IDs
    WARNING_ID = "dataset-histograms-warning"
    DIM_DROPDOWN_ID = "histogram-dimension-dropdown"
    FOLDER_DROPDOWN_ID = "histogram-folder-dropdown"
    PLOT_ID = "histogram-plot"

    def layout(self) -> List:
        """
        Static layout; dropdown options/values are filled by callbacks.
        """
        return [
            html.P(id=self.WARNING_ID, className="text-warning", style={"marginBottom": "15px"}),
            html.Div(
                [
                    html.Label("Select histogram dimension to plot:"),
                    dcc.Dropdown(
                        id=self.DIM_DROPDOWN_ID,
                        options=[],   # populated in callback
                        value=None,
                        clearable=False,
                        style={"width": "300px", "marginTop": "10px", "marginBottom": "20px"},
                    ),
                ]
            ),
            html.Div(
                [
                    html.Label("Select folder names to compare:"),
                    dcc.Dropdown(
                        id=self.FOLDER_DROPDOWN_ID,
                        options=[],   # populated in callback
                        value=[],
                        multi=True,
                        style={"width": "300px", "marginTop": "10px", "marginBottom": "20px"},
                    ),
                ]
            ),
            dcc.Graph(id=self.PLOT_ID, style={"height": "600px"}),
            html.Div(
                className="markdown-content",
                children=[
                    html.H4("Histogram Visualization"),
                    html.P(
                        [
                            "The mean histogram for each selected group (folder name) is shown. "
                            "You can select which histogram dimension to visualize (e.g., global, per z-slice, per channel, etc.). ",
                            "Multiple groups can be overlayed for direct comparison. ",
                            "Histograms are normalized to sum to 1 for density comparison.",
                        ]
                    ),
                ],
            ),
        ]

    def register(self, app, df_global: pl.DataFrame):
        # Populate dropdowns
        @app.callback(
            Output(self.DIM_DROPDOWN_ID, "options"),
            Output(self.DIM_DROPDOWN_ID, "value"),
            Output(self.FOLDER_DROPDOWN_ID, "options"),
            Output(self.FOLDER_DROPDOWN_ID, "value"),
            Input("color-map-store", "data"),
        )
        def populate_dropdowns(color_map: Dict[str, str]):  # color_map unused; kept for consistency
            # If required columns/patterns are missing, return empty controls

            # histogram columns (dynamic)
            histogram_columns = [c for c in df_global.columns if c.startswith("histogram")]
            dropdown_options = [{"label": c, "value": c} for c in histogram_columns]
            default_histogram = histogram_columns[0] if histogram_columns else None

            # folder names
            folder_names = df_global["imported_path_short"].unique().to_list()
            folder_options = [{"label": str(Path(f).name), "value": f} for f in folder_names]
            default_folders = folder_names[:2] if len(folder_names) > 1 else folder_names

            return dropdown_options, default_histogram, folder_options, default_folders

        # Update plot + warning
        @app.callback(
            Output(self.PLOT_ID, "figure"),
            Output(self.WARNING_ID, "children"),
            Input("color-map-store", "data"),
            Input(self.DIM_DROPDOWN_ID, "value"),
            Input(self.FOLDER_DROPDOWN_ID, "value"),
        )
        def update_histogram_plot(color_map: Dict[str, str], histogram_key: str, selected_folders: List[str]):
            color_map = color_map or {}

            if not histogram_key or not selected_folders:
                return go.Figure(), "Please select a histogram dimension and at least one folder."

            if histogram_key not in df_global.columns:
                return go.Figure(), "No histogram data found in the selected images."

            chart = go.Figure()

            for folder in selected_folders:
                df_group = df_global.filter(pl.col("imported_path_short") == folder)
                if df_group.is_empty():
                    continue

                # Collect per-file histograms
                histograms = [h for h in df_group[histogram_key].to_list() if h is not None]
                if not histograms:
                    continue

                histograms = [np.array(h) for h in histograms]
                mean_hist = np.mean(histograms, axis=0)
                mean_hist = mean_hist / mean_hist.sum() if mean_hist.sum() > 0 else mean_hist
                color = color_map.get(folder)

                # Individual (semi-transparent) lines
                for h, file_name in zip(histograms, df_group["name"].to_list()):
                    h_norm = h / h.sum() if h.sum() > 0 else h
                    chart.add_trace(
                        go.Scatter(
                            x=list(range(len(h_norm))),
                            y=h_norm,
                            mode="lines",
                            name=Path(folder).name,
                            line=dict(width=1, color=color),
                            opacity=0.2,
                            showlegend=False,
                            legendgroup=Path(folder).name,
                            hovertemplate=(
                                f"File: {file_name}<br>Pixel: %{{x}}<br>Freq: %{{y:.3f}}<extra></extra>"
                            ),
                        )
                    )

                # Mean (bold filled) line
                chart.add_trace(
                    go.Scatter(
                        x=list(range(len(mean_hist))),
                        y=mean_hist,
                        mode="lines",
                        name=Path(folder).name,
                        line=dict(width=2, color=color),
                        fill="tozeroy",
                        opacity=0.7,
                        legendgroup=Path(folder).name,
                        hovertemplate=(
                            f"Folder: {Path(folder).name}<br>Pixel: %{{x}}<br>Mean Freq: %{{y:.3f}}<extra></extra>"
                        ),
                    )
                )

            chart.update_layout(
                title="Mean Pixel Value Histogram (per group)",
                xaxis_title="Pixel Value normalized to 0-255",
                yaxis_title="Normalized Frequency",
                legend_title="Folder name",
            )
            return chart, ""
