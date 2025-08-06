from pathlib import Path
from typing import List
import polars as pl
import plotly.graph_objects as go
from dash import html, dcc, Input, Output
import numpy as np
from pixel_patrol.report.widget_interface import PixelPatrolWidget
from pixel_patrol.report.widget_categories import WidgetCategories


class DatasetHistogramsWidget(PixelPatrolWidget):
    @property
    def tab(self) -> str:
        return WidgetCategories.DATASET_STATS.value

    @property
    def name(self) -> str:
        return "Pixel Value Histograms"

    def required_columns(self) -> List[str]:
        # All histogram keys, including per-dimension, will be available
        return ["histogram"]

    def layout(self) -> List:
        """
        Defines the static layout of the Pixel Value Histograms widget.
        Dropdowns are initialized empty; options are populated in the callback.
        """
        return [
            html.P(
                id="dataset-histograms-warning",
                className="text-warning",
                style={"marginBottom": "15px"},
            ),
            html.Div(
                [
                    html.Label("Select histogram dimension to plot:"),
                    dcc.Dropdown(
                        id="histogram-dimension-dropdown",
                        options=[],  # Populated in callback
                        value=None,
                        clearable=False,
                        style={
                            "width": "300px",
                            "marginTop": "10px",
                            "marginBottom": "20px",
                        },
                    ),
                ]
            ),
            html.Div(
                [
                    html.Label("Select folder names to compare:"),
                    dcc.Dropdown(
                        id="histogram-folder-dropdown",
                        options=[],  # Populated in callback
                        value=[],
                        multi=True,
                        style={
                            "width": "300px",
                            "marginTop": "10px",
                            "marginBottom": "20px",
                        },
                    ),
                ]
            ),
            dcc.Graph(id="histogram-plot", style={"height": "600px"}),
            html.Div(
                className="markdown-content",
                children=[
                    html.H4("Histogram Visualization"),
                    html.P(
                        [
                            "The mean histogram for each selected group (folder name) is shown. You can select which histogram dimension to visualize (e.g., global, per z-slice, per channel, etc.). ",
                            "Multiple groups can be overlayed for direct comparison. ",
                            "Histograms are normalized to sum to 1 for density comparison.",
                        ]
                    ),
                ],
            ),
        ]

    def register_callbacks(self, app, df_global: pl.DataFrame):
        # Callback to populate dropdown options and update the plot
        @app.callback(
            Output("histogram-dimension-dropdown", "options"),
            Output("histogram-dimension-dropdown", "value"),
            Output("histogram-folder-dropdown", "options"),
            Output("histogram-folder-dropdown", "value"),
            Input("color-map-store", "data"),
        )
        def populate_dropdowns(color_map):
            # Extract all histogram columns (including per-dimension)
            histogram_columns = [
                col for col in df_global.columns if col.startswith("histogram")
            ]
            dropdown_options = [
                {"label": col, "value": col} for col in histogram_columns
            ]
            default_histogram = histogram_columns[0] if histogram_columns else None

            # Folder selection
            folder_names = (
                df_global["imported_path_short"].unique().to_list()
                if "imported_path_short" in df_global.columns
                else []
            )
            folder_options = [
                {"label": str(Path(f).name), "value": f} for f in folder_names
            ]
            default_folders = (
                folder_names[:2] if len(folder_names) > 1 else folder_names
            )

            return dropdown_options, default_histogram, folder_options, default_folders

        # Callback to update the histogram plot
        @app.callback(
            Output("histogram-plot", "figure"),
            Output("dataset-histograms-warning", "children"),
            Input("color-map-store", "data"),
            Input("histogram-dimension-dropdown", "value"),
            Input("histogram-folder-dropdown", "value"),
        )
        def update_histogram_plot(color_map, histogram_key, selected_folders):
            if not histogram_key or not selected_folders:
                return (
                    go.Figure(),
                    "Please select a histogram dimension and at least one folder.",
                )
            if histogram_key not in df_global.columns:
                return go.Figure(), "No histogram data found in the selected images."
            chart = go.Figure()
            for folder in selected_folders:
                df_group = df_global.filter(pl.col("imported_path_short") == folder)
                if df_group.is_empty():
                    continue
                histograms = df_group[histogram_key].to_list()
                histograms = [np.array(h) for h in histograms if h is not None]
                if not histograms:
                    continue
                mean_hist = np.mean(histograms, axis=0)
                mean_hist = (
                    mean_hist / mean_hist.sum() if mean_hist.sum() > 0 else mean_hist
                )
                color = color_map.get(folder, None) if color_map else None
                # Plot individual histograms as semi-transparent lines
                for h, file_name in zip(histograms, df_group["name"].to_list()):
                    # normalize frequency to sum to 1 per histogram. This is not a density histogram, but a normalized frequency histogram.
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
                # Plot the mean histogram as a bold line
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
