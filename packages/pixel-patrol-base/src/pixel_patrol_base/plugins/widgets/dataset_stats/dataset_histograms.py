from pathlib import Path
from typing import List
import polars as pl
import plotly.graph_objects as go
from dash import html, dcc, Input, Output, ALL
import numpy as np
from pixel_patrol_base.report.widget_categories import WidgetCategories
from pixel_patrol_base.plugins.processors.histogram_processor import safe_hist_range
from pixel_patrol_base.report.base_widget import BaseReportWidget
from pixel_patrol_base.report.factory import create_dimension_selectors


class DatasetHistogramWidget(BaseReportWidget):
    NAME: str = "Pixel Value Histograms"
    TAB: str = WidgetCategories.DATASET_STATS.value
    REQUIRES = set()
    REQUIRES_PATTERNS: List[str] = [r"^histogram"]

    @property
    def help_text(self) -> str: # Markdown
        return (
            "The histograms are computed **per image** and grouped based on your groupings.  \n"
            "They are normalized to sum to **1**, and the **mean histogram per group** is shown as a bold line.\n\n"
            "**Modes**\n"
            "- **0–255 bins (shape comparison)**  \n"
            "  Uses 256 fixed bins (0–255) regardless of the actual pixel range. "
            "Best for comparing overall *shape* across images with different value ranges or data types.\n"
            "- **Native pixel-range bins**  \n"
            "  Bins are defined using the actual min/max pixel values across the selected images. "
            "Best for seeing where intensities lie in *absolute* terms, but shapes may differ if ranges vary.\n"
        )

    def get_content_layout(self) -> List:
        return [
            html.P(id="dataset-histograms-warning", className="text-warning", style={"marginBottom": "15px"}),
            html.Div(
                [
                    html.Label("Select histogram dimensions to plot:"),
                    html.Div(id="histogram-filters-container"),
                    dcc.Store(id="histogram-dims-store"),
                ]
            ),
            html.Div(
                [
                    html.Label("Histogram plot mode:"),
                    dcc.RadioItems(
                        id="histogram-remap-mode",
                        options=[
                            {"label": "Fixed 0–255 bins", "value": "shape"},
                            {"label": "Bin on the native pixel range", "value": "native"},
                        ],
                        value="shape",
                        labelStyle={"display": "inline-block", "marginRight": "12px"},
                    ),
                ],
                style={"marginBottom": "8px"},
            ),
            html.Div(
                [
                    html.Label("Select folder names to compare:"),
                    dcc.Dropdown(
                        id="histogram-folder-dropdown",
                        options=[],
                        value=[],
                        multi=True,
                        style={"width": "300px", "marginTop": "10px", "marginBottom": "20px"},
                    ),
                ]
            ),
            dcc.Graph(id="histogram-plot", style={"height": "600px"}),
            html.Div([
                html.Label("Show bars for specific file (optional):"),
                dcc.Dropdown(
                    id="histogram-file-dropdown",
                    options=[],
                    value=None,
                    clearable=True,
                    style={"width": "300px", "marginTop": "10px", "marginBottom": "20px"},
                ),
            ]),
        ]

    # --- Static Helpers (Restored from old file) ---
    @staticmethod
    def _edges_from_minmax(n_bins, minv, maxv):
        if float(minv).is_integer() and float(maxv).is_integer():
            sample = np.array([int(minv), int(maxv)], dtype=np.int64)
        else:
            sample = np.array([minv, maxv], dtype=float)
        smin, _smax, max_adj = safe_hist_range(sample)
        smin_f, max_adj_f = float(smin), float(max_adj)
        width = (max_adj_f - smin_f) / float(n_bins)
        lefts = smin_f + np.arange(n_bins, dtype=float) * width
        return np.concatenate([lefts, [smin_f + n_bins * width]])

    @staticmethod
    def _folder_minmax_using_polars(df_group, min_key, max_key):
        if min_key in df_group.columns and max_key in df_group.columns:
            agg = df_group.select([pl.col(min_key).min().alias("_min"), pl.col(max_key).max().alias("_max")]).to_dict(
                as_series=False)
            return (float(agg["_min"][0]) if agg["_min"][0] is not None else None,
                    float(agg["_max"][0]) if agg["_max"][0] is not None else None)
        return None, None

    @staticmethod
    def _compute_edges(counts, minv, maxv):
        n = counts.size
        if minv is None or maxv is None:
            edges = np.arange(n + 1).astype(float)
            return edges, edges[:-1], np.diff(edges)
        if float(minv).is_integer() and float(maxv).is_integer():
            sample = np.array([int(minv), int(maxv)], dtype=np.int64)
        else:
            sample = np.array([minv, maxv], dtype=float)
        smin, _smax, max_adj = safe_hist_range(sample)
        smin_f, max_adj_f = float(smin), float(max_adj)
        width = (max_adj_f - smin_f) / float(n)
        lefts = smin_f + np.arange(n, dtype=float) * width
        return np.concatenate([lefts, [smin_f + n * width]]), lefts, np.full(n, width, dtype=float)

    @staticmethod
    def _rebin_via_cdf(counts, src_edges, tgt_edges):
        counts = np.asarray(counts, float)
        se = np.asarray(src_edges, float)
        te = np.asarray(tgt_edges, float)
        if counts.size == 0 or counts.sum() <= 0: return np.zeros(te.size - 1, float)
        cdf_src = np.concatenate([[0.0], np.cumsum(counts) / counts.sum()])
        cdf_t = np.interp(te, se, cdf_src, left=0.0, right=1.0)
        return np.diff(cdf_t)

    def register(self, app, df_global: pl.DataFrame):
        @app.callback(
            Output("histogram-filters-container", "children"),
            Output("histogram-dims-store", "data"),
            Output("histogram-folder-dropdown", "options"),
            Output("histogram-folder-dropdown", "value"),
            Input("color-map-store", "data"),
        )
        def populate_dropdowns(color_map):
            from pixel_patrol_base.report.data_utils import extract_dimension_tokens

            children, dims_order = create_dimension_selectors(
                tokens=extract_dimension_tokens(df_global.columns, "histogram_counts"),
                id_type="histogram-dim-filter"
            )

            folder_names = (
                df_global["imported_path_short"].unique().to_list()
                if "imported_path_short" in df_global.columns
                else []
            )
            folder_options = [{"label": str(Path(f).name), "value": f} for f in folder_names]
            default_folders = folder_names[:2] if len(folder_names) > 1 else folder_names

            return children, dims_order, folder_options, default_folders

        @app.callback(
            Output("histogram-file-dropdown", "options"),
            Output("histogram-file-dropdown", "value"),
            Input("histogram-folder-dropdown", "value"),
        )
        def update_file_options(selected_folders):
            if "name" not in df_global.columns: return [], None
            if not selected_folders:
                names = df_global["name"].unique().to_list()
            else:
                names = df_global.filter(pl.col("imported_path_short").is_in(selected_folders))[
                    "name"].unique().to_list()
            return [{"label": n, "value": n} for n in names], None

        @app.callback(
            Output("histogram-plot", "figure"),
            Output("dataset-histograms-warning", "children"),
            Input("color-map-store", "data"),
            Input("histogram-remap-mode", "value"),
            Input({"type": "histogram-dim-filter", "dim": ALL}, "value"),
            Input("histogram-folder-dropdown", "value"),
            Input("histogram-file-dropdown", "value"),
            Input("histogram-dims-store", "data"),
        )
        def update_histogram_plot(color_map, remap_mode, dim_values_list, selected_folders, selected_file, dims_order):
            if not selected_folders: return go.Figure(), "Select a folder."

            selections = {}
            if dims_order and dim_values_list:
                for dim_name, val in zip(dims_order, dim_values_list):
                    selections[dim_name] = val

            from pixel_patrol_base.report.data_utils import find_best_matching_column
            histogram_columns = [col for col in df_global.columns if "histogram" in col]
            base = "histogram_counts"
            histogram_key = find_best_matching_column(histogram_columns, base, selections) or base

            # Identify suffix (e.g., '_t0_z0') from the chosen key
            suffix = histogram_key[len(base):]

            # Construct related column names based on that suffix
            min_key = f"histogram_min{suffix}"
            max_key = f"histogram_max{suffix}"
            bin_key = f"histogram_bin_size{suffix}"

            if histogram_key not in df_global.columns:
                return go.Figure(), f"Column {histogram_key} not found."

            chart = go.Figure()

            # Optional: Show individual file
            if selected_file:
                row = df_global.filter(pl.col("name") == selected_file)
                if row.height > 0:
                    counts = row.get_column(histogram_key).to_list()[0]
                    if counts:
                        counts = np.array(counts, dtype=float)
                        if counts.sum() > 0:
                            counts /= counts.sum()

                        minv = row.get_column(min_key).to_list()[0] if min_key in row.columns else 0
                        maxv = row.get_column(max_key).to_list()[0] if max_key in row.columns else 255

                        edges, centers, width = self._compute_edges(counts, minv, maxv)

                        chart.add_trace(go.Bar(
                            x=centers,
                            y=counts,
                            width=width,
                            name=f"File: {selected_file}",
                            marker_color="black", opacity=0.3
                        ))

            # Main Logic: Plot Group Means
            for folder in selected_folders:
                df_group = df_global.filter(pl.col("imported_path_short") == folder)
                if df_group.height == 0: continue

                color = color_map.get(folder, "#333333")

                # Mode 1: Remap to Fixed 0-255 (Shape comparison)
                if remap_mode == "shape":
                    accumulated = np.zeros(256, dtype=float)
                    count_files = 0

                    fixed_edges = np.linspace(0, 255, 257)  # 256 bins

                    for row in df_group.iter_rows(named=True):
                        c_list = row.get(histogram_key)
                        if c_list is None or len(c_list) == 0: continue
                        c_arr = np.array(c_list, dtype=float)
                        if c_arr.sum() == 0: continue

                        minv = row.get(min_key)
                        maxv = row.get(max_key)

                        src_edges, _, _ = self._compute_edges(c_arr, minv, maxv)

                        # Rebin to fixed 256 bins for averaging
                        rebinned = self._rebin_via_cdf(c_arr, src_edges, fixed_edges)
                        accumulated += rebinned
                        count_files += 1

                    if count_files > 0:
                        avg_hist = accumulated / count_files
                        if avg_hist.sum() > 0: avg_hist /= avg_hist.sum()

                        centers = 0.5 * (fixed_edges[:-1] + fixed_edges[1:])
                        chart.add_trace(go.Scatter(
                            x=centers, y=avg_hist, mode='lines',
                            name=folder, line=dict(color=color, width=2),
                            fill='tozeroy', opacity=0.6
                        ))

                # Mode 2: Native Range (using group min/max)
                else:
                    gmin, gmax = self._folder_minmax_using_polars(df_group, min_key, max_key)
                    if gmin is None or gmax is None: continue

                    # Heuristic for bin count: max of lengths of individual histograms
                    # or fixed 256. Let's use 256 for smoothness.
                    n_bins = 256
                    group_edges = self._edges_from_minmax(n_bins, gmin, gmax)

                    accumulated = np.zeros(n_bins, dtype=float)
                    count_files = 0

                    for row in df_group.iter_rows(named=True):
                        c_list = row.get(histogram_key)
                        if not c_list: continue
                        c_arr = np.array(c_list, dtype=float)
                        if c_arr.sum() == 0: continue

                        minv = row.get(min_key)
                        maxv = row.get(max_key)
                        src_edges, _, _ = self._compute_edges(c_arr, minv, maxv)

                        rebinned = self._rebin_via_cdf(c_arr, src_edges, group_edges)
                        accumulated += rebinned
                        count_files += 1

                    if count_files > 0:
                        avg_hist = accumulated / count_files
                        if avg_hist.sum() > 0: avg_hist /= avg_hist.sum()

                        centers = 0.5 * (group_edges[:-1] + group_edges[1:])
                        chart.add_trace(go.Scatter(
                            x=centers, y=avg_hist, mode='lines',
                            name=folder, line=dict(color=color, width=2),
                            fill='tozeroy', opacity=0.6
                        ))

            chart.update_layout(
                title="Mean Pixel Value Histogram (per group)",
                xaxis_title="Pixel intensity (0-255)" if remap_mode == "shape" else "Native Pixel Intensity",
                yaxis_title="Normalized Frequency",
                hovermode="x unified"
            )
            return chart, ""