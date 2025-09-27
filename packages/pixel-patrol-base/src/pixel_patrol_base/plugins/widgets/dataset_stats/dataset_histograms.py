import json
import re
from collections import defaultdict
from typing import List, Dict, Any, Tuple

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
from dash import html, dcc, Input, Output, ALL, ctx
from polars.selectors import alpha

from pixel_patrol_base.report.utils import _parse_dynamic_col
from pixel_patrol_base.report.widget_categories import WidgetCategories


class DatasetHistogramWidget:
    """
    Overlayed histograms of all files with dimension slicing controls (except X/Y).
    Expects columns like 'histogram_z4' or 'histogram_ch1_t3' parsed by _parse_dynamic_col
    into metric='histogram' and dims={'z':4, 'ch':1, 't':3, ...}.

    Uses per-file color from 'color-map-store' keyed by df_global['imported_path_short'] if present.
    """

    NAME: str = "Overlayed Histograms"
    TAB: str = WidgetCategories.DATASET_STATS.value
    # needs at least file identity + optional color key; histogram_* columns are discovered dynamically
    REQUIRES = {"path"}  # 'imported_path_short' is optional but recommended for colors
    REQUIRES_PATTERNS = None

    def __init__(self, widget_id: str = "overlay-hist"):
        self.widget_id = widget_id
        self.graph_id = f"{widget_id}-graph"
        self.filters_container_id = f"{widget_id}-filters"

    # ---- helpers ----------------------------------------------------------------

    @staticmethod
    def _is_histogram_col(col: str) -> Tuple[bool, Dict[str, int]]:
        parsed = _parse_dynamic_col(col, supported_metrics=["histogram"])
        if not parsed:
            return False, {}
        return parsed

    def _downsample_hist(self, x: np.ndarray, y: np.ndarray, max_points: int = 400) -> tuple[np.ndarray, np.ndarray]:
        """Uniform stride decimation to cap points per trace."""
        n = x.size
        if n <= max_points:
            return x, y
        step = int(np.ceil(n / max_points))
        return x[::step], y[::step]

    def _rebin_to_max_bins(self, counts: np.ndarray, edges: np.ndarray, max_bins: int = 256) -> tuple[
        np.ndarray, np.ndarray]:
        """Rebin by uniform grouping of adjacent bins to a target max bin count."""
        nb = counts.size
        if nb <= max_bins:
            # return midpoints + normalized counts
            mids = 0.5 * (edges[:-1] + edges[1:])
            return mids, counts
        group = int(np.ceil(nb / max_bins))
        # pad to multiple of group
        pad = (-nb) % group
        if pad:
            counts = np.pad(counts, (0, pad), constant_values=0)
            edges = np.pad(edges, (0, pad), mode="edge")
        counts = counts.reshape(-1, group).sum(axis=1)
        # new edges by taking every 'group' edge
        new_edges = edges[::group]
        if new_edges.size == counts.size:  # ensure one more edge than counts
            new_edges = np.append(new_edges, edges[-1])
        mids = 0.5 * (new_edges[:-1] + new_edges[1:])
        return mids, counts


    @staticmethod
    def _load_hist_payload(cell: Any) -> Tuple[np.ndarray, np.ndarray]:
        """
        Accepts dict or JSON string; returns (counts, bin_edges) as numpy arrays.
        Supported keys for edges: 'bin_edges' or 'bins'.
        """
        if cell is None:
            return None, None
        if isinstance(cell, str):
            try:
                cell = json.loads(cell)
            except Exception:
                return None, None
        if isinstance(cell, dict):
            counts = cell.get("counts")
            edges = cell.get("bin_edges", cell.get("bins"))
            if counts is None or edges is None:
                return None, None
            return np.asarray(counts), np.asarray(edges)
        # if already a (counts, edges) tuple or custom object, try to coerce
        if isinstance(cell, (list, tuple)) and len(cell) == 2:
            return np.asarray(cell[0]), np.asarray(cell[1])
        return None, None

    # ---- layout & callbacks ------------------------------------------------------

    def layout(self) -> List:
        return [
            html.Div(
                [
                    html.P("Pick slices per dimension (defaults to All) to overlay histograms across files."),
                    html.Div(id=self.filters_container_id, className="row", style={"marginBottom": "8px"}),
                ]
            ),
            dcc.Graph(id=self.graph_id, figure=go.Figure()),
        ]

    def register(self, app, df_global: pl.DataFrame):
        # 1) Build dimension dropdowns (excluding x,y)
        @app.callback(
            Output(self.filters_container_id, "children"),
            Input("color-map-store", "data"),  # dummy trigger so filters rebuild if needed
        )
        def _populate_filters(_color_map) -> List:
            # discover all histogram columns and collect indices per dimension
            dim_values = defaultdict(set)
            for col in df_global.columns:
                is_hist, dims = self._is_histogram_col(col)
                if not is_hist:
                    continue
                for dim_name, dim_idx in dims.items():
                    if dim_name not in ("x", "y"):
                        dim_values[dim_name].add(dim_idx)

            # build dropdowns
            dropdowns = []
            for dim_name, indices in sorted(dim_values.items()):
                dropdown_id = {"type": f"slice-filter-{self.widget_id}", "dim": dim_name}
                options = [{"label": "All", "value": "all"}] + [
                    {"label": str(i), "value": i} for i in sorted(indices)
                ]
                dropdowns.append(
                    html.Div(
                        [
                            html.Label(f"{dim_name.upper()} slice"),
                            dcc.Dropdown(id=dropdown_id, options=options, value="all", clearable=False),
                        ],
                        className="three columns",
                    )
                )
            return dropdowns

        @app.callback(
            Output(self.graph_id, "figure"),
            Input({"type": f"slice-filter-{self.widget_id}", "dim": ALL}, "value"),
            Input("color-map-store", "data"),
        )
        def _update_overlay(selected_values, color_map: Dict[str, Any]):
            color_map = color_map or {}
            filters = {}
            if ctx.inputs_list and ctx.inputs_list[0]:
                filters = {i["id"]["dim"]: val for i, val in zip(ctx.inputs_list[0], selected_values)}

            ## 1. Filter columns (No change here) ##
            selected_count_cols = [c for c in df_global.columns if c.startswith("histogram_counts")]
            for dim, val in filters.items():
                if val == "all":
                    pattern = f"_{dim.lower()}\\d+"
                    selected_count_cols = [c for c in selected_count_cols if not re.search(pattern, c)]
                else:
                    pattern = f"_{dim.lower()}{val}(?!\\d)"
                    selected_count_cols = [c for c in selected_count_cols if re.search(pattern, c)]

            if not selected_count_cols:
                return go.Figure(layout={"title": "No histograms match the selected filters."})

            ## 2. Reshape data (No change here) ##
            id_vars = ["path", "imported_path_short"] if "imported_path_short" in df_global.columns else ["path"]
            selected_bins_cols = [c.replace("_counts", "_bins") for c in selected_count_cols]

            df_safe = df_global.clone()
            all_hist_cols = selected_count_cols + selected_bins_cols
            for col_name in all_hist_cols:
                if col_name in df_safe.columns and df_safe[col_name].dtype == pl.Object:
                    py_list = df_safe[col_name].to_list()
                    transformed_list = [list(x) if x is not None else None for x in py_list]
                    target_dtype = pl.List(pl.Int64) if col_name in selected_count_cols else pl.List(pl.Float64)
                    new_series = pl.Series(name=col_name, values=transformed_list, dtype=target_dtype)
                    df_safe = df_safe.with_columns(new_series)

            df_counts = df_safe.melt(
                id_vars=id_vars, value_vars=selected_count_cols,
                variable_name="source_column", value_name="counts"
            )
            df_bins = df_safe.melt(
                id_vars=id_vars, value_vars=selected_bins_cols,
                variable_name="source_column", value_name="bins"
            )

            df_joined = df_counts.with_columns(
                pl.col("source_column").str.replace("_counts", "").alias("join_key")
            ).join(
                df_bins.with_columns(pl.col("source_column").str.replace("_bins", "").alias("join_key")),
                on=id_vars + ["join_key"]
            )

            # ✨ FIX: Use .sum() and .to_numpy() on the Series object 'c' inside the lambda.
            long_df = (
                df_joined.drop_nulls(["counts", "bins"])
                .with_columns(
                    pl.col("bins").map_elements(lambda b: (np.array(b[:-1]) + np.array(b[1:])) / 2,
                                                return_dtype=pl.List(pl.Float64)).alias("pixel_value"),
                    pl.col("counts").map_elements(lambda c: c.to_numpy() / c.sum() if c.sum() > 0 else [],
                                                  return_dtype=pl.List(pl.Float64)).alias("frequency")
                )
                .explode(["pixel_value", "frequency"])
            )

            if long_df.is_empty():
                return go.Figure(layout={"title": "No valid histogram data for the selected filters."})

            id_col = "imported_path_short" if "imported_path_short" in df_joined.columns else "path"
            fig = go.Figure()

            # Vectorized normalization & midpoints per row, then decimate per trace
            counts_series = df_joined["counts"].to_list()
            bins_series = df_joined["bins"].to_list()
            ids_series = df_joined[id_col].to_list()
            paths_series = df_joined["path"].to_list()  # for stable line_group
            seen = set()
            for counts, bins, label, path in zip(counts_series, bins_series, ids_series, paths_series):
                if counts is None or bins is None:
                    continue
                counts = np.asarray(counts)
                if counts.size == 0 or counts.sum() == 0:
                    continue
                bins = np.asarray(bins, dtype=float)
                # midpoints
                x, counts = self._rebin_to_max_bins(counts, bins, max_bins=256)
                y = counts / counts.sum()
                x, y = self._downsample_hist(x, y, max_points=300)

                showlegend = label not in seen
                seen.add(label)

                fig.add_trace(go.Scattergl(
                    x=x, y=y,
                    mode="lines",
                    name=str(label),
                    legendgroup=str(label),   # group by file
                    showlegend=showlegend,
                    line=dict(width=1.5),
                    marker=dict(size=0),
                    opacity=0.5
                ))

            # Apply color map to matching trace names (if provided)
            if color_map:
                for tr in fig.data:
                    if tr.name in color_map:
                        tr.line.color = color_map[tr.name]

            if not fig.data:
                return go.Figure(layout={"title": "No valid histogram data for the selected filters."})

            fig.update_layout(
                title="Overlayed Pixel Histograms",
                legend_title_text="Source File",
                xaxis_title="Pixel Value",
                yaxis_title="Normalized Frequency",
                margin=dict(t=40, l=20, r=20, b=20),
                height=420,
            )
            return fig