from pathlib import Path
from typing import List, Dict, Any
import warnings
import polars as pl
import plotly.graph_objects as go
from dash import html, dcc, Input, Output
import numpy as np
from pixel_patrol_base.report.widget_categories import WidgetCategories
from pixel_patrol_base.plugins.processors.histogram_processor import safe_hist_range


class DatasetHistogramWidget:
    # ---- Declarative spec (plugin registry expects these at class-level) ----
    NAME: str = "Pixel Value Histograms"
    TAB: str = WidgetCategories.DATASET_STATS.value
    # Only require a histogram-prefixed column to display the widget; the widget
    # itself handles missing path/imported_path_short gracefully when building options.
    REQUIRES = set()
    REQUIRES_PATTERNS: List[str] = [r"^histogram"]

    @property
    def tab(self) -> str:
        return self.TAB

    @property
    def name(self) -> str:
        return self.NAME

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
            html.Div([
                html.Label("Histogram plot mode:"),
                dcc.RadioItems(
                    id="histogram-remap-mode",
                    options=[
                        {"label": "Spread the content of the image onto range 0-255 for comparison", "value": "remap"},
                        {"label": "Show native histogram bins based on the images' content (show true ranges and values)", "value": "native"},
                    ],
                    value="remap",
                    labelStyle={"display": "inline-block", "marginRight": "12px"},
                ),
            ], style={"marginBottom": "8px"}),
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

    def _map_counts_to_fixed_bins(self, counts: np.ndarray, edges: np.ndarray, target_bins: int = 256,
                                  target_min: float = 0.0, target_max: float = 256.0,
                                  raw_max: float | None = None) -> tuple[np.ndarray, float, float]:
        """
        Map source histogram counts+edges to a fixed target grid of `target_bins` spanning [target_min, target_max].
        Each source bin's count is assigned to the single target bin whose center contains the mapped source-bin center.
        Returns (normalized_out_counts, src_min, src_max).
        """
        counts, edges = np.asarray(counts, dtype=float), np.asarray(edges, dtype=float)
        if counts.size == 0:
            return np.zeros(target_bins, dtype=float), None, None
        if edges.size == 0:
            warnings.warn("Histogram edges not provided; assuming uniform integer edges in (0, 256).")
            edges = np.arange(counts.size + 1).astype(float)
        if edges.size != counts.size + 1:
            # throw warning that images lack edges
            warnings.warn("Histogram edges not provided; assuming integer bin centers.")
        if edges[0] == edges[-1]:
            edges[-1] += 1.0  # avoid division by zero
        
        edges = np.asarray(edges, dtype=float)
        src_min, src_max = edges[0], edges[-1]

        # If the source range is degenerate (src_max == src_min) treat it as a
        # special case: map all counts to the single target bin corresponding to
        # the raw value (prefer raw_max if provided). This avoids collapsing a
        # single-source-bin histogram into an arbitrary target bin.
        denom = (src_max - src_min)
        if denom == 0.0:
            # decide representative value inside the degenerate bin
            rep_value = raw_max if raw_max is not None else float(src_min)
            # scale rep_value into target grid
            if (target_max - target_min) == 0.0:
                target_idx = 0
            else:
                scaled_rep = (rep_value - target_min) / (target_max - target_min) * target_bins
                target_idx = int(np.floor(scaled_rep))
            target_idx = int(np.clip(target_idx, 0, target_bins - 1))
            out = np.zeros(target_bins, dtype=float)
            out[target_idx] = counts.sum()
            s = out.sum()
            if s > 0:
                out = out / s
            return out, float(src_min), float(src_max)

        # Non-degenerate case: map each source bin using a representative point
        # strictly inside the bin (nextafter of right_edge toward left_edge).
        right_edges = edges[1:]
        left_edges = edges[:-1]
        reps = np.nextafter(right_edges, left_edges)
        scaled = (reps - src_min) / denom * (target_max - target_min) + target_min
        idx = np.floor(scaled).astype(int)
        idx = np.clip(idx, 0, target_bins - 1)
        # Optionally, ensure the source bin containing raw_max maps to the
        # final target bin if explicitly requested by the caller. This is a
        # safety measure but not the default mapping behavior.
        if raw_max is not None and edges is not None and edges.size >= 2:
            src_bin = np.digitize([raw_max], edges) - 1
            # Only force mapping to the final target bin if the source bin that
            # contains raw_max is itself the last source bin. This avoids
            # remapping degenerate or very-small-range images (e.g., min==max==0)
            # to the top of the target grid.
            if src_bin.size > 0 and 0 <= src_bin[0] < idx.size and src_bin[0] == (idx.size - 1):
                idx[src_bin[0]] = target_bins - 1

        out = np.zeros(target_bins, dtype=float)
        np.add.at(out, idx, counts)

        s = out.sum()
        if s > 0:
            out = out / s
        return out, float(src_min), float(src_max)

    def _compute_edges(self, counts: np.ndarray, minv: float | None, maxv: float | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute left-oriented edges, centers and widths for a histogram given counts and optional min/max.

        Behavior:
        - If minv/maxv are provided, compute right-edge using the processor's `safe_hist_range`
          so the widget uses the exact same right-edge semantics as the processor.
        - Edges is an array of length (n_bins + 1). The returned `lefts` are edges[:-1]
          and represent the left boundary of each bin (left-oriented representation).
        - If minv/maxv are missing the function assumes integer bins starting at 0 with width 1.

        Returns (edges, lefts, widths) where:
        - edges: float array length n_bins+1
        - lefts: edges[:-1] (left-oriented x positions for bars)
        - widths: np.diff(edges) (bin widths)
        """
        n = counts.size
        if minv is None or maxv is None:
            # assume integer bins 0..n-1 with unit width
            edges = np.arange(n + 1).astype(float)
            lefts = edges[:-1]
            widths = np.diff(edges)
            return edges, lefts, widths

        # compute adjusted right-edge using shared helper to maintain identical semantics
        # Use an integer-typed sample when minv/maxv are integer-valued so
        # safe_hist_range mirrors the processor's integer-branch behavior.
        if float(minv).is_integer() and float(maxv).is_integer():
            sample = np.array([int(minv), int(maxv)], dtype=np.int64)
        else:
            sample = np.array([minv, maxv], dtype=float)
        smin, smax, max_adj = safe_hist_range(sample)

        # Instead of using linspace (which computes edges and then diffs),
        # compute a uniform bin width and construct left edges directly. This
        # makes the arithmetic explicit and avoids subtle floating rounding
        # differences that can arise from linspace endpoints.
        smin_f = float(smin)
        max_adj_f = float(max_adj)
        width = (max_adj_f - smin_f) / float(n)
        lefts = smin_f + np.arange(n, dtype=float) * width
        edges = np.concatenate([lefts, np.array([smin_f + n * width], dtype=float)])
        widths = np.full(n, width, dtype=float)
        return edges, lefts, widths

    def register(self, app, df_global: pl.DataFrame):
        @app.callback(
            Output("histogram-dimension-dropdown", "options"),
            Output("histogram-dimension-dropdown", "value"),
            Output("histogram-folder-dropdown", "options"),
            Output("histogram-folder-dropdown", "value"),
            Input("color-map-store", "data"),
        )
        def populate_dropdowns(color_map):
            histogram_columns = [
                col for col in df_global.columns if col.startswith("histogram_counts")
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

        @app.callback(
            Output("histogram-plot", "figure"),
            Output("dataset-histograms-warning", "children"),
            Input("color-map-store", "data"),
            Input("histogram-remap-mode", "value"),
            Input("histogram-dimension-dropdown", "value"),
            Input("histogram-folder-dropdown", "value"),
        )
        def update_histogram_plot(color_map, remap_mode, histogram_key, selected_folders):
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
                # Attempt to load corresponding min/max columns if present (processor now stores ranges)
                min_key = histogram_key.replace("counts", "min")
                max_key = histogram_key.replace("counts", "max")
                counts_list = df_group[histogram_key].to_list()
                min_list = (
                    df_group[min_key].to_list() if min_key in df_group.columns else [None] * len(counts_list)
                )
                max_list = (
                    df_group[max_key].to_list() if max_key in df_group.columns else [None] * len(counts_list)
                )
                names = df_group["name"].to_list() if "name" in df_group.columns else [""] * len(counts_list)

                # Remap mode: map each histogram to fixed 0..255 grid using helper
                if remap_mode == "remap":
                    remapped = []
                    color = color_map.get(folder, None) if color_map else None
                    for counts, minv, maxv, file_name in zip(counts_list, min_list, max_list, names):
                        if counts is None:
                            continue
                        # reconstruct source edges from min/max (using shared helper) so
                        # remapping uses the exact same right-edge semantics as the processor.
                        if minv is None or maxv is None:
                            src_edges = None
                        else:
                            _, src_lefts, _ = self._compute_edges(np.asarray(counts), minv, maxv)
                            # recover full edges from lefts and widths by appending the last edge
                            # compute widths assuming unit spacing if degenerate
                            if src_lefts.size > 0:
                                # widths may be non-uniform; reconstruct edges by taking lefts and adding last left + width
                                # call compute_edges to also return edges if needed
                                src_edges, _, _ = self._compute_edges(np.asarray(counts), minv, maxv)
                            else:
                                src_edges = None
                        mapped, src_min, src_max = self._map_counts_to_fixed_bins(np.asarray(counts), src_edges, raw_max=maxv)
                        remapped.append(mapped)
                        hover_extra = (
                            f"<br>Range: {src_min:.3f}..{src_max:.3f}" if (src_min is not None and src_max is not None) else ""
                        )
                        chart.add_trace(
                            go.Scatter(
                                x=list(range(mapped.size)),
                                y=mapped,
                                mode="lines",
                                name=Path(folder).name,
                                line=dict(width=1, color=color),
                                opacity=0.2,
                                showlegend=False,
                                legendgroup=Path(folder).name,
                                hovertemplate=(
                                    f"File: {file_name}<br>Pixel: %{{x}}<br>Freq: %{{y:.3f}}{hover_extra}<extra></extra>"
                                ),
                            )
                        )
                    if not remapped:
                        continue
                    mean_hist = np.mean(remapped, axis=0)
                    mean_hist = mean_hist / mean_hist.sum() if mean_hist.sum() > 0 else mean_hist
                    color = color_map.get(folder, None) if color_map else None
                    chart.add_trace(
                        go.Scatter(
                            x=list(range(mean_hist.size)),
                            y=mean_hist,
                            mode="lines",
                            name=Path(folder).name,
                            line=dict(width=2, color=color),
                            fill="tozeroy",
                            opacity=0.6,
                            legendgroup=Path(folder).name,
                            hovertemplate=(
                                f"Folder: {Path(folder).name}<br>Pixel: %{{x}}<br>Mean Freq: %{{y:.3f}}<extra></extra>"
                            ),
                        )
                    )
                else:
                    # Native mode: plot individual histograms at their native bin centers/bars
                    # and compute the group mean without shifting. If images in the group
                    # have differing ranges, remap each image to the group's full range
                    # before averaging (so the mean is meaningful across different ranges).
                    valid_items = [
                        (np.asarray(c), mn, mx, nm)
                        for c, mn, mx, nm in zip(counts_list, min_list, max_list, names)
                        if c is not None
                    ]
                    if not valid_items:
                        continue

                    # Detect whether all items share the same (min,max)
                    mins = [v for (_, v, _, _) in valid_items if v is not None]
                    maxs = [v for (_, _, v, _) in valid_items if v is not None]
                    same_range = False
                    if mins and maxs and len(mins) == len(valid_items) and len(maxs) == len(valid_items):
                        # all items provide min/max; check if they are all equal within tolerance
                        same_range = all(np.isclose(m, mins[0]) for m in mins) and all(np.isclose(M, maxs[0]) for M in maxs)

                    color = color_map.get(folder, None) if color_map else None

                    # Plot individual histograms as semi-transparent bars at native centers
                    for counts, minv, maxv, file_name in valid_items:
                        h = counts.astype(float)
                        if h.sum() > 0:
                            h_norm = h / h.sum()
                        else:
                            h_norm = h

                        # compute edges/lefts/widths using helper (left-oriented)
                        edges, lefts, widths = self._compute_edges(np.asarray(h), minv, maxv)

                        # if lefts are integer-aligned, cast to int for cleaner hover/labels
                        if np.allclose(lefts, np.round(lefts)):
                            lefts = np.round(lefts).astype(int)
                            widths = np.ones_like(lefts)

                        # compute centers for hover display (left + width/2)
                        centers = lefts + widths / 2.0
                        # build hover text per-bar that shows center and nearest integer
                        hover_texts = [
                            f"File: {file_name}<br>Pixel center: {c:.3f}<br>Nearest: {int(round(c))}<br>Freq: {float(hv):.3f}"
                            for c, hv in zip(centers, h_norm)
                        ]
                        # use Bar so the bins are represented accurately (no visual center-shift)
                        chart.add_trace(
                            go.Bar(
                                x=list(lefts),
                                y=list(h_norm),
                                width=list(widths),
                                name=Path(folder).name,
                                marker=dict(color=color, opacity=0.3),
                                showlegend=False,
                                legendgroup=Path(folder).name,
                                text=hover_texts,
                                hovertemplate="%{text}<extra></extra>",
                            )
                        )

                    # Compute group mean. If ranges differ, remap to group's full range first.
                    if same_range:
                        # all share same min/max -> average counts directly and normalize
                        mats = [c.astype(float) for c, _, _, _ in valid_items]
                        mean_hist = np.mean(mats, axis=0)
                        mean_hist = mean_hist / mean_hist.sum() if mean_hist.sum() > 0 else mean_hist
                        # left edges from the common range using helper
                        _, mean_lefts, mean_widths = self._compute_edges(np.zeros_like(mean_hist), valid_items[0][1], valid_items[0][2])
                        mean_centers = mean_lefts
                    else:
                        # remap each histogram to the group's combined min/max and average
                        group_min = min([v for v in mins]) if mins else 0.0
                        group_max = max([v for v in maxs]) if maxs else 255.0
                        # compute group edges/centers for plotting
                        n_bins = valid_items[0][0].size
                        group_edges, group_centers, group_widths = self._compute_edges(np.zeros(n_bins), group_min, group_max)
                        remapped = []
                        for counts, minv, maxv, _ in valid_items:
                            # compute source edges
                            src_edges, _, _ = self._compute_edges(np.asarray(counts), minv, maxv)
                            # remap into group's real-value edge span (group_edges are full edges)
                            mapped, _, _ = self._map_counts_to_fixed_bins(np.asarray(counts), src_edges, target_bins=n_bins, target_min=group_edges[0], target_max=group_edges[-1], raw_max=maxv)
                            remapped.append(mapped)
                        if not remapped:
                            continue
                        mean_hist = np.mean(remapped, axis=0)
                        mean_hist = mean_hist / mean_hist.sum() if mean_hist.sum() > 0 else mean_hist
                        mean_centers = group_edges[:-1]
                    # prepare hover text for mean trace
                    try:
                        mean_widths  # type: ignore
                    except NameError:
                        mean_widths = None
                    if mean_widths is not None:
                        mean_display_centers = np.asarray(mean_centers) + np.asarray(mean_widths) / 2.0
                    else:
                        mean_display_centers = np.asarray(mean_centers)

                    mean_hover_texts = [
                        f"Folder: {Path(folder).name}<br>Pixel center: {float(c):.3f}<br>Nearest: {int(round(c))}<br>Mean Freq: {float(v):.3f}"
                        for c, v in zip(mean_display_centers, mean_hist)
                    ]

                    # Plot the mean trace at the display centers (left + width/2)
                    chart.add_trace(
                        go.Scatter(
                            x=list(mean_display_centers),
                            y=list(mean_hist),
                            mode="lines",
                            name=Path(folder).name,
                            line=dict(width=2, color=color),
                            fill="tozeroy",
                            opacity=0.6,
                            legendgroup=Path(folder).name,
                            text=mean_hover_texts,
                            hovertemplate="%{text}<extra></extra>",
                        )
                    )
            chart.update_layout(
                title="Mean Pixel Value Histogram (per group)",
                xaxis_title=("Pixel Value, normalized to 0-255" if remap_mode == "remap" else "Pixel Value"),
                yaxis_title="Normalized Frequency",
                legend_title="Folder name",
            )
            return chart, ""
