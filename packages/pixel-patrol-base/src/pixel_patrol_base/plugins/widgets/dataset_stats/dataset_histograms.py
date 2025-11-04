from pathlib import Path
from typing import List, Tuple, Optional
import polars as pl
import plotly.graph_objects as go
from dash import html, dcc, Input, Output
import numpy as np
from pixel_patrol_base.report.widget_categories import WidgetCategories
from pixel_patrol_base.plugins.processors.histogram_processor import safe_hist_range


class DatasetHistogramWidget:
    # ---- Declarative spec ----
    NAME: str = "Pixel Value Histograms"
    TAB: str = WidgetCategories.DATASET_STATS.value
    REQUIRES = set()
    REQUIRES_PATTERNS: List[str] = [r"^histogram"]

    @property
    def tab(self) -> str:
        return self.TAB

    @property
    def name(self) -> str:
        return self.NAME

    def required_columns(self) -> List[str]:
        return ["histogram"]

    # ------------------------------ UI Layout ------------------------------ #
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
                        options=[],
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
                    html.Label("Histogram plot mode:"),
                    dcc.RadioItems(
                        id="histogram-remap-mode",
                        options=[
                            {
                                "label": "Fixed 0–255 bins",
                                "value": "shape",
                            },
                            {
                                "label": "Bin on the native pixel range",
                                "value": "native",
                            },
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
                            "The histograms are computed per image and grouped by the selected folder names. The computation focuses on each images' content range, hence it is based on the range of pixel values present in the image data. ",
                            "The mean histogram for each selected group (folder name) is shown as a bold line. ",
                            "Histograms are normalized to sum to 1. \n",
                            "Select your dimension and the visualization mode: ",
                            html.Ul(
                                [
                                    html.Li(
                                        "Show the bins 0-255 to compare the images based on their contents' distribution shapes: "
                                        "This mode displays 256 bins (0-255) regardless of the actual pixel value range. "
                                        "This allows comparing the shape of the distributions across images with different intensity ranges or even data types."
                                    ),
                                    html.Li(
                                        "Map the bins in regards to the used pixel value ranges across all images: "
                                        "This mode uses the actual pixel value ranges found in the images to define the histogram bins. "
                                        "This allows investigation of where most pixel values lie in absolute terms, but may make shape comparison more difficult if the images have different intensity ranges."
                                    ),
                                ]
                            ),
                        ]
                    ),
                ],
            ),
        ]

    # ---------------------------- Core helpers ----------------------------- #
    @staticmethod
    def _edges_from_minmax(n_bins: int, minv: float, maxv: float) -> np.ndarray:
        """
        Folder-level edges using safe_hist_range to mirror processor semantics (left-bounded,
        right edge adjusted via max_adj).
        Args:
            n_bins (int): Number of histogram bins.
            minv (float): Minimum pixel value for the histogram range.
            maxv (float): Maximum pixel value present in the image for the histogram range.
        Returns:
            np.ndarray: Array of bin edges of length n_bins + 1.
        """
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
    def _folder_minmax_using_polars(
        df_group: pl.DataFrame, min_key: str, max_key: str
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Fast folder-wide min/max using Polars.
        Args:
            df_group (pl.DataFrame): DataFrame filtered to the folder group.
            min_key (str): Column name for per-image minimum values.
            max_key (str): Column name for per-image maximum values.
        Returns:
            Tuple[Optional[float], Optional[float]]: (folder_min, folder_max)
        """
        if min_key in df_group.columns and max_key in df_group.columns:
            agg = df_group.select(
                [
                    pl.col(min_key).min().alias("_min"),
                    pl.col(max_key).max().alias("_max"),
                ]
            ).to_dict(as_series=False)
            gmin = agg["_min"][0]
            gmax = agg["_max"][0]
            return (
                float(gmin) if gmin is not None else None,
                float(gmax) if gmax is not None else None,
            )
        return None, None

    @staticmethod
    def _compute_edges(
        counts: np.ndarray, minv: float | None, maxv: float | None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute left-oriented edges/widths consistent with processor semantics via safe_hist_range.

        Args:
            counts (np.ndarray): 1-D array of non-negative counts (frequencies) for each bin.
            minv (float | None): Minimum pixel value for the histogram range.
            maxv (float | None): Maximum pixel value present in the image for the histogram range.
        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: edges, lefts, widths
        """
        n = counts.size
        if minv is None or maxv is None:
            # assume bins 0..n
            edges = np.arange(n + 1).astype(float)
            lefts = edges[:-1]
            widths = np.diff(edges)
            return edges, lefts, widths

        # compute adjusted right-edge
        if float(minv).is_integer() and float(maxv).is_integer():
            sample = np.array([int(minv), int(maxv)], dtype=np.int64)
        else:
            sample = np.array([minv, maxv], dtype=float)
        smin, _smax, max_adj = safe_hist_range(sample)

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

    @staticmethod
    def _rebin_via_cdf(
        counts: np.ndarray, src_edges: np.ndarray, tgt_edges: np.ndarray
    ) -> np.ndarray:
        """
        Rebin via CDF, evaluated at target edges; returns per-target-bin probability.

        Rebin histogram counts from source bin edges to target bin edges by evaluating the
        CDF at the target edges and taking differences.
        The method treats the input counts as frequencies for contiguous source bins defined
        by src_edges (length = counts.size + 1). It constructs a stepwise CDF from the
        source histogram, interpolates that CDF at the target bin edges, and returns the
        probability mass assigned to each target bin as the difference of interpolated CDF
        values at consecutive target edges.

        Args:
            counts (np.ndarray):
                1-D array of non-negative counts (frequencies) for each source bin.
                Converted to float internally. May be empty.
            src_edges (np.ndarray):
                1-D array of source bin edges of length counts.size + 1. Edges must be
                monotonic (increasing). Values outside the range are supported via
                interpolation/clamping rules described below.
            tgt_edges (np.ndarray):
                1-D array of target bin edges (length M+1) at which the CDF is evaluated.
                The function returns an array of length M giving the probability mass in
                each target bin.
        Returns:
            np.ndarray:
                1-D array of length len(tgt_edges) - 1 containing the probability mass for
                each target bin (sums to 1.0 for a strictly positive total count). If
                counts is empty or the total count is <= 0, returns an array of zeros of
                the appropriate length. The returned values are floats.
        Raises:
            ValueError:
                If src_edges does not have length counts.size + 1.
        Notes:
            - The source CDF is constructed as a left-continuous step function with values
              starting at 0.0 and increasing by counts/total at each source edge.
            - Interpolation of the CDF at target edges uses numpy.interp with left=0.0
              and right=1.0: target edges left of the first source edge map to 0.0, and
              edges right of the last source edge map to 1.0.
            - Output is suitable as per-target-bin probabilities (not raw counts).
        """
        counts = np.asarray(counts, float)
        se = np.asarray(src_edges, float)
        te = np.asarray(tgt_edges, float)
        if counts.size == 0:
            return np.zeros(te.size - 1, float)
        if se.size != counts.size + 1:
            raise ValueError("src_edges must have len(counts)+1")

        total = counts.sum()
        if total <= 0:
            return np.zeros(te.size - 1, float)

        # Step CDF at source edges
        cdf_src = np.concatenate([[0.0], np.cumsum(counts) / total])  # len N+1
        # Interpolate CDF at target edges
        cdf_t = np.interp(te, se, cdf_src, left=0.0, right=1.0)
        out = np.diff(cdf_t)  # sums to 1
        return out

    # ---------------------------- Dash callbacks --------------------------- #
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
        def update_histogram_plot(
            color_map, remap_mode, histogram_key, selected_folders
        ):
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

                # Column keys
                min_key = histogram_key.replace("counts", "min")
                max_key = histogram_key.replace("counts", "max")

                counts_list = df_group[histogram_key].to_list()
                min_list = (
                    df_group[min_key].to_list()
                    if min_key in df_group.columns
                    else [None] * len(counts_list)
                )
                max_list = (
                    df_group[max_key].to_list()
                    if max_key in df_group.columns
                    else [None] * len(counts_list)
                )
                names = (
                    df_group["name"].to_list()
                    if "name" in df_group.columns
                    else [""] * len(counts_list)
                )

                n_bins = (
                    len(counts_list[0])
                    if counts_list and counts_list[0] is not None
                    else 256
                )
                color = color_map.get(folder, None) if color_map else None

                if remap_mode == "shape":
                    # ------------------- Shape (bin-index) mode ------------------- #
                    mats = []
                    for counts, minv, maxv, file_name in zip(
                        counts_list, min_list, max_list, names
                    ):
                        if counts is None:
                            continue
                        h = np.asarray(counts, float)
                        s = h.sum()
                        p = (h / s) if s > 0 else h
                        mats.append(p)

                        # make p + bin_width*0.5 to shift bar centers to the right so it aligns 

                        chart.add_trace(
                            go.Bar(
                                x=list(range(p.size + 1)),
                                y=list(p)+[0],
                                width=1,
                                name=Path(folder).name,
                                marker=dict(color=color, opacity=0.25),
                                showlegend=False,
                                legendgroup=Path(folder).name,
                                hovertemplate=(
                                    f"File: {file_name}<br>Bin idx: %{{x}}<br>Prob: %{{y:.3f}}"
                                    + (
                                        f"<br>Range: {minv:.3f}..{maxv:.3f}"
                                        if (minv is not None and maxv is not None)
                                        else ""
                                    )
                                    + "<extra></extra>",
                                ),
                                # barmode="overlay",
                                # bargap=0.0,
                                # bargroupgap=0.0,
                                offset=0.0,
                            )
                        )
                        print("x:", list(range(p.size + 1)))
                    if not mats:
                        continue

                    mean_hist = np.mean(mats, axis=0)
                    mean_hist = (
                        mean_hist / mean_hist.sum()
                        if mean_hist.sum() > 0
                        else mean_hist
                    )

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
                                f"Folder: {Path(folder).name}<br>Bin idx: %{{x}}<br>Mean Prob: %{{y:.3f}}<extra></extra>"
                            ),
                        )
                    )

                else:
                    # --------------------- Native (true-range) --------------------- #
                    valid_items = [
                        (np.asarray(c, float), mn, mx, nm)
                        for c, mn, mx, nm in zip(counts_list, min_list, max_list, names)
                        if c is not None
                    ]
                    if not valid_items:
                        continue

                    # Per-image bars at native edges
                    for counts, minv, maxv, file_name in valid_items:
                        edges, lefts, widths = self._compute_edges(counts, minv, maxv)
                        total = counts.sum()
                        h_norm = counts / total if total > 0 else counts

                        chart.add_trace(
                            go.Bar(
                                x=list(lefts),
                                y=list(h_norm),
                                width=list(widths),
                                name=Path(folder).name,
                                marker=dict(color=color, opacity=0.25),
                                showlegend=False,
                                legendgroup=Path(folder).name,
                                hovertemplate=(
                                    f"File: {file_name}<br>Pixel value: %{{x:.3f}}<br>Freq: %{{y:.3f}}<extra></extra>"
                                ),
                                offset=0.0,
                            )
                        )

                    # Folder group edges from Polars-derived min/max (data-derived native range)
                    gmin, gmax = self._folder_minmax_using_polars(
                        df_group, min_key, max_key
                    )
                    if gmin is None or gmax is None:
                        # Fallback from per-image lists
                        valid_mins = [m for _, m, _, _ in valid_items if m is not None]
                        valid_maxs = [M for _, _, M, _ in valid_items if M is not None]
                        if valid_mins and valid_maxs:
                            gmin, gmax = float(np.min(valid_mins)), float(
                                np.max(valid_maxs)
                            )

                    if gmin is not None and gmax is not None:
                        group_edges = self._edges_from_minmax(n_bins, gmin, gmax)
                    else:
                        group_edges, _, _ = self._compute_edges(
                            valid_items[0][0], valid_items[0][1], valid_items[0][2]
                        )

                    # Rebin each image to group_edges via CDF; then mean (equal image weight)
                    mats = []
                    for counts, minv, maxv, _ in valid_items:
                        src_edges, _, _ = self._compute_edges(counts, minv, maxv)
                        reb_prob = self._rebin_via_cdf(
                            counts, src_edges, group_edges
                        )  # sums to 1
                        mats.append(reb_prob)

                    mean_hist = np.mean(mats, axis=0)
                    # TODO: I think mean centers are still not correct here
                    mean_centers = group_edges[:-1] #+ 0.5 * np.diff(group_edges) to have the center in the middle of the bin
                    mean_hover_texts = [
                        f"Mean of folder: {Path(folder).name}<br>Pixel value: {float(c):.3f}<br>Nearest pixel: {int(round(c))}<br>Mean Prob: {float(v):.3f}"
                        for c, v in zip(mean_centers, mean_hist)
                    ]
                    chart.add_trace(
                        go.Scatter(
                            x=list(mean_centers),
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
                xaxis_title=(
                    "Pixel intensity - Fixed 0–255 Bins"
                    if remap_mode == "shape"
                    else "Native-range bins (actual values)"
                ),
                yaxis_title="Normalized Frequency",
                legend_title="Folder name",
            )
            return chart, ""
