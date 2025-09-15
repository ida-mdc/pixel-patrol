from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set

import numpy as np
import plotly.graph_objects as go
import polars as pl
from dash import html, dcc, Input, Output, ALL, ctx

from pixel_patrol_base.report.widget_categories import WidgetCategories


# Helper to parse column names like 'histogram_c0_z5' -> ('histogram', {'c':0,'z':5})
def _parse_dynamic_col(col: str, supported_metrics: List[str]) -> Optional[Tuple[str, Dict[str, int]]]:
    for metric in supported_metrics:
        if col.startswith(f"{metric}_"):
            parts = col[len(metric) + 1:].split('_')
            dims: Dict[str, int] = {}
            try:
                for part in parts:
                    dim_name = part[0]
                    dim_idx = int(part[1:])
                    dims[dim_name] = dim_idx
                return metric, dims
            except (ValueError, IndexError):
                continue
    if col in supported_metrics:
        return col, {}
    return None


class SlicedHistogramsWidget:
    # ---- Declarative spec ----
    NAME: str = "Sliced Pixel Value Histograms"
    TAB: str = WidgetCategories.DATASET_STATS.value
    REQUIRES: Set[str] = {"imported_path_short"}     # needs folder labels
    REQUIRES_PATTERNS: List[str] = [r"^histogram"]   # needs at least one histogram* column

    def __init__(self):
        self.widget_id = "sliced-histograms"

    def layout(self) -> List:
        return [
            html.Div(
                [
                    html.Div(
                        [
                            html.P("Filter by dimension slices to view the corresponding aggregated histogram."),
                            html.Div(id=f"{self.widget_id}-filters-container", className="row"),
                        ],
                        className="nine columns",
                    ),
                    html.Div(
                        [
                            html.Label("Select folders to compare:"),
                            dcc.Dropdown(
                                id=f"{self.widget_id}-folder-dropdown",
                                options=[],
                                value=[],
                                multi=True,
                                style={"marginTop": "10px", "marginBottom": "20px"},
                            ),
                        ],
                        className="three columns",
                    ),
                ],
                className="row",
                style={"alignItems": "flex-end"},
            ),
            dcc.Graph(id=f"{self.widget_id}-plot", style={"height": "600px"}),
            html.Div(id=f"{self.widget_id}-warning", className="text-warning", style={"marginBottom": "15px"}),
        ]

    def register(self, app, df_global: pl.DataFrame):
        @app.callback(
            Output(f"{self.widget_id}-filters-container", "children"),
            Output(f"{self.widget_id}-folder-dropdown", "options"),
            Output(f"{self.widget_id}-folder-dropdown", "value"),
            Input("color-map-store", "data"),
        )
        def populate_all_filters(_color_map):
            # discover histogram dimensions from column names
            all_dims = defaultdict(set)
            hist_cols = [c for c in df_global.columns if c.startswith("histogram")]
            for col in hist_cols:
                parsed = _parse_dynamic_col(col, supported_metrics=["histogram"])
                if parsed:
                    _, dims = parsed
                    for dim_name, dim_idx in dims.items():
                        all_dims[dim_name].add(dim_idx)

            # dimension filter dropdowns
            filter_dropdowns = []
            for dim_name, indices in sorted(all_dims.items()):
                dropdown_id = {"type": f"dynamic-filter-{self.widget_id}", "dim": dim_name}
                filter_dropdowns.append(
                    html.Div(
                        [
                            html.Label(f"Dimension '{dim_name.upper()}'"),
                            dcc.Dropdown(
                                id=dropdown_id,
                                options=[{"label": "All", "value": "all"}]
                                        + [{"label": i, "value": i} for i in sorted(indices)],
                                value="all",
                                clearable=False,
                            ),
                        ],
                        className="three columns",
                    )
                )

            # folder selector
            folder_names = (
                df_global["imported_path_short"].unique().to_list()
                if "imported_path_short" in df_global.columns
                else []
            )
            folder_options = [{"label": str(Path(f).name), "value": f} for f in folder_names]
            default_folders = folder_names[:2] if len(folder_names) > 1 else folder_names

            return filter_dropdowns, folder_options, default_folders

        @app.callback(
            Output(f"{self.widget_id}-plot", "figure"),
            Output(f"{self.widget_id}-warning", "children"),
            Input({"type": f"dynamic-filter-{self.widget_id}", "dim": ALL}, "value"),
            Input(f"{self.widget_id}-folder-dropdown", "value"),
            Input("color-map-store", "data"),
        )
        def update_histogram_plot(filter_values, selected_folders, color_map):
            color_map = color_map or {}
            filters = {prop["id"]["dim"]: value for prop, value in zip(ctx.inputs_list[0], filter_values)}

            # choose histogram key from selected slice
            hist_key = "histogram"
            if any(v != "all" for v in filters.values()):
                key_parts = ["histogram"] + sorted([f"{dim}{idx}" for dim, idx in filters.items() if idx != "all"])
                hist_key = "_".join(key_parts)

            if not selected_folders:
                return go.Figure(), "Please select at least one folder."
            if hist_key not in df_global.columns:
                return go.Figure(), f"The selected slice ('{hist_key}') does not exist."

            df_filtered = (
                df_global
                .filter(pl.col("imported_path_short").is_in(selected_folders))
                .select(["imported_path_short", hist_key])
                .drop_nulls()
            )
            if df_filtered.is_empty():
                return go.Figure(), "No data available for the selection."

            hist_dicts_obj = df_filtered.get_column(hist_key)

            # keep only valid dicts
            valid_mask = np.array([
                isinstance(d, dict) and "counts" in d and "bins" in d and len(d["counts"]) > 0
                for d in hist_dicts_obj
            ])
            if not np.any(valid_mask):
                return go.Figure(), "No valid histogram data found for the selection."

            valid_hist_dicts = hist_dicts_obj.filter(valid_mask)
            folder_keys = df_filtered.get_column("imported_path_short").to_numpy()[valid_mask]

            try:
                bin_edges = np.asarray(valid_hist_dicts[0]["bins"])
                counts_all = np.stack([np.asarray(d["counts"]) for d in valid_hist_dicts]).astype(np.float64)
            except (ValueError, TypeError, KeyError, IndexError):
                return go.Figure(), "Histogram data is malformed (e.g., inconsistent bin counts)."

            # aggregate by folder (mean)
            unique_folders, group_ids = np.unique(folder_keys, return_inverse=True)
            sums = np.zeros((len(unique_folders), counts_all.shape[1]), dtype=np.float64)
            np.add.at(sums, group_ids, counts_all)
            counts = np.bincount(group_ids, minlength=len(unique_folders))
            mean_counts = np.divide(sums, counts[:, None], out=np.zeros_like(sums), where=counts[:, None] != 0)
            mean_results = dict(zip(unique_folders, mean_counts))

            # build chart
            chart = go.Figure()
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            for folder, mean_hist_counts in mean_results.items():
                total_sum = mean_hist_counts.sum()
                if total_sum == 0:
                    continue
                normalized_counts = mean_hist_counts / total_sum
                color = color_map.get(folder)

                chart.add_trace(go.Bar(
                    x=bin_centers,
                    y=normalized_counts,
                    name=Path(folder).name,
                    marker_color=color,
                    opacity=0.7,
                ))

            chart.update_layout(
                title=f"Aggregated Pixel Value Histogram for: <b>{hist_key}</b>",
                xaxis_title=f"Pixel Value (Range: {bin_edges[0]:.2f} to {bin_edges[-1]:.2f})",
                yaxis_title="Normalized Frequency",
                legend_title="Folder Name",
                bargap=0,
                barmode="overlay",
            )
            return chart, ""
