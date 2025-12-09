from typing import List, Set
import polars as pl
import numpy as np
from dash import html, dcc, Input, Output, ALL, no_update

from pixel_patrol_base.report.widget_categories import WidgetCategories
from pixel_patrol_base.report.base_widget import BaseReportWidget
from pixel_patrol_base.report.factory import create_dimension_selectors, plot_grouped_histogram
from pixel_patrol_base.report.data_utils import (
    extract_dimension_tokens,
    find_best_matching_column,
    aggregate_histograms_by_group,
    compute_histogram_edges
)
from pixel_patrol_base.report.global_controls import (
    apply_global_config,
    GLOBAL_CONFIG_STORE_ID,
)


class DatasetHistogramWidget(BaseReportWidget):
    NAME: str = "Pixel Value Histograms"
    TAB: str = WidgetCategories.DATASET_STATS.value
    REQUIRES: Set[str] = {"name"}
    REQUIRES_PATTERNS: List[str] = [r"^histogram"]

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._df: pl.DataFrame | None = None

    @property
    def help_text(self) -> str:
        return (
            "The histograms are computed **per image** and grouped based on your groupings.  \n"
            "They are normalized to sum to **1**, and the **mean histogram per group** is shown as a bold line.\n\n"
            "**Modes**\n"
            "- **0–255 bins (shape comparison)** \n"
            "  Uses 256 fixed bins (0–255) regardless of the actual pixel range.\n"
            "- **Native pixel-range bins** \n"
            "  Bins are defined using the actual min/max pixel values across the selected images.\n"
        )

    def get_content_layout(self) -> List:
        return [
            html.P(id="dataset-histogram-warning", className="text-warning", style={"marginBottom": "15px"}),

            # --- Controls Section ---
            html.Div([
                # 1. Dimensions
                html.Div([
                    html.Label("Select histogram dimensions to plot:", style={"fontWeight": "bold"}),
                    html.Div(id="histogram-filters-container", style={"marginTop": "5px"}),
                    dcc.Store(id="histogram-dims-store"),
                ], style={"marginBottom": "15px"}),

                # 2. Mode
                html.Div([
                    html.Label("Histogram plot mode:", style={"fontWeight": "bold"}),
                    dcc.RadioItems(
                        id="histogram-remap-mode",
                        options=[
                            {"label": "Fixed 0–255 bins (Shape)", "value": "shape"},
                            {"label": "Native pixel range (Absolute)", "value": "native"},
                        ],
                        value="shape",
                        labelStyle={"display": "inline-block", "marginRight": "15px"},
                        style={"marginTop": "5px"}
                    ),
                ], style={"marginBottom": "15px"}),

                # 3. Group Selection
                html.Div([
                    html.Label("Select specific groups to compare (optional):", style={"fontWeight": "bold"}),
                    dcc.Dropdown(
                        id="histogram-group-dropdown",
                        options=[],
                        value=[],
                        multi=True,
                        placeholder="All groups (default)",
                        style={"width": "100%", "maxWidth": "600px"}
                    ),
                ], style={"marginBottom": "15px"}),

                # 4. File Overlay (Moved to Top as requested)
                html.Div([
                    html.Label("Overlay specific file (optional):", style={"fontWeight": "bold"}),
                    dcc.Dropdown(
                        id="histogram-file-dropdown",
                        options=[],
                        value=None,
                        clearable=True,
                        placeholder="Search for a filename...",
                        style={"width": "100%", "maxWidth": "600px"}
                    ),
                ], style={"marginBottom": "25px"}),
            ]),

            # --- Plot Section ---
            dcc.Graph(id="histogram-plot", style={"height": "600px"}),
        ]

    def register(self, app, df_global: pl.DataFrame):
        self._df = df_global

        # 1. Populate basic controls
        app.callback(
            Output("histogram-filters-container", "children"),
            Output("histogram-dims-store", "data"),
            Output("histogram-group-dropdown", "options"),
            Output("histogram-group-dropdown", "value"),
            Input("color-map-store", "data"),  # Trigger on load/map change
            Input(GLOBAL_CONFIG_STORE_ID, "data"),
        )(self._set_control_options)

        # 2. Populate file dropdown based on selected groups
        app.callback(
            Output("histogram-file-dropdown", "options"),
            Output("histogram-file-dropdown", "value"),
            Input("histogram-group-dropdown", "value"),
            Input(GLOBAL_CONFIG_STORE_ID, "data"),
            prevent_initial_call=False,
        )(self._update_file_options)

        # 3. Main Plot Update
        app.callback(
            Output("histogram-plot", "figure"),
            Output("dataset-histogram-warning", "children"),
            Input("color-map-store", "data"),
            Input("histogram-remap-mode", "value"),
            Input({"type": "histogram-dim-filter", "dim": ALL}, "value"),
            Input("histogram-group-dropdown", "value"),
            Input("histogram-file-dropdown", "value"),
            Input("histogram-dims-store", "data"),
            Input(GLOBAL_CONFIG_STORE_ID, "data"),
        )(self._update_plot)

    def _set_control_options(self, _color_map, global_config):
        df = self._df
        if df is None: return [], [], [], []

        children, dims_order = create_dimension_selectors(
            tokens=extract_dimension_tokens(df.columns, "histogram_counts"),
            id_type="histogram-dim-filter",
        )

        df_processed, group_col = apply_global_config(df, global_config)
        if group_col is None or df_processed.is_empty():
            return children, dims_order, [], []

        group_values = df_processed[group_col].unique().sort().to_list()
        group_options = [{"label": str(g), "value": g} for g in group_values]

        # Default: Don't pre-select specific groups, show all implicitly (or select first 2 if you prefer)
        return children, dims_order, group_options, []

    def _update_file_options(self, selected_groups, global_config):
        df = self._df
        if df is None: return [], None

        df_processed, group_col = apply_global_config(df, global_config)

        # Filter options based on the "Group" dropdown to narrow down file search
        if selected_groups:
            df_processed = df_processed.filter(pl.col(group_col).is_in(selected_groups))

        # Limit to 500 files for performance in dropdown
        names = df_processed["name"].unique().head(500).to_list()
        options = [{"label": n, "value": n} for n in names]
        return options, no_update

    def _update_plot(self, color_map, remap_mode, dim_values, selected_groups, selected_file, dims_order,
                     global_config):
        df = self._df
        if df is None: return no_update, "No data available."

        # 1. Resolve Dimensions
        selections = dict(zip(dims_order, dim_values)) if (dims_order and dim_values) else {}

        # 2. Identify Columns
        base = "histogram_counts"
        histogram_key = find_best_matching_column(df.columns, base, selections) or base
        if histogram_key not in df.columns:
            return no_update, f"Column {histogram_key} not found."

        suffix = histogram_key.replace("histogram_counts", "")
        min_key, max_key = f"histogram_min{suffix}", f"histogram_max{suffix}"

        # 3. Apply Filters (Global + Local Group Selection)
        df_processed, group_col = apply_global_config(df, global_config)

        if selected_groups:
            df_processed = df_processed.filter(pl.col(group_col).is_in(selected_groups))

        if df_processed.is_empty():
            return no_update, "No data after filters."

        # 4. Process Data (Delegated to Data Utils)
        # Result: { "group_name": (x_centers_array, y_counts_array) }
        group_data = aggregate_histograms_by_group(
            df=df_processed,
            group_col=group_col,
            hist_col=histogram_key,
            min_col=min_key,
            max_col=max_key,
            mode=remap_mode
        )

        # 5. Process Single File Overlay (if selected)
        overlay_data = None
        if selected_file:
            row = df.filter(pl.col("name") == selected_file)
            if row.height > 0:
                c_list = row.get_column(histogram_key).item()
                minv = row.get_column(min_key).item() if min_key in row.columns else 0
                maxv = row.get_column(max_key).item() if max_key in row.columns else 255

                if c_list is not None:
                    c_arr = np.array(c_list, dtype=float)
                    if c_arr.size > 0 and c_arr.sum() > 0:
                        c_arr /= c_arr.sum()
                        _, centers, width = compute_histogram_edges(c_arr, minv, maxv)

                        overlay_data = {
                            "x": centers, "y": c_arr, "width": width, "name": f"File: {selected_file}"
                        }

        # 6. Plot (Delegated to Factory)
        fig = plot_grouped_histogram(
            group_data=group_data,
            color_map=color_map or {},
            overlay_data=overlay_data,
            title=None,  # Clean title, card handles it
        )

        return fig, ""
