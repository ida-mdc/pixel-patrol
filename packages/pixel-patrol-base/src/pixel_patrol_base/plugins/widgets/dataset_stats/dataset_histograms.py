from typing import List, Set, Dict
import polars as pl
import numpy as np
from dash import html, dcc, Input, Output, no_update

from pixel_patrol_base.report.widget_categories import WidgetCategories
from pixel_patrol_base.report.base_widget import BaseReportWidget
from pixel_patrol_base.report.factory import plot_grouped_histogram, show_no_data_message
from pixel_patrol_base.report.data_utils import (
    aggregate_histograms_by_group,
    compute_histogram_edges,
    format_selection_title,
    sort_strings_alpha,
    select_needed_columns
)
from pixel_patrol_base.report.global_controls import prepare_widget_data
from pixel_patrol_base.report.constants import (FILTERED_INDICES_STORE_ID,
                                                GLOBAL_CONFIG_STORE_ID,
                                                GC_DIMENSIONS,
                                                MAX_RECORDS_IN_MENU)


class DatasetHistogramWidget(BaseReportWidget):
    NAME: str = "Pixel Value Histograms"
    TAB: str = WidgetCategories.DATASET_STATS.value
    REQUIRES: Set[str] = {"name"}
    REQUIRES_PATTERNS: List[str] = [r"^histogram"]
    CONTENT_ID = "histogram-content-container"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._df: pl.DataFrame | None = None

        # Avoids re-filtering the same DataFrame when multiple callbacks fire
        self._prepare_cache_key: tuple | None = None
        self._prepare_cache_result: tuple | None = None

        # Avoids re-aggregating when only the overlay file changes
        self._agg_cache_key: tuple | None = None
        self._agg_cache_result: dict | None = None

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
            # --- Controls Section ---
            html.Div([
                # 1. Mode
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

                # 2. Group Selection
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

                # 3. File Overlay (Moved to Top as requested)
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
            html.Div(id=self.CONTENT_ID, style={"minHeight": "400px"}),
        ]

    def register(self, app, df: pl.DataFrame):
        self._df = df

        # 1. Populate basic controls (based on globally filtered rows)
        app.callback(
            Output("histogram-group-dropdown", "options"),
            Output("histogram-group-dropdown", "value"),
            Input(FILTERED_INDICES_STORE_ID, "data"),
            Input(GLOBAL_CONFIG_STORE_ID, "data"),
        )(self._set_control_options)

        # 2. Populate file dropdown based on selected groups
        app.callback(
            Output("histogram-file-dropdown", "options"),
            Output("histogram-file-dropdown", "value"),
            Input("histogram-group-dropdown", "value"),
            Input(GLOBAL_CONFIG_STORE_ID, "data"),
            Input(FILTERED_INDICES_STORE_ID, "data"),
            prevent_initial_call=False,
        )(self._update_file_options)

        # 3. Main Plot Update
        app.callback(
            Output(self.CONTENT_ID, "children"),
            Input("color-map-store", "data"),
            Input("histogram-remap-mode", "value"),
            Input("histogram-group-dropdown", "value"),
            Input("histogram-file-dropdown", "value"),
            Input(FILTERED_INDICES_STORE_ID, "data"),
            Input(GLOBAL_CONFIG_STORE_ID, "data"),
        )(self._update_plot)


    def _set_control_options(self, subset_indices, global_config):

        df_processed, group_col, _resolved, _warning, _order = self._get_prepared_data(
            subset_indices,
            global_config,
            metric_base="histogram_counts",  # Consistent with other callbacks for cache hits
        )

        if not group_col or df_processed.is_empty():
            return [], []

        group_values = sort_strings_alpha(df_processed[group_col].unique().sort().to_list())
        group_options = [{"label": str(g), "value": g} for g in group_values]

        return group_options, []

    def _update_file_options(self, selected_groups, global_config, subset_indices):

        df_processed, group_col, _resolved, _warning, _order = self._get_prepared_data(
            subset_indices,
            global_config,
            metric_base="histogram_counts",  # Consistent with other callbacks for cache hits
        )

        if selected_groups:
            df_processed = df_processed.filter(pl.col(group_col).is_in(selected_groups))

        if df_processed.is_empty():
            return [], None

        # Limit to MAX_RECORDS_IN_MENU files for performance in dropdown
        names_all = df_processed["name"].unique().to_list()
        limited = len(names_all) > MAX_RECORDS_IN_MENU

        names = sort_strings_alpha(names_all[:MAX_RECORDS_IN_MENU])

        options = [{"label": n, "value": n} for n in names]

        if limited:
            options.insert(
                0,
                {
                    "label": f"⚠ showing first {MAX_RECORDS_IN_MENU} files",
                    "value": "__LIMIT_WARNING__",
                    "disabled": True,
                },
            )

        return options, no_update

    def _update_plot(
            self,
            color_map: Dict[str, str],
            remap_mode: str | None,
            selected_groups: List[str] | None,
            selected_file: str | None,
            subset_indices: List[int] | None,
            global_config: Dict,
    ):
        metric_base = "histogram_counts"

        df_filtered, group_col, resolved_col, warning_msg, group_order = self._get_prepared_data(
            subset_indices,
            global_config,
            metric_base=metric_base,
        )

        if resolved_col is None or df_filtered.is_empty():
            return show_no_data_message()

        suffix = resolved_col.replace(metric_base, "")
        min_key = f"histogram_min{suffix}"
        max_key = f"histogram_max{suffix}"

        cols_needed = [resolved_col, min_key, max_key, "name"]
        extra = [group_col] if group_col else []
        df_filtered = select_needed_columns(df_filtered, cols_needed, extra_cols=extra)

        # Apply optional (within widget) group selection
        if selected_groups:
            df_filtered = df_filtered.filter(pl.col(group_col).is_in(selected_groups))

        # Excludes selected_file because overlay is computed separately and is cheap
        # This way, changing only the overlay file won't re-aggregate all histograms
        agg_cache_key = (
            tuple(subset_indices) if subset_indices else None,
            tuple(sorted((global_config or {}).items())),
            remap_mode,
            tuple(sorted(selected_groups)) if selected_groups else None,
        )

        # Use cached aggregation if available, otherwise compute and cache
        if self._agg_cache_key == agg_cache_key and self._agg_cache_result is not None:
            group_data = self._agg_cache_result
        else:
            # This is the expensive operation - aggregate per-group histograms
            group_data = aggregate_histograms_by_group(
                df=df_filtered,
                group_col=group_col,
                hist_col=resolved_col,
                min_col=min_key,
                max_col=max_key,
                mode=remap_mode,
            )
            self._agg_cache_key = agg_cache_key
            self._agg_cache_result = group_data

        if not group_data:
            return show_no_data_message()

        # Optional single-file overlay (cheap operation, always recompute)
        overlay_data = get_overly_of_single_row(df_filtered, max_key, min_key, resolved_col, selected_file)

        dims_selection = (global_config or {}).get(GC_DIMENSIONS, {})

        fig = plot_grouped_histogram(
            group_data=group_data,
            color_map=color_map or {},
            overlay_data=overlay_data,
            title=format_selection_title(dims_selection),
            group_order=group_order,
        )

        return dcc.Graph(figure=fig, style={"height": "600px"})


    def _get_prepared_data(self, subset_indices, global_config, metric_base="histogram_counts"):
        """
        Cached wrapper around prepare_widget_data.

        When user changes a global control, multiple callbacks fire with the same
        inputs. This cache ensures we only filter the DataFrame once.

        Cache is invalidated automatically when any input changes.
        """
        # Build a hashable cache key from all inputs
        cache_key = (
            tuple(subset_indices) if subset_indices else None,
            tuple(sorted((global_config or {}).items())),
            metric_base,
        )

        # Return cached result if inputs haven't changed
        if self._prepare_cache_key == cache_key and self._prepare_cache_result is not None:
            return self._prepare_cache_result

        # Compute fresh result and cache it
        result = prepare_widget_data(
            self._df,
            subset_indices,
            global_config or {},
            metric_base=metric_base,
        )

        self._prepare_cache_key = cache_key
        self._prepare_cache_result = result
        return result


def get_overly_of_single_row(df_filtered, max_key, min_key, resolved_col, selected_file):
    overlay_data = None
    if selected_file:
        row = df_filtered.filter(pl.col("name") == selected_file)
        if row.height > 0:
            c_list = row.get_column(resolved_col).item()
            minv = row.get_column(min_key).item() if min_key in row.columns else 0
            maxv = row.get_column(max_key).item() if max_key in row.columns else 255

            if c_list is not None:
                c_arr = np.array(c_list, dtype=float)
                if c_arr.size > 0 and c_arr.sum() > 0:
                    c_arr /= c_arr.sum()
                    _, centers, width = compute_histogram_edges(c_arr, minv, maxv)

                    overlay_data = {
                        "x": centers,
                        "y": c_arr,
                        "width": width,
                        "name": f"File: {selected_file}",
                    }
    return overlay_data