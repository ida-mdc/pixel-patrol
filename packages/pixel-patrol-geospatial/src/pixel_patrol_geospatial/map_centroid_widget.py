from __future__ import annotations

from typing import List, Set, Dict, Any, Tuple
import math

import polars as pl
import plotly.graph_objects as go
from dash import html, dcc, Input, Output

from pixel_patrol_base.report.base_widget import BaseReportWidget
from pixel_patrol_base.report.widget_categories import WidgetCategories
from pixel_patrol_base.report.global_controls import (
    prepare_widget_data,
    GLOBAL_CONFIG_STORE_ID,
)
from pixel_patrol_base.core.report_config import ReportConfig
from pixel_patrol_base.report.constants import FILTERED_INDICES_STORE_ID


def _show_no_data_message(text: str = "No data available after filtering.") -> html.Div:
    return html.Div(
        text,
        className="text-warning p-3",
        style={"textAlign": "center"},
    )

def _calculate_center_zoom(latitudes: List[float], longitudes: List[float]) -> Tuple[Dict[str, float], int]:
    """Calculate center and zoom given a Sequence of latitude and longitude values.

    The center is in the middle between highest and lowest lat/lon value.

    Zoom level is calculated with the range of lat/lon, which is inspired from
    ggmap's calc_zoom method: https://rdrr.io/cran/ggmap/src/R/calc_zoom.r
    """
    lon_range = max(longitudes) - min(longitudes)
    lat_range = max(latitudes) - min(latitudes)
    # clipping for safety, if lat/lon values > +-360/180 appear
    lon_range = min(180, lon_range)
    lat_range = min(360, lat_range)

    center_lat = min(latitudes) + (lat_range / 2)
    center_lon = min(longitudes) + (lon_range / 2)
    center = dict(lat=center_lat, lon=center_lon)

    lon_zoom = math.log2(360 * 2 / lon_range) # z=0 is 360°
    lat_zoom = math.log2(180 * 2 / lat_range)
    zoom = math.floor(max(lon_zoom, lat_zoom))
    # manually decrease zoom level to be more certain that most points are in the default zoom level.
    # still this is not guarenteed right now
    zoom -= 1

    return center, zoom


class CentroidGeoMapWidget(BaseReportWidget):
    NAME: str = "Geolocation Map"
    TAB: str = WidgetCategories.VISUALIZATION.value

    REQUIRES: Set[str] = {"latitude", "longitude", "name"}
    REQUIRES_PATTERNS = None

    CONTENT_ID = "centroid-geo-map-content"
    GRAPH_ID = "centroid-geo-map-graph"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._df: pl.DataFrame | None = None

    @property
    def help_text(self) -> str:
        return (
            "Plots each row as a dot using `latitude` / `longitude`.\n\n"
            "- Respects global filtering.\n"
            "- Colors dots by your active grouping (same as other widgets).\n"
        )

    def get_content_layout(self) -> List:
        return [
                html.Div(
                    dcc.Graph(id=self.GRAPH_ID, style={"height": "700px", "width": "100%"}),
                    id=self.CONTENT_ID,
                ),
        ]

    def register(self, app, df: pl.DataFrame):
        self._df = df

        app.callback(
            Output(self.CONTENT_ID, "children"),
            Input("color-map-store", "data"),
            Input(FILTERED_INDICES_STORE_ID, "data"),
            Input(GLOBAL_CONFIG_STORE_ID, "data"),
        )(self._update_plot)

    def _update_plot(
        self,
        color_map: Dict[str, str] | None,
        subset_indices: List[int] | None,
        global_config_dict: dict | None,
    ):
        report_config = ReportConfig.from_dict(global_config_dict) if global_config_dict else None
        df_filtered, group_col, _resolved, _warning_msg, group_order = prepare_widget_data(
            self._df,
            subset_indices,
            report_config,
            metric_base=None,
        )

        needed = {"latitude", "longitude", "name"}
        if group_col:
            needed.add(group_col)

        if "latitude" not in df_filtered.columns or "longitude" not in df_filtered.columns:
            return _show_no_data_message("Missing `latitude` / `longitude` columns.")

        cols = [c for c in needed if c in df_filtered.columns]
        df = df_filtered.select(cols).drop_nulls(["latitude", "longitude"])
        if df.height == 0:
            return _show_no_data_message()

        # Pull into plain Python lists (NO pandas)
        d: Dict[str, List[Any]] = df.to_dict(as_series=False)
        lats = d.get("latitude", [])
        lons = d.get("longitude", [])
        names = d.get("name", [""] * len(lats))
        groups = d.get(group_col, [None] * len(lats)) if group_col else [None] * len(lats)

        style_value = "open-street-map"  # which base layer of the map

        fig = go.Figure()

        def add_markers_to_map_as_trace(mask_idx: List[int], label: str | None, color: str | None):
            # adds markers to the map, either for all values or for a single group (using mask_idx)
            lat_vals = [lats[i] for i in mask_idx]
            lon_vals = [lons[i] for i in mask_idx]
            name_vals = [names[i] for i in mask_idx]

            marker = dict(size=8, opacity=0.80)
            if color:
                marker["color"] = color

            fig.add_trace(
                go.Scattermap(
                    lat=lat_vals,
                    lon=lon_vals,
                    mode="markers",
                    name=label if label is not None else "",
                    marker=marker,
                    text=name_vals,
                    hovertemplate="%{text}<br>lat=%{lat:.4f}, lon=%{lon:.4f}<extra></extra>",
                    showlegend=label is not None,
                )
            )

        if group_col:
            # stable ordering if provided
            unique_groups = list(dict.fromkeys(groups))
            if group_order:
                order = [g for g in group_order if g in set(unique_groups)]
                tail = [g for g in unique_groups if g not in set(order)]
                unique_groups = order + tail

            cmap = color_map or {}
            for g in unique_groups:
                idxs = [i for i, gv in enumerate(groups) if gv == g]
                add_markers_to_map_as_trace(idxs, str(g), cmap.get(str(g)))
        else:
            add_markers_to_map_as_trace(list(range(len(lats))), None, None)


        # Layout
        center, zoom = _calculate_center_zoom(latitudes=lats, longitudes=lons)
        fig.update_layout(
            margin=dict(l=20, r=20, t=20, b=20),
            legend=dict(orientation="h", yanchor="top", y=-0.08, xanchor="center", x=0.5),
            height=700,
            map=dict(
                style=style_value,
                center=center,
                zoom=zoom,
            )
        )
        return dcc.Graph(figure=fig, id=self.GRAPH_ID)