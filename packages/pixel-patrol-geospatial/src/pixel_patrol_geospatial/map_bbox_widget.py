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

from pixel_patrol_geospatial.map_centroid_widget import (
    _show_no_data_message,
    _calculate_center_zoom,
)


def _get_lat_lon_from_bboxes(wests: List, easts: List, norths: List, souths: List, mask_idx: List[int]) -> Tuple[List, List]:
    """plotly/MapLibre expects that lines are lat/lon lists. to add a gap, add a None. so each bounds will be four corners + None. last point = first point."""
    lats, lons = list(), list()
    selected_easts = [easts[i] for i in mask_idx]
    selected_wests = [wests[i] for i in mask_idx]
    selected_souths = [souths[i] for i in mask_idx]
    selected_norths = [norths[i] for i in mask_idx]

    for east, west, south, north in zip(selected_easts, selected_wests, selected_souths, selected_norths):
        lats += [north, north, south, south, north, None]
        lons += [east,  west,  west,  east,  east,  None]
    return lats, lons


class BBoxGeoMapWidget(BaseReportWidget):
    NAME: str = "Geolocation Map Bounding Box"
    TAB: str = WidgetCategories.VISUALIZATION.value

    _BBOX_REQUIRES = {"bbox_west", "bbox_east", "bbox_north", "bbox_south"}
    REQUIRES: Set[str] = _BBOX_REQUIRES | {"name"}
    REQUIRES_PATTERNS = None

    CONTENT_ID = "bbox-geo-map-content"
    GRAPH_ID = "bbox-geo-map-graph"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._df: pl.DataFrame | None = None

    @property
    def help_text(self) -> str:
        return (
            "Plots the bounding box of each row as a rectangle using.\n\n"
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

        needed = self.REQUIRES
        if group_col:
            needed.add(group_col)

        if  not self._BBOX_REQUIRES.issubset(df_filtered.columns):
            return _show_no_data_message(f"Missing `{'|'.join(self._BBOX_REQUIRES)}` columns.")

        cols = [c for c in needed if c in df_filtered.columns]
        df = df_filtered.select(cols).drop_nulls(self._BBOX_REQUIRES)
        if df.height == 0:
            return _show_no_data_message()

        # Pull into plain Python lists (NO pandas)
        d: Dict[str, List[Any]] = df.to_dict(as_series=False)
        bbox_norths = d.get("bbox_north", [])
        bbox_souths = d.get("bbox_south", [])
        bbox_easts = d.get("bbox_east", [])
        bbox_wests = d.get("bbox_west", [])
        names = d.get("name", [""] * len(bbox_norths))
        groups = d.get(group_col, [None] * len(bbox_norths)) if group_col else [None] * len(bbox_norths)

        style_value = "open-street-map"  # which base layer of the map

        fig = go.Figure()

        def add_lines_to_map_as_trace(mask_idx: List[int], label: str | None, color: str | None):
            # adds markers to the map, either for all values or for a single group (using mask_idx)
            lat_vals, lon_vals = _get_lat_lon_from_bboxes(wests=bbox_wests,
                                                          easts=bbox_easts,
                                                          norths=bbox_norths,
                                                          souths=bbox_souths,
                                                          mask_idx=mask_idx,
                                                          )
            name_vals = []
            for i in mask_idx:
                # each BBox has 5 points and the None to separate them.
                name_vals += [names[i]] * 5 + [None]

            marker = dict(size=8, opacity=0.80)
            if color:
                marker["color"] = color

            fig.add_trace(
                go.Scattermap(
                    lat=lat_vals,
                    lon=lon_vals,
                    mode="lines",
                    name=label if label is not None else "",
                    #marker=marker,
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
                add_lines_to_map_as_trace(idxs, str(g), cmap.get(str(g)))
        else:
            add_lines_to_map_as_trace(list(range(len(bbox_easts))), None, None)


        # Layout
        center, zoom = _calculate_center_zoom(latitudes=bbox_souths + bbox_norths,
                                              longitudes=bbox_wests + bbox_easts)
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