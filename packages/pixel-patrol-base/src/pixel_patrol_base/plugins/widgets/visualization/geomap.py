from __future__ import annotations

from typing import List, Set, Dict, Any

import polars as pl
import plotly.graph_objects as go
from dash import html, dcc, Input, Output

from pixel_patrol_base.report.base_widget import BaseReportWidget
from pixel_patrol_base.report.widget_categories import WidgetCategories
from pixel_patrol_base.report.global_controls import (
    prepare_widget_data,
    GLOBAL_CONFIG_STORE_ID,
    FILTERED_INDICES_STORE_ID,
)


def _show_no_data_message(text: str = "No data available after filtering.") -> html.Div:
    return html.Div(
        text,
        className="text-warning p-3",
        style={"textAlign": "center"},
    )


class GeoMapWidget(BaseReportWidget):
    NAME: str = "Geolocation Map"
    TAB: str = WidgetCategories.VISUALIZATION.value

    REQUIRES: Set[str] = {"latitude", "longitude", "name"}
    REQUIRES_PATTERNS = None

    CONTENT_ID = "geo-map-content"
    GRAPH_ID = "geo-map-graph"
    MAP_STYLE_DROPDOWN_ID = "geo-map-style"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._df: pl.DataFrame | None = None

    @property
    def help_text(self) -> str:
        return (
            "Plots each row as a dot using `latitude` / `longitude`.\n\n"
            "- Respects global filtering.\n"
            "- Colors dots by your active grouping (same as other widgets).\n"
            "- Map style selectable (geo vs tiles)."
        )

    def get_content_layout(self) -> List:
        return [
            html.Div(
                [
                    dcc.Dropdown(
                        id=self.MAP_STYLE_DROPDOWN_ID,
                        options=[
                            {"label": "OpenStreetMap tiles", "value": "mapbox:open-street-map"},
                            {"label": "Carto Positron tiles", "value": "mapbox:carto-positron"},
                            {"label": "Natural Earth (Plotly)", "value": "geo:natural earth"},
                            {"label": "Orthographic Globe (Plotly)", "value": "geo:orthographic"},
                        ],
                        value="mapbox:open-street-map",
                        clearable=False,
                        style={"maxWidth": "420px"},
                    ),
                    html.Div(
                        dcc.Graph(id=self.GRAPH_ID, style={"height": "700px", "width": "100%"}),
                        id=self.CONTENT_ID,
                    ),
                ]
            )
        ]

    def register(self, app, df: pl.DataFrame):
        self._df = df

        app.callback(
            Output(self.CONTENT_ID, "children"),
            Input("color-map-store", "data"),
            Input(FILTERED_INDICES_STORE_ID, "data"),
            Input(GLOBAL_CONFIG_STORE_ID, "data"),
            Input(self.MAP_STYLE_DROPDOWN_ID, "value"),
        )(self._update_plot)

    def _update_plot(
        self,
        color_map: Dict[str, str] | None,
        subset_indices: List[int] | None,
        global_config: Dict | None,
        map_style: str | None,
    ):
        df_filtered, group_col, _resolved, _warning_msg, group_order = prepare_widget_data(
            self._df,
            subset_indices,
            global_config or {},
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

        map_style = map_style or "mapbox:open-street-map"

        is_mapbox = map_style.startswith("mapbox:")
        style_value = map_style.split(":", 1)[1]

        fig = go.Figure()

        def add_trace(mask_idx: List[int], label: str | None, color: str | None):
            lat_vals = [lats[i] for i in mask_idx]
            lon_vals = [lons[i] for i in mask_idx]
            name_vals = [names[i] for i in mask_idx]

            marker = dict(size=8, opacity=0.80)
            if color:
                marker["color"] = color

            if is_mapbox:
                fig.add_trace(
                    go.Scattermapbox(
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
            else:
                fig.add_trace(
                    go.Scattergeo(
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
                add_trace(idxs, str(g), cmap.get(str(g)))
        else:
            add_trace(list(range(len(lats))), None, None)

        # Layout
        fig.update_layout(
            margin=dict(l=20, r=20, t=20, b=20),
            legend=dict(orientation="h", yanchor="top", y=-0.08, xanchor="center", x=0.5),
            height=700,
        )

        if is_mapbox:
            fig.update_layout(
                mapbox=dict(
                    style=style_value,
                    center=dict(lat=0, lon=0),
                    zoom=0,
                )
            )
        else:
            fig.update_geos(
                projection_type=style_value,
                showland=True,
                showcountries=True,
                showocean=True,
                lataxis_showgrid=True,
                lonaxis_showgrid=True,
            )

        return dcc.Graph(figure=fig, id=self.GRAPH_ID)
