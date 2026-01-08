from typing import List, Dict, Set

import numpy as np
import polars as pl
from dash import html, dcc, Input, Output

from PIL import Image

from pixel_patrol_base.report.data_utils import get_sortable_columns
from pixel_patrol_base.report.widget_categories import WidgetCategories
from pixel_patrol_base.report.base_widget import BaseReportWidget
from pixel_patrol_base.report.global_controls import (
    prepare_widget_data,
    GLOBAL_CONFIG_STORE_ID,
    FILTERED_INDICES_STORE_ID,
)
from pixel_patrol_base.report.factory import show_no_data_message, plot_image_mosaic

SPRITE_SIZE = 32


class ImageMosaikWidget(BaseReportWidget):
    NAME: str = "Image Mosaic"
    TAB: str = WidgetCategories.VISUALIZATION.value
    REQUIRES: Set[str] = {"thumbnail", "name"}
    REQUIRES_PATTERNS = None

    SORT_ID = "mosaic-sort-column-dropdown"
    GRAPH_ID = "image-mosaic-graph"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._df: pl.DataFrame | None = None

    @property
    def help_text(self) -> str:
        return (
            "Displays an **image mosaic**, one thumbnail per file.\n\n"
            "- Thumbnails are generated from the central slice in all non-XY dimensions.  \n"
            "- Sorting by a measurement (e.g. mean, min, max) can reveal visual trends.\n"
            "- Border colors indicate the group of each image."
        )

    def get_content_layout(self) -> List:
        return [
            html.Div(
                [
                    html.Label("Sort mosaic by:"),
                    dcc.Dropdown(
                        id=self.SORT_ID,
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
            dcc.Graph(id=self.GRAPH_ID, style={"width": "100%"}),
        ]

    def register(self, app, df: pl.DataFrame):
        self._df = df

        app.callback(
            Output(self.SORT_ID, "options"),
            Output(self.SORT_ID, "value"),
            Input("color-map-store", "data"),
        )(self._set_control_options)

        app.callback(
            Output(self.GRAPH_ID, "figure"),
            Input("color-map-store", "data"),
            Input(self.SORT_ID, "value"),
            Input(FILTERED_INDICES_STORE_ID, "data"),
            Input(GLOBAL_CONFIG_STORE_ID, "data"),
        )(self._update_plot)


    def _set_control_options(self, _color_map: Dict[str, str]):
        sortable = get_sortable_columns(self._df)
        options = [{"label": c, "value": c} for c in sortable]

        default = None
        if "name" in sortable:
            default = "name"
        elif sortable:
            default = sortable[0]

        return options, default


    def _update_plot(
        self,
        color_map: Dict[str, str] | None,
        sort_column: str | None,
        subset_indices: List[int] | None,
        global_config: Dict | None,
    ):

        df_filtered, group_col, _resolved, _warning_msg = prepare_widget_data(
            self._df,
            subset_indices,
            global_config or {},
            metric_base=None,
        )

        cols_needed = {"thumbnail"}
        if group_col: cols_needed.add(group_col)
        if sort_column: cols_needed.add(sort_column)

        df = df_filtered.select([c for c in cols_needed if c in df_filtered.columns])

        if df.height == 0:
            return show_no_data_message()

        if sort_column and sort_column in df.columns:
            df = df.sort(sort_column)

        sprite = _create_sprite_image(
            df,
            color_map = color_map,
            group_col = group_col,
            border = True,
            border_size = 2,
        )

        sprite_np = np.array(sprite)

        unique_groups = (
            df.select(group_col)
            .unique()
            .get_column(group_col)
            .to_list()
        )

        fig = plot_image_mosaic(
            sprite_np,
            unique_groups=unique_groups,
            color_map=color_map or {},
            height=min(1200, sprite.height + 60),
        )
        return fig


def _create_sprite_image(
    df: pl.DataFrame,
    color_map: Dict[str, str],
    group_col: str,
    border: bool = False,
    border_size: int = 0,
):
    """Builds a grid mosaic from df['thumbnail']."""
    if "thumbnail" not in df.columns or df.get_column("thumbnail").is_empty():
        return Image.new("RGBA", (SPRITE_SIZE, SPRITE_SIZE), (0, 0, 0, 0))

    images = df.get_column("thumbnail").to_list()
    groups = df.get_column(group_col).to_list()

    processed = []
    for img_data, group_value in zip(images, groups):
        if img_data is None:
            continue

        color_hex = color_map.get(group_value, "#FFFFFF")
        border_rgb = tuple(int(color_hex[i : i + 2], 16) for i in (1, 3, 5))

        if isinstance(img_data, Image.Image):
            img = img_data.convert("RGB")
        else:
            arr = np.asarray(img_data)
            if arr.dtype in (np.float32, np.float64):
                arr = (arr * 255).astype(np.uint8)
            else:
                arr = arr.astype(np.uint8)
            img = Image.fromarray(arr).convert("RGB")

        resized = img.resize((SPRITE_SIZE, SPRITE_SIZE))

        if border and border_size > 0:
            new_img = Image.new(
                "RGB",
                (SPRITE_SIZE + 2 * border_size, SPRITE_SIZE + 2 * border_size),
                border_rgb,
            )
            new_img.paste(resized, (border_size, border_size))
            processed.append(new_img)
        else:
            processed.append(resized)

    if not processed:
        return Image.new("RGBA", (SPRITE_SIZE, SPRITE_SIZE), (0, 0, 0, 0))

    n = len(processed)
    per_row = int(np.ceil(np.sqrt(n)))
    effective_dim = SPRITE_SIZE + (2 * border_size if border else 0)

    width = per_row * effective_dim
    height = int(np.ceil(n / per_row)) * effective_dim

    sprite = Image.new("RGBA", (width, height), (0, 0, 0, 0))

    for i, img in enumerate(processed):
        r, c = divmod(i, per_row)
        sprite.paste(img.convert("RGBA"), (c * effective_dim, r * effective_dim))

    return sprite