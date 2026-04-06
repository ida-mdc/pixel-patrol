from typing import List, Dict, Set, Tuple

import numpy as np
import polars as pl
from dash import html, dcc, Input, Output
import logging

from PIL import Image

from pixel_patrol_base.config import SPRITE_SIZE
from pixel_patrol_base.report.data_utils import get_sortable_columns, sort_strings_alpha, select_needed_columns
from pixel_patrol_base.report.widget_categories import WidgetCategories
from pixel_patrol_base.report.base_widget import BaseReportWidget
from pixel_patrol_base.report.global_controls import prepare_widget_data
from pixel_patrol_base.core.report_config import ReportConfig
from pixel_patrol_base.report.constants import GLOBAL_CONFIG_STORE_ID, FILTERED_INDICES_STORE_ID, HOVER_LABEL_COL
from pixel_patrol_base.report.factory import show_no_data_message, plot_image_mosaic

_SPRITE_SIZE = 32
_NORM_COLS = ("thumbnail_norm_min", "thumbnail_norm_max", "thumbnail_dtype")
_DEFAULT_COL = 'mean_intensity'

logger = logging.getLogger(__name__)

class ImageMosaicWidget(BaseReportWidget):
    NAME: str = "image-mosaic"
    TAB: str = WidgetCategories.VISUALIZATION.value
    REQUIRES: Set[str] = {"thumbnail", "name"}
    REQUIRES_PATTERNS = None

    SORT_ID    = "mosaic-sort-column-dropdown"
    GRAPH_ID   = "image-mosaic-graph"
    DISPLAY_ID = "image-mosaic-display-mode"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._df: pl.DataFrame | None = None
        self._sortable_cols_cache: List[str] | None = None

    @property
    def help_text(self) -> str:
        return (
            "Displays an **image mosaic**, one thumbnail per file.\n\n"
            "- Thumbnails are generated from the central slice in all non-XY dimensions.  \n"
            "- Sorting by a measurement (e.g. mean, min, max) can reveal visual trends.\n"
            "- Border colors indicate the group of each image.\n"
            "- **Hover** over an image to see its filename."
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
                        style={"width": "300px", "marginTop": "10px", "marginBottom": "12px"},
                    ),
                    dcc.RadioItems(
                        id=self.DISPLAY_ID,
                        options=[
                            {"label": "Normalised",   "value": "normalized"},
                            {"label": "Denormalised", "value": "denormalized"},
                        ],
                        value="normalized",
                        inline=True,
                        style={"marginBottom": "20px"},
                    ),
                ]
            ),
            dcc.Graph(id=self.GRAPH_ID, style={"width": "100%"}),
        ]

    def register(self, app, df: pl.DataFrame):
        self._df = df
        self._sortable_cols_cache = sort_strings_alpha(get_sortable_columns(df))

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
            Input(self.DISPLAY_ID, "value"),
        )(self._update_plot)

    def _set_control_options(self, _color_map: Dict[str, str]):
        sortable = self._sortable_cols_cache or []
        options = [{"label": c, "value": c} for c in sortable]

        default = None
        if _DEFAULT_COL in sortable:
            default = _DEFAULT_COL
        elif sortable:
            default = sortable[0]

        return options, default

    def _update_plot(
        self,
        color_map: Dict[str, str] | None,
        sort_column: str | None,
        subset_indices: List[int] | None,
        global_config_dict: Dict | None,
        display_mode: str = "normalized",
    ):
        report_config = ReportConfig.from_dict(global_config_dict) if global_config_dict else None
        df_filtered, group_col, _resolved, _warning_msg, _order = prepare_widget_data(
            self._df,
            subset_indices,
            report_config,
            metric_base=None,
        )

        cols_needed = ["thumbnail", HOVER_LABEL_COL]
        # Include norm columns when available (needed for denormalized display)
        norm_cols_present = [c for c in _NORM_COLS if c in df_filtered.columns]
        cols_needed += norm_cols_present

        extra = [group_col] if group_col else []
        if sort_column:
            extra.append(sort_column)

        df = select_needed_columns(df_filtered, cols_needed, extra_cols=extra)

        if df.height == 0:
            return show_no_data_message()

        if sort_column and sort_column in df.columns:
            df = df.sort(sort_column)

        sprite, hover_info = _create_sprite_image(
            df,
            color_map=color_map,
            group_col=group_col,
            border=True,
            border_size=2,
            display_mode=display_mode if norm_cols_present else "normalized",
        )

        sprite_np = np.array(sprite)

        unique_groups = df.get_column(group_col).unique().to_list()

        fig = plot_image_mosaic(
            sprite_np,
            unique_groups=unique_groups,
            color_map=color_map or {},
            height=min(1200, sprite.height + 60),
            hover_info=hover_info,  # NEW parameter
        )
        return fig


def _hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """Convert hex color to RGB tuple."""
    return (
        int(hex_color[1:3], 16),
        int(hex_color[3:5], 16),
        int(hex_color[5:7], 16),
    )


def _denormalize(arr: np.ndarray, norm_min: float, norm_max: float, dtype_name: str) -> np.ndarray:
    """Undo normalization from [0, 255] using `norm_min`/`norm_max`, then map to uint8 display values for the target integer dtype."""
    try:
        info = np.iinfo(np.dtype(dtype_name))
        dtype_lo, dtype_hi = float(info.min), float(info.max)
    except (ValueError, TypeError):
        return arr  # float or unknown dtype — nothing to do
    scale = (norm_max - norm_min) / 255.0
    display = (arr.astype(np.float32) * scale + norm_min - dtype_lo) / (dtype_hi - dtype_lo) * 255.0
    return np.clip(display, 0, 255).astype(np.uint8)


def _create_sprite_image(
    df: pl.DataFrame,
    color_map: Dict[str, str],
    group_col: str,
    border: bool = False,
    border_size: int = 0,
    display_mode: str = "normalized",
) -> Tuple[Image.Image, Dict]:
    """
    Builds a grid mosaic from df['thumbnail'].

    Returns:
        (sprite_image, hover_info) where hover_info contains x, y coords and names for hover
    """
    effective_dim = _SPRITE_SIZE + (2 * border_size if border else 0)

    if "thumbnail" not in df.columns or df.get_column("thumbnail").is_empty():
        return Image.new("RGBA", (_SPRITE_SIZE, _SPRITE_SIZE), (0, 0, 0, 0)), {}

    # Extract columns once as lists
    images = df.get_column("thumbnail").to_list()
    groups = df.get_column(group_col).to_list()
    hover_labels = df.get_column(HOVER_LABEL_COL).to_list()

    has_norm = all(c in df.columns for c in _NORM_COLS)
    norm_mins   = df.get_column("thumbnail_norm_min").to_list()   if has_norm else None
    norm_maxs   = df.get_column("thumbnail_norm_max").to_list()   if has_norm else None
    dtype_names = df.get_column("thumbnail_dtype").to_list()      if has_norm else None

    # Pre-compute RGB colors for all unique groups
    color_map = color_map or {}
    color_rgb_cache: Dict[str, Tuple[int, int, int]] = {
        group_value: _hex_to_rgb(color_map.get(group_value, "#FFFFFF"))
        for group_value in set(groups)
    }

    # Process images and collect hover data
    processed = []
    processed_hover_labels = []

    for i, (img_data, group_value, hover_label) in enumerate(zip(images, groups, hover_labels)):
        if img_data is None:
            continue

        border_rgb = color_rgb_cache[group_value]

        arr = np.frombuffer(bytes(img_data), dtype=np.uint8).reshape(SPRITE_SIZE, SPRITE_SIZE, 4).copy()
        if display_mode == "denormalized" and has_norm:
            arr[:, :, :3] = _denormalize(arr[:, :, :3], norm_mins[i], norm_maxs[i], dtype_names[i])
        img = Image.fromarray(arr, mode="RGBA")

        resized = img.resize((_SPRITE_SIZE, _SPRITE_SIZE))

        if border and border_size > 0:
            new_img = Image.new("RGBA", (effective_dim, effective_dim), border_rgb + (255,))
            # Punch out the interior so only the outer ring keeps the border color
            new_img.paste(Image.new("RGBA", (_SPRITE_SIZE, _SPRITE_SIZE), (0, 0, 0, 0)), (border_size, border_size))
            new_img.paste(resized, (border_size, border_size), mask=resized.split()[3])
            processed.append(new_img)
        else:
            processed.append(resized)

        processed_hover_labels.append(hover_label)


    if not processed:
        return Image.new("RGBA", (_SPRITE_SIZE, _SPRITE_SIZE), (0, 0, 0, 0)), {}

    n = len(processed)
    per_row = int(np.ceil(np.sqrt(n)))

    width = per_row * effective_dim
    height = int(np.ceil(n / per_row)) * effective_dim

    # Create sprite using numpy array directly
    sprite_arr = np.zeros((height, width, 4), dtype=np.uint8)

    # Build hover coordinates while placing images
    hover_x = []
    hover_y = []
    half_dim = effective_dim / 2

    for i, img in enumerate(processed):
        row, col = divmod(i, per_row)
        y_start = row * effective_dim
        x_start = col * effective_dim

        # Place image in sprite
        sprite_arr[y_start:y_start + effective_dim, x_start:x_start + effective_dim] = np.array(img)

        # Store center coordinates for hover (in image pixel coordinates)
        hover_x.append(x_start + half_dim)
        hover_y.append(y_start + half_dim)

    hover_info = {
        "x": hover_x,
        "y": hover_y,
        "names": hover_labels,
        "marker_size": effective_dim,
    }

    return Image.fromarray(sprite_arr, mode="RGBA"), hover_info