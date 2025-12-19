from typing import List, Dict, Set

import polars as pl
from dash import html, Input, Output

from pixel_patrol_image.plugins.processors.quality_metrics_processor import QualityMetricsProcessor

from pixel_patrol_base.core.feature_schema import patterns_from_processor
from pixel_patrol_base.report.base_widget import BaseReportWidget
from pixel_patrol_base.report.factory import generate_column_violin_plots, show_no_data_message
from pixel_patrol_base.report.widget_categories import WidgetCategories
from pixel_patrol_base.report.data_utils import get_dim_aware_column
from pixel_patrol_base.report.global_controls import (
    prepare_widget_data,
    GLOBAL_CONFIG_STORE_ID,
    FILTERED_INDICES_STORE_ID,
)


class QualityMetricsWidget(BaseReportWidget):
    # ---- Declarative spec ----
    NAME: str = "Image Quality Metrics"
    TAB: str = WidgetCategories.DATASET_STATS.value
    REQUIRES: Set[str] = {"name"}
    # Dynamic metric columns come from the processor (regex patterns).
    REQUIRES_PATTERNS: List[str] = patterns_from_processor(QualityMetricsProcessor)

    CONTENT_ID = "image-quality-container"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._df: pl.DataFrame | None = None

    @property
    def help_text(self) -> str:
        return (
            "Visualizes **image quality metrics** as violin plots across groups.\n\n"
            "Use these plots to quickly spot outliers, compare image sets, and detect quality differences.\n\n"
            "**Metrics**\n"
            "- **Laplacian variance** – Edge-based sharpness estimate. Higher values indicate a sharper image.\n"
            "- **Tenengrad** – Focus measure based on Sobel gradients; captures overall edge strength.\n"
            "- **Brenner** – Measures fine structural detail using pixel intensity differences.\n"
            "- **Noise std** – Estimated pixel-level noise standard deviation; higher noise reduces clarity.\n"
            "- **Blocking records** – Strength of blocky compression artifacts (e.g. JPEG blocking).\n"
            "- **Ringing records** – Edge oscillation artifacts around sharp boundaries, often due to compression.\n\n"
        )

    def get_content_layout(self) -> List:
        return [html.Div(id=self.CONTENT_ID)]


    def register(self, app, df: pl.DataFrame) -> None:
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
            global_config: Dict | None,
    ):

        df_filtered, group_col, _resolved, _warning_msg = prepare_widget_data(
            self._df,
            subset_indices,
            global_config,
            metric_base = None,
        )

        if df_filtered.height == 0:
            return show_no_data_message()

        # Base metric "names" coming from the processor
        base_metric_names = [
            "laplacian_variance",
            "tenengrad",
            "brenner",
            "noise_std",
            "blocking_records",
            "ringing_records",
        ]

        dims_selection = global_config.get("dimensions", {})

        resolved_metric_cols: List[str] = []
        for base in base_metric_names:
            col = get_dim_aware_column(df_filtered.columns, base, dims_selection)
            if col is not None:
                resolved_metric_cols.append(col)

        if df_filtered.is_empty() or not group_col or not resolved_metric_cols:
            return show_no_data_message()

        return generate_column_violin_plots(df_filtered,
                                            color_map,
                                            resolved_metric_cols,
                                            group_col=group_col)


