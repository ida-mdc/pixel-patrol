from typing import List, Dict, Set

import polars as pl
from dash import html, Input, Output

from pixel_patrol_image.plugins.processors.quality_metrics_processor import QualityMetricsProcessor
from pixel_patrol_base.core.feature_schema import patterns_from_processor
from pixel_patrol_base.report.utils import generate_column_violin_plots
from pixel_patrol_base.report.widget_categories import WidgetCategories


class ImageQualityWidget:
    # ---- Declarative spec ----
    NAME: str = "Image Quality"
    TAB: str = WidgetCategories.DATASET_STATS.value
    # Grouping by folder is typical for these plots; require the label column.
    REQUIRES: Set[str] = {"imported_path_short"}
    # Dynamic metric columns come from the processor (regex patterns).
    REQUIRES_PATTERNS: List[str] = patterns_from_processor(QualityMetricsProcessor)

    # Component IDs
    CONTAINER_ID = "image-quality-container"

    def get_descriptions(self) -> Dict[str, str]:
        """Descriptions of image quality metrics shown below."""
        return {
            "laplacian_variance": (
                "Measures sharpness via the variance of the Laplacian (edges). "
                "Higher values indicate sharper images."
            ),
            "tenengrad": (
                "Edge strength from Sobel gradients. Higher values often mean better focus."
            ),
            "brenner": (
                "Intensity differences between neighboring pixels; higher suggests more fine detail."
            ),
            "noise_std": (
                "Estimated random noise level; higher noise reduces clarity."
            ),
            # "wavelet_energy": "High-frequency detail via wavelet energy.",
            "blocking_records": (
                "Compression blockiness (e.g., JPEG); higher indicates stronger blocking records."
            ),
            "ringing_records": (
                "Edge ghosting/oscillations from compression; higher indicates stronger ringing."
            ),
        }

    def layout(self) -> List:
        """Static description + a single container populated by the callback."""
        description_items = [
            html.Li([html.Strong(f"{k.replace('_', ' ').title()}: "), v])
            for k, v in self.get_descriptions().items()
        ]
        return [
            html.Div(
                className="markdown-content",
                children=[
                    html.H4("Image Quality Metric Descriptions"),
                    html.P(
                        "The following metrics assess various aspects of image quality, such as sharpness, noise, "
                        "and compression records. Each plot shows the distribution of a quality score for images "
                        "in the selected folders."
                    ),
                    html.Ul(description_items),
                ],
            ),
            html.Hr(),
            html.Div(id=self.CONTAINER_ID),
        ]

    def register(self, app, df_global: pl.DataFrame):
        """One callback that renders all violin plots."""

        @app.callback(
            Output(self.CONTAINER_ID, "children"),
            Input("color-map-store", "data"),
        )
        def update_image_quality_layout(color_map: Dict[str, str]):
            metric_cols = list(self.get_descriptions().keys())
            return generate_column_violin_plots(df_global, color_map or {}, metric_cols)
