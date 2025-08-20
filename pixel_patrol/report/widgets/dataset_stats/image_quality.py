from typing import List, Dict

import polars as pl
from dash import html, Input, Output

from pixel_patrol.core.processors.quality_metrics_processor import QualityMetricsProcessor
from pixel_patrol.core.spec_provider import get_requirements_as_patterns
from pixel_patrol.report.utils import generate_column_violin_plots
from pixel_patrol.report.widget_categories import WidgetCategories
from pixel_patrol.report.widget_interface import PixelPatrolWidget


class ImageQualityWidget(PixelPatrolWidget):
    @property
    def tab(self) -> str:
        return WidgetCategories.DATASET_STATS.value

    @property
    def name(self) -> str:
        return "Image Quality"

    def required_columns(self) -> List[str]:
        return get_requirements_as_patterns(QualityMetricsProcessor())

    def get_descriptions(self) -> Dict[str, str]:
        """Returns a dictionary of image quality metrics and their descriptions."""
        return {
            "laplacian_variance": (
                "Measures the sharpness of an image by calculating the variance of the Laplacian. "
                "The Laplacian operator highlights regions of rapid intensity change, such as edges. "
                "A higher value indicates a sharper image with more pronounced edges, while a lower value suggests a blurrier image."
            ),
            "tenengrad": (
                "Reflects the strength of edges in an image by computing the gradient magnitude using the Sobel operator. "
                "Stronger edges typically indicate a clearer and more detailed image. "
                "This metric is often used to assess image focus and sharpness."
            ),
            "brenner": (
                "Captures the level of detail in an image by measuring intensity differences between neighboring pixels. "
                "A higher Brenner score indicates more fine details and textures, while a lower score suggests a smoother or blurrier image. "
                "This metric is particularly useful for evaluating image focus."
            ),
            "noise_std": (
                "Estimates the level of random noise present in an image. "
                "Noise can appear as graininess or speckles and is often caused by low light conditions or sensor limitations. "
                "A higher noise level can reduce image clarity and make it harder to distinguish fine details."
            ),
            # "wavelet_energy": (
            #     "Summarizes the amount of high-frequency detail in an image using wavelet transforms. "
            #     "Wavelets decompose an image into different frequency components, and the energy in the high-frequency bands reflects fine details and textures. "
            #     "A higher wavelet energy indicates more intricate details, while a lower value suggests a smoother image."
            # ),
            "blocking_artifacts": (
                "Detects compression artifacts known as 'blocking,' which occur when an image is heavily compressed (e.g., in JPEG format). "
                "Blocking artifacts appear as visible 8x8 pixel blocks, especially in smooth or gradient regions. "
                "A higher score indicates more severe blocking artifacts, which can degrade image quality."
            ),
            "ringing_artifacts": (
                "Identifies compression artifacts known as 'ringing,' which appear as ghosting or oscillations near sharp edges. "
                "Ringing artifacts are common in compressed images and can make edges look blurry or distorted. "
                "A higher score indicates more pronounced ringing artifacts, which can reduce image clarity."
            ),
        }

    def layout(self) -> List:
        """Defines the new layout with metric descriptions and a container for all plots."""
        # Create a list of descriptions from the get_descriptions method
        description_items = [
            html.Li([
                html.Strong(f"{key.replace('_', ' ').title()}: "),
                value
            ]) for key, value in self.get_descriptions().items()
        ]

        return [
            # Static description section
            html.Div(className="markdown-content", children=[
                html.H4("Image Quality Metric Descriptions"),
                html.P(
                    "The following metrics assess various aspects of image quality, such as sharpness, noise, and compression artifacts. "
                    "Each plot below shows the distribution of a specific quality score for all images in the selected folders. "
                    "Statistical annotations indicate significant differences between groups."
                ),
                html.Ul(description_items),
            ]),
            # A horizontal line for visual separation
            html.Hr(),
            # This single container will be populated by the callback with all plots
            html.Div(id="image-quality-container"),
        ]

    def register_callbacks(self, app, df_global: pl.DataFrame):
        """Registers a single callback to generate all quality metric plots."""

        @app.callback(
            Output("image-quality-container", "children"),
            Input("color-map-store", "data")
        )
        def update_image_quality_layout(color_map: Dict[str, str]):
            """Generates violin plots for all defined quality metrics."""
            # Get the list of columns to plot from the description keys
            quality_metric_cols = list(self.get_descriptions().keys())

            # Use the utility function to generate and return the plots
            return generate_column_violin_plots(df_global, color_map, quality_metric_cols)