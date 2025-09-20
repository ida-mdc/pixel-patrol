import re
from typing import List, Dict, Set

import polars as pl
from dash import html, Input, Output

from pixel_patrol_base.plugins.processors.basic_stats_processor import BasicStatsProcessor
from pixel_patrol_base.core.feature_schema import get_requirements_as_patterns
from pixel_patrol_base.report.utils import generate_column_violin_plots, get_sortable_columns
from pixel_patrol_base.report.widget_categories import WidgetCategories


class DatasetStatsWidget:
    # ---- Declarative spec ----
    NAME: str = "Pixel Value Statistics"
    TAB: str = WidgetCategories.DATASET_STATS.value

    # We need at least the grouping column, plus “some” BasicStats numeric columns.
    REQUIRES: Set[str] = {"imported_path_short"}
    # Use the processor’s declared output patterns to ensure at least one metric exists.
    REQUIRES_PATTERNS = tuple(get_requirements_as_patterns(BasicStatsProcessor))

    # Component IDs
    CONTAINER_ID = "dataset-stats-container"

    def layout(self) -> List:
        """Single container populated by the callback."""
        return [
            html.Div(id=self.CONTAINER_ID),
            # (Optional) keep your long markdown block commented out or add here
        ]

    def register(self, app, df_global: pl.DataFrame):
        @app.callback(
            Output(self.CONTAINER_ID, "children"),
            Input("color-map-store", "data"),
        )
        def update_dataset_stats_layout(color_map: Dict[str, str]):
            color_map = color_map or {}

            # Resolve the actual metric columns from the declared regex patterns
            cols = get_sortable_columns(df_global)

            patterns = self.REQUIRES_PATTERNS or ()
            numeric_cols = sorted(
                {c for c in cols for pat in patterns if re.search(pat, c)}
            )

            # Delegate plotting to your existing helper
            return generate_column_violin_plots(df_global, color_map, numeric_cols)
