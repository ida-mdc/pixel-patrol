import os
from pathlib import Path
from typing import List, Dict, Any, Set

import plotly.graph_objects as go
import polars as pl
from dash import dcc, Input, Output
from pixel_patrol_base.report.widget_categories import WidgetCategories


class FileSunburstWidget:
    """Display file structure as a sunburst plot."""

    # ---- Declarative spec ----
    NAME: str = "File Structure Sunburst"
    TAB: str = WidgetCategories.SUMMARY.value
    REQUIRES: Set[str] = {"path", "size_bytes", "imported_path_short"}
    REQUIRES_PATTERNS = None

    # Component IDs
    GRAPH_ID = "file-sunburst-plot"

    def layout(self) -> List:
        return [dcc.Graph(id=self.GRAPH_ID)]

    def register(self, app, df_global: pl.DataFrame):
        @app.callback(
            Output(self.GRAPH_ID, "figure"),
            Input("color-map-store", "data"),
        )
        def update_sunburst_plot(color_map: Dict[str, Any]) -> go.Figure:
            color_map = color_map or {}

            # --- 1. Find common root and create relative paths ---
            all_file_paths = df_global["path"].drop_nulls().to_list()
            if not all_file_paths:
                return go.Figure()
            common_root = os.path.commonpath(all_file_paths)
            common_root_path = Path(common_root)
            vis_root_name = common_root_path.name

            files_df = (
                df_global.select(
                    pl.col("path").map_elements(
                        lambda p: str(Path(p).relative_to(common_root_path)),
                        return_dtype=pl.String,
                    ).alias("path"),
                    pl.col("path").map_elements(
                        lambda p: str(Path(p).parent.relative_to(common_root_path)),
                        return_dtype=pl.String,
                    ).alias("parent"),
                    pl.col("size_bytes"),
                    pl.col("imported_path_short"),
                    pl.lit(1, dtype=pl.Int64).alias("file_count"),
                )
                # ensure files at root attach to the root node
                .with_columns(
                    pl.when(pl.col("parent") == ".")
                    .then(pl.lit(vis_root_name))
                    .otherwise(pl.col("parent"))
                    .alias("parent")
                )
            )

            # --- 2. Build the complete folder hierarchy ---
            all_folders = set()
            parents_from_files = files_df.with_columns(
                pl.when(pl.col("parent") == ".")
                .then(pl.lit(vis_root_name))
                .otherwise(pl.col("parent"))
                .alias("parent")
            )
            for p in parents_from_files["parent"].drop_nulls():
                if p and p != vis_root_name:
                    parts = p.split(os.sep)
                    for i in range(1, len(parts) + 1):
                        all_folders.add(os.sep.join(parts[:i]))

            folders_df = (
                pl.DataFrame({"path": list(all_folders)})
                .with_columns(
                    pl.col("path").map_elements(
                        lambda p: str(Path(p).parent),
                        return_dtype=pl.String
                    ).alias("parent"),
                    pl.lit(0, dtype=pl.Int64).alias("size_bytes"),
                    pl.lit(0, dtype=pl.Int64).alias("file_count"),
                    pl.lit(None, dtype=pl.String).alias("imported_path_short"),
                )
                .with_columns(
                    pl.when(pl.col("parent") == ".")
                    .then(pl.lit(vis_root_name))
                    .otherwise(pl.col("parent"))
                    .alias("parent")
                )
            )

            # --- 3. Root node ---
            root_df = pl.DataFrame(
                {
                    "path": [vis_root_name],
                    "parent": [""],
                    "size_bytes": [0],
                    "file_count": [0],
                    "imported_path_short": [None],
                }
            )

            # --- 4. Final schema & concat ---
            final_cols = ["path", "parent", "size_bytes", "file_count", "imported_path_short"]
            final_df = (
                pl.concat(
                    [
                        files_df.select(final_cols),
                        folders_df.select(final_cols),
                        root_df.select(final_cols),
                    ]
                )
                .unique(subset=["path"], keep="first")
            )

            # --- 5. Labels & colors ---
            display_labels = final_df["path"].map_elements(
                lambda p: Path(p).name,
                return_dtype=pl.String,
            )

            marker_colors = []
            for row in final_df.iter_rows(named=True):
                if row["parent"] == vis_root_name:
                    # Color direct children of root by their original imported_path_short
                    original_folder_name = files_df.filter(pl.col("path").str.starts_with(row["path"]))[
                        "imported_path_short"
                    ].head(1)
                    if len(original_folder_name) > 0:
                        marker_colors.append(color_map.get(original_folder_name[0], "#cccccc"))
                    else:
                        marker_colors.append("#cccccc")
                else:
                    marker_colors.append(None)

            fig = go.Figure(
                go.Sunburst(
                    ids=final_df["path"],
                    labels=display_labels,
                    parents=final_df["parent"],
                    values=final_df["file_count"],
                    branchvalues="remainder",
                    marker_colors=marker_colors,
                    hovertext=display_labels,
                    hovertemplate="<b>%{hovertext}</b><br>Path: %{id}<extra></extra>",
                )
            )
            fig.update_layout(margin=dict(t=40, l=20, r=20, b=20), height=500)
            return fig
