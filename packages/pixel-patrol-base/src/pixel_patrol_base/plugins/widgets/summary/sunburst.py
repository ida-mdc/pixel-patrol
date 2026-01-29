import os
from pathlib import Path
from typing import List, Dict, Set, Tuple, Any

import polars as pl
from dash import dcc, Input, Output, html

from pixel_patrol_base.report.widget_categories import WidgetCategories
from pixel_patrol_base.report.base_widget import BaseReportWidget
from pixel_patrol_base.report.global_controls import prepare_widget_data
from pixel_patrol_base.report.constants import GLOBAL_CONFIG_STORE_ID, FILTERED_INDICES_STORE_ID, MIXED_GROUPING_COLOR
from pixel_patrol_base.report.factory import plot_sunburst, show_no_data_message

MAX_FILES_FOR_SUNBURST = 500  # Maximum number of files before switching to folders-only view

class FileSunburstWidget(BaseReportWidget):
    """Display file structure as a sunburst plot."""

    # ---- Declarative spec ----
    NAME: str = "File Structure Sunburst"
    TAB: str = WidgetCategories.SUMMARY.value
    REQUIRES: Set[str] = {"path", "size_bytes"}
    REQUIRES_PATTERNS = None
    CONTENT_ID = "file-sunburst-content"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._df: pl.DataFrame | None = None

    @property
    def help_text(self) -> str:
        return (
            "Sunburst view of the **file and folder hierarchy**.\n\n"
            "Click a slice to zoom in; click the center to zoom out.\n"
        )

    def get_content_layout(self) -> List:
        return [html.Div(id=self.CONTENT_ID)]

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
        color_map: Dict[str, Any] | None,
        subset_indices: List[int] | None,
        global_config: Dict | None,
    ):

        df_filtered, group_col, _resolved, _warning_msg, _order = prepare_widget_data(
            self._df,
            subset_indices,
            global_config or {},
            metric_base=None,
        )

        if df_filtered.is_empty():
            return show_no_data_message()

        # 2. Build Hierarchy and Colors
        ids, labels, parents, values, colors = self._build_hierarchy_and_colors(
            df_filtered, group_col, color_map
        )

        # 3. Construct Figure
        fig = plot_sunburst(
                ids=ids,
                labels=labels,
                parents=parents,
                values=values,
                colors=colors,
                hovertemplate="<b>%{label}</b><br>Files: %{value}<br>Group: %{id}<extra></extra>",
        )

        return dcc.Graph(figure=fig)

    @staticmethod
    def _build_hierarchy_and_colors(
            df: pl.DataFrame,
            group_col: str,
            color_map: Dict[str, str]
    ) -> Tuple[List[str], List[str], List[str], List[int], List[str]]:
        """
        Constructs the node lists required for Plotly Sunburst, aggregating by file count.
        If the number of files exceeds MAX_FILES_FOR_SUNBURST, only folders are shown.
        """

        # --- 1. Identify Common Root ---
        path_series = df["path"]
        if path_series.is_empty():
            return [], [], [], [], []


        all_paths = path_series.to_list()

        num_files = len(all_paths)
        folders_only = num_files > MAX_FILES_FOR_SUNBURST

        try:
            common_root = os.path.commonpath(all_paths)
        except ValueError:
            common_root = ""

        path_sep = os.sep
        vis_root_name = Path(common_root).name if common_root else ""
        display_root_label = "Root"

        # --- 2. Pre-compute relative paths using Polars ---

        df_processed = df.select([
            pl.col("path"),
            pl.col(group_col).fill_null("Unknown").cast(pl.String).alias("group"),
        ])

        if common_root:
            # Strip common root prefix using Polars (vectorized)
            # Handle both with and without trailing separator
            common_root_with_sep = common_root.rstrip(path_sep) + path_sep

            df_processed = df_processed.with_columns(
                pl.when(pl.col("path") == common_root)
                .then(pl.lit("."))
                .when(pl.col("path").str.starts_with(common_root_with_sep))
                .then(pl.col("path").str.slice(len(common_root_with_sep)))
                .otherwise(pl.col("path"))
                .alias("rel_path")
            )
        else:
            df_processed = df_processed.with_columns(
                pl.col("path").alias("rel_path")
            )

        # Add visual root prefix if needed (vectorized)
        if vis_root_name:
            df_processed = df_processed.with_columns(
                pl.when(pl.col("rel_path").is_in(["", "."]))
                .then(pl.lit(vis_root_name))
                .otherwise(pl.lit(vis_root_name) + pl.lit(path_sep) + pl.col("rel_path"))
                .alias("rel_path")
            )

        # --- 3. Extract file data in batch ---
        # Convert to list of tuples once (faster than iter_rows for processing)
        file_data = df_processed.select(["rel_path", "group"]).rows()

        # --- 4. Data Structures for Aggregation ---
        node_file_count: Dict[str, int] = {"": 0}
        node_groups: Dict[str, Set[str]] = {"": set()}
        node_parent: Dict[str, str] = {"": ""}
        node_label: Dict[str, str] = {"": display_root_label}

        if vis_root_name:
            node_file_count[vis_root_name] = 0
            node_groups[vis_root_name] = set()
            node_parent[vis_root_name] = ""
            node_label[vis_root_name] = vis_root_name

        # --- 5. Build Tree ---
        for rel_path, group_val in file_data:
            # For files, always compute parent (files aren't in node_parent)
            parent_id = _get_parent(rel_path, path_sep)

            if folders_only:
                current_path = parent_id
            else:
                # --- Process Leaf (File) ---
                node_file_count[rel_path] = 1
                node_groups[rel_path] = {group_val}
                node_parent[rel_path] = parent_id
                node_label[rel_path] = _get_label(rel_path, path_sep, display_root_label)
                current_path = parent_id

            # --- Process Ancestors (Folders) ---
            while True:
                if current_path in node_file_count:
                    # Existing folder: propagate counts up using stored parents
                    while True:
                        node_file_count[current_path] += 1
                        node_groups[current_path].add(group_val)
                        if current_path == "":
                            break
                        current_path = node_parent[current_path]
                    break
                else:
                    # New folder: compute parent, store everything
                    parent_of_folder = _get_parent(current_path, path_sep)
                    node_file_count[current_path] = 1
                    node_groups[current_path] = {group_val}
                    node_parent[current_path] = parent_of_folder
                    node_label[current_path] = _get_label(current_path, path_sep, display_root_label)

                    if current_path == "":
                        break
                    current_path = parent_of_folder

        # --- 6. Generate Final Lists and Colors ---
        ids = []
        labels = []
        parents = []
        values = []
        colors = []

        # Reserve capacity (Python lists don't have reserve, but this helps readability)
        # In practice, the append operations are already efficient

        color_map = color_map or {}

        for node_id in node_file_count:
            # Skip empty root special case handled below
            if node_id == "" and node_parent.get("", "") == "":
                # Root node special handling
                ids.append(node_id)
                labels.append(node_label[node_id])
                parents.append("")
                values.append(node_file_count[node_id])
                colors.append(MIXED_GROUPING_COLOR)
                continue

            # When folders_only is True, we never created file nodes, so all nodes are folders
            ids.append(node_id)
            labels.append(node_label[node_id])
            parents.append(node_parent[node_id])
            values.append(node_file_count[node_id])

            # Color logic: single group = group color, multiple = mixed
            unique_groups = node_groups[node_id]
            if len(unique_groups) == 1:
                g_val = next(iter(unique_groups))  # Faster than list()[0]
                colors.append(color_map.get(g_val, MIXED_GROUPING_COLOR))
            else:
                colors.append(MIXED_GROUPING_COLOR)

        return ids, labels, parents, values, colors

def _get_parent(path: str, sep: str) -> str:
    """Get parent path from a path string."""
    idx = path.rfind(sep)
    return path[:idx] if idx != -1 else ""

def _get_label(path: str, sep: str, default: str) -> str:
    """Get the last component (label) from a path string."""
    if not path:
        return default
    idx = path.rfind(sep)
    return path[idx + 1:] if idx != -1 else path