import os
from pathlib import Path
from typing import List, Dict, Set, Tuple, Any

import polars as pl
from dash import dcc, Input, Output, html

from pixel_patrol_base.report.widget_categories import WidgetCategories
from pixel_patrol_base.report.base_widget import BaseReportWidget
from pixel_patrol_base.report.global_controls import (
    prepare_widget_data,
    GLOBAL_CONFIG_STORE_ID,
    FILTERED_INDICES_STORE_ID,
)
from pixel_patrol_base.report.factory import plot_sunburst, show_no_data_message

MIXED_COLOR = "#cccccc"

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
        """

        # --- 1. Identify Common Root ---
        all_paths = df["path"].to_list()
        if not all_paths:
            return [], [], [], [], []

        try:
            common_root = os.path.commonpath(all_paths)
        except ValueError:
            common_root = ""

        path_sep = os.sep

        vis_root_name = Path(common_root).name if common_root else ""
        display_root_label = "Root"

        # --- 2. Data Structures for Aggregation ---
        # Stores file_count for size and 'groups' for coloring logic
        # format: path_str -> {file_count: int, groups: set(), parent: str, label: str}
        nodes: Dict[str, Dict] = {
            "": {  # virtual root node; label is the visible root name
                "file_count": 0,
                "groups": set(),
                "parent": "",
                "label": display_root_label,
            }
        }

        # so the sunburst always shows at least two levels (folder -> files).
        if vis_root_name:
            nodes[vis_root_name] = {
                "file_count": 0,
                "groups": set(),
                "parent": "",
                "label": vis_root_name,
                }

        # --- 3. Iterate Files and Build Tree Upwards ---
        # Selects only path and group column (file size is not needed for file count)
        subset = df.select([
            pl.col("path"),
            pl.col(group_col).fill_null("Unknown").cast(pl.String).alias("group"),
        ])

        for full_path, group_val in subset.iter_rows():

            # Normalize path relative to common root
            if common_root:
                try:
                    rel_path = os.path.relpath(full_path, common_root)
                except ValueError:
                    rel_path = full_path
            else:
                rel_path = full_path


            if vis_root_name:
                rel_path = os.path.join(vis_root_name, rel_path
                                        ) if rel_path not in ("", ".") else vis_root_name
            parts = rel_path.split(path_sep)

            # --- Process Leaf (File) ---
            file_id = rel_path
            file_label = parts[-1]
            parent_id = path_sep.join(parts[:-1]) if len(parts) > 1 else ""

            # Add Leaf Node (size = 1 file)
            nodes[file_id] = {
                "file_count": 1,
                "groups": {group_val},
                "parent": parent_id,
                "label": file_label,
            }

            # --- Process Ancestors (Folders) ---
            current_path = parent_id

            while True:
                if current_path in nodes:
                    # Existing folder: update stats
                    nodes[current_path]["file_count"] += 1
                    nodes[current_path]["groups"].add(group_val)
                else:
                    # New folder node
                    folder_label = os.path.basename(current_path) if current_path else display_root_label
                    parent_of_folder = os.path.dirname(current_path) if current_path else ""

                    nodes[current_path] = {
                        "file_count": 1,
                        "groups": {group_val},
                        "parent": parent_of_folder,
                        "label": folder_label,
                    }

                if current_path == "":
                    break

                # Move one level up
                current_path = os.path.dirname(current_path)

        # --- 4. Generate Final Lists and Colors ---

        ids = []
        labels = []
        parents = []
        values = []
        colors = []

        for node_id, data in nodes.items():
            # Skip the root's parent logic
            if node_id == "" and data["parent"] == "":
                pass
            elif node_id == "":
                # Root node special handling
                ids.append(node_id)
                labels.append(data["label"])
                parents.append("")
                values.append(data["file_count"])
                colors.append(MIXED_COLOR)
                continue

            ids.append(node_id)
            labels.append(data["label"])
            parents.append(data["parent"])
            values.append(data["file_count"])  # Use file_count for values

            # Logic: If exactly one group is present in this node (and all its children),
            # use that group's color. Otherwise, Gray.
            unique_groups = data["groups"]

            if len(unique_groups) == 1:
                g_val = list(unique_groups)[0]
                c = color_map.get(g_val, MIXED_COLOR)
                colors.append(c)
            else:
                colors.append(MIXED_COLOR)

        return ids, labels, parents, values, colors