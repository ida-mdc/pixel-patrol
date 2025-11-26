from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import polars as pl
from dash import dcc, html
import dash_bootstrap_components as dbc
import matplotlib.pyplot as plt


# ---------- ID CONSTANTS ----------

PALETTE_SELECTOR_ID = "palette-selector"
GLOBAL_CONFIG_STORE_ID = "global-config-store"
GLOBAL_GROUPBY_COLS_ID = "global-groupby-cols"
GLOBAL_FILTER_COLUMN_ID = "global-filter-column"
GLOBAL_FILTER_TEXT_ID = "global-filter-text"
GLOBAL_APPLY_BUTTON_ID = "global-apply-button"


# ---------- LAYOUT: SIDEBAR + STORES ----------

def _find_candidate_columns(df: pl.DataFrame) -> Tuple[List[str], List[str]]:
    MAX_UNIQUE_GROUP = 50 # TODO: move to config
    MAX_UNIQUE_FILTER = 200 # TODO: once we allow for more complex filtering (eg. >x), this will not be needed


    group_cols: List[str] = []
    filter_cols: List[str] = []

    for c in df.columns:
        s = df[c]

        # cardinality
        try:
            n_unique = df.select(pl.col(c).n_unique()).item()
        except Exception:
            n_unique = None

        if n_unique is None:
            continue

        # FILTER candidates
        if n_unique <= MAX_UNIQUE_FILTER:
            if s.dtype in (pl.Utf8, pl.Boolean):
                filter_cols.append(c)
            # if you later want ints too, you can add a simple check here
            # (e.g. s.dtype == pl.Int64, etc.)

        # GROUP-BY candidates: stricter
        if n_unique <= MAX_UNIQUE_GROUP:
            if s.dtype in (pl.Utf8, pl.Boolean):
                group_cols.append(c)

    return group_cols, filter_cols

def build_sidebar(df: pl.DataFrame, default_palette_name: str):
    """
    Build sidebar controls and associated dcc.Store components.

    Returns:
        sidebar_controls, stores_list
    """
    candidate_group_cols, candidate_filter_cols = _find_candidate_columns(df)

    default_group_value: List[str] = []
    if "imported_path_short" in df.columns:
        default_group_value = ["imported_path_short"]
    elif "imported_path" in df.columns:
        default_group_value = ["imported_path"]

    sidebar_controls = dbc.Card(
        [
            dbc.CardHeader(html.H4("Global settings", className="mb-0")),
            dbc.CardBody(
                [
                    # Color palette
                    html.Label("Color palette", className="mt-1"),
                    dcc.Dropdown(
                        id=PALETTE_SELECTOR_ID,
                        options=[
                            {"label": name, "value": name}
                            for name in sorted(plt.colormaps())
                        ],
                        value=default_palette_name,
                        clearable=False,
                        style={"width": "100%"},
                    ),
                    html.Hr(),

                    # Global grouping
                    html.Label("Group by column(s)", className="mt-1"),
                    dcc.Dropdown(
                        id=GLOBAL_GROUPBY_COLS_ID,
                        options=[{"label": c, "value": c} for c in candidate_group_cols],
                        value=default_group_value,
                        multi=True,
                        clearable=True,
                        placeholder="e.g. imported_path_short, condition",
                        style={"width": "100%"},
                    ),
                    html.Small(
                        "If left empty, falls back to imported_path_short (if available).",
                        className="text-muted",
                    ),
                    html.Hr(),

                    # Simple global filter
                    html.Label("Filter rows (optional)", className="mt-1"),
                    dcc.Dropdown(
                        id=GLOBAL_FILTER_COLUMN_ID,
                        options=[{"label": c, "value": c} for c in candidate_filter_cols],
                        placeholder="Choose column to filter on",
                        clearable=True,
                        style={"width": "100%"},
                    ),
                    dcc.Input(
                        id=GLOBAL_FILTER_TEXT_ID,
                        type="text",
                        placeholder="Allowed values (comma-separated)",
                        style={"width": "100%", "marginTop": "4px"},
                    ),

                    html.Button(
                        "Apply",
                        id=GLOBAL_APPLY_BUTTON_ID,
                        n_clicks=0,
                        className="btn btn-primary btn-sm mt-3 w-100",
                    ),
                ]
            ),
        ],
        className="mb-3",
        style={"position": "sticky", "top": "20px"},
    )

    stores = [
        dcc.Store(
            id=GLOBAL_CONFIG_STORE_ID,
            data={"group_cols": default_group_value, "filters": {}},
        )
    ]

    return sidebar_controls, stores


# ---------- LOGIC: APPLY GLOBAL CONFIG ----------

def apply_global_config(
    df: pl.DataFrame,
    global_config: Optional[Dict],
    default_group_col: str = "imported_path_short",
) -> Tuple[pl.DataFrame, str]:
    """
    Apply global filters & produce a grouping column.

    Returns:
        (df_with_group_col, group_col_name)
    """
    global_config = global_config or {}
    group_cols = list(global_config.get("group_cols") or [])

    # ensure default grouping exists
    if default_group_col not in df.columns:
        if "imported_path" in df.columns:
            df = df.with_columns(
                pl.col("imported_path").map_elements(
                    lambda x: Path(x).name if x is not None else "Unknown Folder",
                    return_dtype=pl.String,
                ).alias(default_group_col)
            )

    if not group_cols:
        group_cols = [default_group_col]

    group_cols = [c for c in group_cols if c in df.columns] or [default_group_col]

    # filters: {column -> [allowed_values]}
    filters = global_config.get("filters") or {}
    for col, allowed_vals in filters.items():
        if col in df.columns and allowed_vals:
            df = df.filter(pl.col(col).is_in(allowed_vals))

    group_col_name = "__global_group"
    if len(group_cols) == 1:
        df = df.with_columns(pl.col(group_cols[0]).alias(group_col_name))
    else:
        df = df.with_columns(
            pl.concat_str(group_cols, separator=" | ").alias(group_col_name)
        )

    return df, group_col_name
