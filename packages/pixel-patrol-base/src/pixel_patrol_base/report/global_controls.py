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
EXPORT_CSV_BUTTON_ID = "export-csv-button"
EXPORT_PROJECT_BUTTON_ID = "export-project-button"
EXPORT_CSV_DOWNLOAD_ID = "export-csv-download"
EXPORT_PROJECT_DOWNLOAD_ID = "export-project-download"

# ---------- GLOBAL CONSTANTS ----------

REPORT_GROUP_COL = "report_group"
DEFAULT_GROUP_COL = REPORT_GROUP_COL

MAX_UNIQUE_GROUP = 12  # TODO: move to config
MAX_UNIQUE_FILTER = 200 # TODO: once we allow for more complex filtering (eg. >x), this will not be needed

# ---------- LAYOUT: SIDEBAR + STORES ----------

def _find_candidate_columns(df: pl.DataFrame) -> Tuple[List[str], List[str]]:

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

        # GROUP-BY candidates: stricter
        if n_unique <= MAX_UNIQUE_GROUP:
            if s.dtype in (pl.Utf8, pl.Boolean):
                group_cols.append(c)

    return group_cols, filter_cols


def build_sidebar(df: pl.DataFrame, default_palette_name: str):

    candidate_group_cols, candidate_filter_cols = _find_candidate_columns(df)

    default_group_value: List[str] = []
    if REPORT_GROUP_COL in df.columns:
        default_group_value = [REPORT_GROUP_COL]

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
                        value=default_group_value[0] if default_group_value else None,
                        multi=False,
                        clearable=True,
                        placeholder="Choose grouping column",
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
                    html.Div(
                        [
                            html.Button(
                                "Export current table (CSV)",
                                id=EXPORT_CSV_BUTTON_ID,
                                n_clicks=0,
                                className="btn btn-outline-secondary btn-sm mt-3 w-100",
                            ),
                            html.Button(
                                "Export filtered project",
                                id=EXPORT_PROJECT_BUTTON_ID,
                                n_clicks=0,
                                className="btn btn-outline-secondary btn-sm mt-2 w-100",
                            ),
                        ]
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

    extra_components = [
        dcc.Download(id=EXPORT_CSV_DOWNLOAD_ID),
        dcc.Download(id=EXPORT_PROJECT_DOWNLOAD_ID),
    ]

    return sidebar_controls, stores, extra_components


# ---------- LOGIC: APPLY GLOBAL CONFIG ----------

def apply_global_config(
    df: pl.DataFrame,
    global_config: Optional[Dict],
    default_group_col: str = DEFAULT_GROUP_COL,
) -> Tuple[pl.DataFrame, str]:

    global_config = global_config or {}

    # group_cols in the store is always a list; we take the first element
    group_cols = list(global_config.get("group_cols") or [])

    if not group_cols:
        group_col = default_group_col
    else:
        group_col = group_cols[0]

    # ensure the chosen group_col exists; otherwise fall back to DEFAULT_GROUP_COL
    if group_col not in df.columns:
        group_col = default_group_col
    if group_col not in df.columns:
        # as a last resort pick the first column
        group_col = df.columns[0]

    # filters: {column -> [allowed_values]}
    filters = global_config.get("filters") or {}
    for col, allowed_vals in filters.items():
        if col in df.columns and allowed_vals:
            df = df.filter(pl.col(col).is_in(allowed_vals))

    # no extra column; widgets will group directly by `group_col`
    return df, group_col
