from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import polars as pl
from dash import dcc, html
import dash_bootstrap_components as dbc
import matplotlib.pyplot as plt
from textwrap import dedent

from pixel_patrol_base.report.data_utils import get_all_available_dimensions, get_dim_aware_column
from pixel_patrol_base.report.factory import create_info_icon

# ---------- ID CONSTANTS ----------

PALETTE_SELECTOR_ID = "palette-selector"
GLOBAL_CONFIG_STORE_ID = "global-config-store"
FILTERED_INDICES_STORE_ID = "global-filtered-indices-store"

GLOBAL_GROUPBY_COLS_ID = "global-groupby-cols"
GLOBAL_FILTER_COLUMN_ID = "global-filter-column"
GLOBAL_FILTER_OP_ID = "global-filter-op"
GLOBAL_FILTER_TEXT_ID = "global-filter-text"
GLOBAL_DIM_FILTER_TYPE = "global-dim-filter" # _TYPE refers to a dynamic group

GLOBAL_APPLY_BUTTON_ID = "global-apply-button"
GLOBAL_RESET_BUTTON_ID = "global-reset-button"

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

DIMENSION_COL_PATTERN = re.compile(r"(?:_[a-zA-Z]\d+)+$")

def _find_candidate_columns(df: pl.DataFrame) -> Tuple[List[str], List[str]]:
    group_cols: List[str] = []
    filter_cols: List[str] = []

    schema = df.schema

    for c in df.columns:
        # 1. Skip technical dimension columns (BIGGEST SPEEDUP)
        if DIMENSION_COL_PATTERN.search(c):
            continue

        dtype = schema[c]

        # 2. Check if it's a numeric float
        is_float = dtype in (pl.Float32, pl.Float64)
        is_int = dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                           pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64)

        # 3. Handle Floats (No cardinality check)
        if is_float:
            # Floats are always valid for filtering (comparisons), never for grouping
            filter_cols.append(c)
            continue

            # 4. Handle Categoricals/Ints/Bools (Slow path - needs cardinality)
        # We only run n_unique() here because we need it to decide if
        # a string/int is "categorical enough" to be a dropdown option.
        try:
            n_unique = df.select(pl.col(c).n_unique()).item()
        except Exception:
            continue

        # Strings/Bools/Ints with low cardinality are good for Grouping
        if n_unique <= MAX_UNIQUE_GROUP:
            group_cols.append(c)

        # Strings/Bools/Ints with medium cardinality are good for Filtering
        # (Ints are always added to filter_cols below, so this covers str/bool)
        if n_unique <= MAX_UNIQUE_FILTER:
            filter_cols.append(c)

        # 5. Handle Integers specifically
        # Integers are always valid filter candidates (like Floats), even if high cardinality
        if is_int:
            # Avoid duplicates if it was already added by the n_unique check above
            if c not in filter_cols:
                filter_cols.append(c)

    return group_cols, filter_cols

def build_sidebar(df: pl.DataFrame, default_palette_name: str):

    filter_help_md = dedent("""
    Filter matches rows by one column.

    **Text**  
    - contains: substring match  
    - doesn't contain: inverse substr match  
    - equals (=): exact match  
    - is in: comma-separated exact matches (text)  

    **Numbers**  
    >, ≥, <, ≤, equals (=)
    """).strip()


    candidate_group_cols, candidate_filter_cols = _find_candidate_columns(df)
    available_dims = get_all_available_dimensions(df)

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
                        "If left empty, `report_group` is `imported_path_short` (if available).",
                        className="text-muted",
                    ),
                    html.Hr(),

                    # Simple global filter
                    html.Div(
                        [
                            html.Label("Filter rows (optional)", className="mt-1 mb-0"),
                            create_info_icon("global-filter", filter_help_md),
                        ],
                        className="d-flex align-items-center justify-content-between",
                    ),
                    dcc.Dropdown(
                        id=GLOBAL_FILTER_COLUMN_ID,
                        options=[{"label": c, "value": c} for c in candidate_filter_cols],
                        placeholder="Choose column to filter on",
                        clearable=True,
                        style={"width": "100%"},
                    ),
                    dcc.Dropdown(
                        id=GLOBAL_FILTER_OP_ID,
                        options=[
                            {"label": "contains (text)", "value": "contains"},
                            {"label": "doesn't contain (text)", "value": "not_contains"},
                            {"label": "equals (=) (text/number)", "value": "eq"},
                            {"label": "> (number)", "value": "gt"},
                            {"label": "≥ (number)", "value": "ge"},
                            {"label": "< (number)", "value": "lt"},
                            {"label": "≤ (number)", "value": "le"},
                            {"label": "is in (comma-separated) (text)", "value": "in"},
                        ],
                        value="in",
                        clearable=False,
                        style={"width": "100%", "marginTop": "4px"},
                    ),
                    dcc.Input(
                        id=GLOBAL_FILTER_TEXT_ID,
                        type="text",
                        placeholder="Value (or comma-separated list for 'in')",
                        style={"width": "100%", "marginTop": "4px"},
                    ),
                    html.Hr(),

                    html.Label("Select Dimensions", className="mt-1"),
                    html.Div([
                        dbc.Row(
                            [
                                dbc.Col([
                                    html.Small(dim.upper()),
                                    dcc.Dropdown(
                                        id={"type": GLOBAL_DIM_FILTER_TYPE, "index": dim},
                                        options=[{"label": "All", "value": "All"}] + [{"label": x, "value": x} for x in
                                                                                      vals],
                                        value="All",
                                        clearable=False,
                                        className="mb-1"
                                    )
                                ], width=3, className="px-1")
                                for dim, vals in sorted(available_dims.items())
                                # Filter (>1) is now handled in get_all_available_dimensions
                            ],
                            className="g-0 mb-2"
                        )
                    ]),
                    html.Hr(),

                    dbc.Row([
                        dbc.Col(
                            html.Button(
                                "Apply",
                                id=GLOBAL_APPLY_BUTTON_ID,
                                n_clicks=0,
                                className="btn btn-primary btn-sm w-100",
                            ), width=6
                        ),
                        dbc.Col(
                            html.Button(
                                "Reset",
                                id=GLOBAL_RESET_BUTTON_ID,
                                n_clicks=0,
                                className="btn btn-outline-danger btn-sm w-100",
                            ), width=6
                        ),
                    ], className="mt-3 g-2"),

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
            data={"group_cols": default_group_value, "filters": {}, "dimensions": {}},
        )
    ]

    extra_components = [
        dcc.Download(id=EXPORT_CSV_DOWNLOAD_ID),
        dcc.Download(id=EXPORT_PROJECT_DOWNLOAD_ID),
    ]

    return sidebar_controls, stores, extra_components


# ---------- LOGIC: CENTRALIZED HELPERS ----------

def compute_filtered_indices(df: pl.DataFrame,
                             global_config: Optional[Dict]
                             ) -> Optional[List[int]]:
    """
    Computes the indices of rows that match the global row filters AND dimension filters.
    Widgets will use these indices to quickly slice the dataframe.
    """
    global_config = global_config or {}
    filters = global_config.get("filters") or {}
    dimensions = global_config.get("dimensions") or {}

    # If no filters and no dimension constraints, return all indices
    if not filters and not dimensions:
        return None

    # 1. Apply Standard Value Filters
    mask = pl.lit(True)
    for col_name, spec in filters.items():
        if col_name in df.columns:
            mask = mask & _filter_expr(col_name, spec)

    df_subset = df.filter(mask)

    # 2. Apply Dimension Filters (Optimization: computed once globally)
    # This filters out rows that have NO data for the selected dimensions (e.g. T=0)
    df_subset = filter_rows_by_any_dimension(df_subset, dimensions)

    return df_subset.select(pl.col("unique_id")).to_series().to_list()


def resolve_group_column(df: pl.DataFrame, global_config: Optional[Dict]) -> str:
    """Helper to safely extract the active grouping column."""
    global_config = global_config or {}
    group_cols = list(global_config.get("group_cols") or [])

    group_col = group_cols[0] if group_cols else DEFAULT_GROUP_COL
    if group_col not in df.columns:
        group_col = DEFAULT_GROUP_COL if DEFAULT_GROUP_COL in df.columns else df.columns[0]

    return group_col


def filter_rows_by_any_dimension(
    df: pl.DataFrame,
    dims_selection: Dict[str, str],
) -> pl.DataFrame:
    """
    Keep only rows that have *any* non-null metric column
    for the currently selected dimensions.
    """
    dims_selection = dims_selection or {}

    # Build tokens like "_t0", "_z1", ignoring "All"
    tokens: List[str] = []
    for dim, val in dims_selection.items():
        if not val or val == "All":
            continue
        token = val if val.startswith(dim) else f"{dim}{val}"
        tokens.append(f"_{token}")

    if not tokens:
        return df  # no dim filters -> no change

    # Columns that match *all* selected dim tokens (order-independent)
    dim_cols = [
        c for c in df.columns
        if all(tok in c for tok in tokens)
    ]
    if not dim_cols:
        return df.head(0)  # no column exists for those dims

    # Keep rows where at least one of those columns is non-null
    mask = pl.col(dim_cols[0]).is_not_null()
    for c in dim_cols[1:]:
        mask = mask | pl.col(c).is_not_null()

    return df.filter(mask)


def prepare_widget_data(
        df: pl.DataFrame,
        subset_indices: Optional[List[int]],
        global_config: Dict,
        metric_base: Optional[str] = None
) -> Tuple[pl.DataFrame, str, Optional[str], Optional[str]]:
    """
    The main coordinator for widgets.
    1. Slices the global `df` using `subset_indices`.
    2. Resolves the correct grouping column.
    3. If `metric_base` is provided, attempts to find the dimension-specific column (e.g. 'area_t0').

    Returns:
        (df_filtered, group_col, resolved_column_name, warning_message)
    """
    # 1. Filter Rows
    if subset_indices is not None:
        df_filtered = df.filter(pl.col("unique_id").is_in(subset_indices))
    else:
        df_filtered = df

    if df_filtered.is_empty():
        return df_filtered, "", None, "No data matches the current filters."

    # 2. Resolve Grouping
    group_col = resolve_group_column(df_filtered, global_config)

    # 3. Resolve Dimension-Specific Column (if needed)
    resolved_col = None
    warning_msg = None
    dims_selection = global_config.get("dimensions", {})

    if metric_base:
        resolved_col = get_dim_aware_column(
            df_filtered.columns,
            metric_base,
            dims_selection,
        )

        if resolved_col is None:
            has_dim_filter = any(v and v != "All" for v in dims_selection.values())
            if has_dim_filter:
                warning_msg = (
                    f"Metric '{metric_base}' is not available for the selected dimensions "
                    f"({', '.join(f'{k}={v}' for k, v in dims_selection.items() if v != 'All')})."
                )
            else:
                warning_msg = f"Metric '{metric_base}' not found in dataset."
            df_filtered = df_filtered.head(0)

        else:
            df_filtered = df_filtered.filter(pl.col(resolved_col).is_not_null())
            if df_filtered.is_empty():
                warning_msg = (
                    "No data matches the selected dimensions for "
                    f"metric '{metric_base}'."
                )

    return df_filtered, group_col, resolved_col, warning_msg


def apply_global_row_filters_and_grouping(
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

def _parse_list(raw: str) -> List[str]:
    return [v.strip() for v in raw.split(",") if v.strip()]

def _try_float(raw: str) -> Optional[float]:
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None

def _filter_expr(col: str, spec: object) -> pl.Expr:
    # Back-compat: old format was list[str] meaning "in"
    if isinstance(spec, list):
        vals = [str(v) for v in spec if str(v).strip()]
        return pl.col(col).is_in(vals)

    if not isinstance(spec, dict):
        return pl.lit(True)

    op = spec.get("op")
    raw = str(spec.get("value", "")).strip()
    if not op or not raw:
        return pl.lit(True)

    s = pl.col(col)

    if op == "in":
        vals = _parse_list(raw)
        return s.cast(pl.Utf8).is_in(vals)

    if op == "eq":
        # numeric if possible, else string
        num = _try_float(raw)
        if num is not None:
            return s.cast(pl.Float64) == num
        return s.cast(pl.Utf8) == raw

    if op == "contains":
        return s.cast(pl.Utf8).fill_null("").str.contains(raw, literal=True)

    if op == "not_contains":
        return ~s.cast(pl.Utf8).fill_null("").str.contains(raw, literal=True)

    # numeric comparisons
    num = _try_float(raw)
    if num is None:
        return pl.lit(True)

    x = s.cast(pl.Float64)
    if op == "gt":
        return x > num
    if op == "ge":
        return x >= num
    if op == "lt":
        return x < num
    if op == "le":
        return x <= num

    return pl.lit(True)
