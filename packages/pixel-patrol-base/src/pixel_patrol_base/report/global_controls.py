from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import polars as pl
from dash import dcc, html
import dash_bootstrap_components as dbc
import matplotlib.pyplot as plt
from textwrap import dedent

from pixel_patrol_base.report.data_utils import (get_all_available_dimensions,
                                                 get_dim_aware_column,
                                                 ensure_discrete_grouping,
                                                 sort_strings_alpha)
from pixel_patrol_base.report.factory import create_info_icon
from pixel_patrol_base.report.constants import (MAX_UNIQUE_GROUP,
                                                GC_GROUP_COL,
                                                GC_FILTER,
                                                GC_DIMENSIONS,
                                                GC_IS_SHOW_SIGNIFICANCE,
                                                DEFAULT_REPORT_GROUP_COL,
                                                PALETTE_SELECTOR_ID,
                                                GLOBAL_CONFIG_STORE_ID,
                                                GLOBAL_GROUPBY_COLS_ID,
                                                NO_GROUPING_LABEL,
                                                NO_GROUPING_COL,
                                                GLOBAL_FILTER_COLUMN_ID,
                                                GLOBAL_FILTER_OP_ID,
                                                GLOBAL_FILTER_TEXT_ID,
                                                GLOBAL_DIM_FILTER_TYPE,
                                                GLOBAL_SHOW_SIGNIFICANCE_ID,
                                                GLOBAL_APPLY_BUTTON_ID,
                                                GLOBAL_RESET_BUTTON_ID,
                                                EXPORT_CSV_BUTTON_ID,
                                                EXPORT_CSV_DOWNLOAD_ID,
                                                EXPORT_PROJECT_BUTTON_ID,
                                                EXPORT_PROJECT_DOWNLOAD_ID,
                                                SAVE_SNAPSHOT_DOWNLOAD_ID,
                                                SAVE_SNAPSHOT_BUTTON_ID,
                                                )

import logging
logger = logging.getLogger(__name__)

# ---------- GLOBAL CONSTANTS ----------

_ALLOWED_FILTER_OPS = {"contains", "not_contains", "eq", "gt", "ge", "lt", "le", "in"}
_NUMERIC_FILTER_OPS = {"gt", "ge", "lt", "le"}
_DIMENSION_COL_PATTERN = re.compile(r"(?:_[a-zA-Z]\d+)+$")


# ---------- LAYOUT: SIDEBAR + STORES ----------

def is_group_col_accepted(df: pl.DataFrame, col: str) -> bool:
    """Single source of truth: is this column allowed for grouping?"""
    try:
        if not col or col not in df.columns:
            return False
        if _DIMENSION_COL_PATTERN.search(col):
            return False

        dtype = df.schema[col]
        if dtype in (pl.Float32, pl.Float64):
            return False

        n_unique = df.select(pl.col(col).n_unique()).item()
        return n_unique <= MAX_UNIQUE_GROUP
    except Exception:
        return False


def init_global_config(df: pl.DataFrame, initial: Optional[Dict]) -> Dict:
    """
    One-shot initializer for global config:
    - fills defaults
    - validates keys/columns/ops/types
    - returns a sanitized dict
    """
    cfg = {
        GC_GROUP_COL: DEFAULT_REPORT_GROUP_COL,
        GC_FILTER: {},
        GC_DIMENSIONS: {},
        GC_IS_SHOW_SIGNIFICANCE: False,
    }

    if initial:
        cfg.update(initial)
        # ensure nested dicts exist even if someone passes None
        cfg[GC_FILTER] = cfg.get(GC_FILTER) or {}
        cfg[GC_DIMENSIONS] = cfg.get(GC_DIMENSIONS) or {}
        cfg[GC_GROUP_COL] = cfg.get(GC_GROUP_COL)
    return _validate_global_config(df, cfg)


def _validate_global_config(df: pl.DataFrame, global_config: Optional[Dict]) -> Dict:
    cfg: Dict = dict(global_config or {})

    g = cfg.get(GC_GROUP_COL)
    if g and not is_group_col_accepted(df, g):
        logger.warning(
            "Global group-by column '%s' is not accepted; falling back to default.",
            g,
        )
        cfg[GC_GROUP_COL] = None

    # --- filters ---
    filters = cfg.get(GC_FILTER) or {}
    clean_filters: Dict = {}
    for col_name, spec in filters.items():
        if col_name not in df.columns:
            logger.warning("Global filter column '%s' not found; skipping filter.", col_name)
            continue

        # Back-compat: list -> "in"
        if isinstance(spec, list):
            vals = [str(v) for v in spec if str(v).strip()]
            if vals:
                clean_filters[col_name] = vals
            continue

        if not isinstance(spec, dict):
            logger.warning("Global filter for '%s' has invalid spec type (%s); skipping.", col_name,
                           type(spec).__name__)
            continue

        op = spec.get("op")
        raw = str(spec.get("value", "")).strip()

        if op not in _ALLOWED_FILTER_OPS:
            logger.warning("Global filter op '%s' for column '%s' is invalid; skipping.", op, col_name)
            continue
        if not raw:
            logger.warning("Global filter value for column '%s' is empty; skipping.", col_name)
            continue
        if op in _NUMERIC_FILTER_OPS and _try_float(raw) is None:
            logger.warning("Global filter op '%s' for column '%s' requires a number; got '%s'. Skipping.", op, col_name,
                           raw)
            continue

        clean_filters[col_name] = {"op": op, "value": raw}

    cfg[GC_FILTER] = clean_filters

    # --- dimensions ---
    dims = cfg.get(GC_DIMENSIONS) or {}
    available_dims = get_all_available_dimensions(df)  # {dim: [values]}
    clean_dims: Dict[str, str] = {}

    for dim, val in dims.items():
        if dim not in available_dims:
            logger.warning("Global dimension '%s' not available; skipping.", dim)
            continue
        if not val or val == "All":
            continue

        # accept "0" as well as "c0"
        vals_set = set(available_dims[dim])
        if val in vals_set or f"{dim}{val}" in vals_set:
            clean_dims[dim] = val
        else:
            logger.warning(
                "Global dimension selection '%s=%s' not found (known: %s); keeping it anyway (may yield empty data).",
                dim, val, ", ".join(list(available_dims[dim])[:10]) + ("..." if len(available_dims[dim]) > 10 else "")
            )
            clean_dims[dim] = val

    cfg[GC_DIMENSIONS] = clean_dims
    return cfg


def _find_candidate_columns(df: pl.DataFrame) -> tuple[list[str], list[str]]:
    if df.height == 0:
        return [], []

    schema = df.schema

    # 1. candidate columns only
    candidates = [
        c for c in df.columns
        if not _DIMENSION_COL_PATTERN.search(c)
        and schema.get(c) not in (pl.List, pl.Struct, pl.Array, pl.Object)
    ]

    if not candidates:
        return [], []

    # 2. compute n_unique in one shot
    try:
        nuniq = dict(
            zip(
                candidates,
                df.select([pl.col(c).n_unique().alias(c) for c in candidates]).row(0),
            )
        )
    except Exception:
        return [], []

    group_cols = []
    filter_cols = []

    for c, n_unique in nuniq.items():
        if n_unique is None or n_unique < 2:
            continue

        filter_cols.append(c)

        if n_unique <= MAX_UNIQUE_GROUP:
            group_cols.append(c)

    return sort_strings_alpha(group_cols), sort_strings_alpha(filter_cols)


def _create_export_btn(label, btn_id, popover_text, mt="mt-2"):
    return [
        html.Button(
            label,
            id=btn_id,
            n_clicks=0,
            className=f"btn btn-outline-dark {mt} w-100",
        ),
        dbc.Popover(
            popover_text,
            target=btn_id,
            trigger="hover",
            placement="top",
            body=True,
            style={"fontSize": "15px", "maxWidth": "300px"},
        ),
    ]


def build_sidebar(df: pl.DataFrame, default_palette_name: str, initial_global_config: Optional[Dict] = None):
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

    initial_global_config = initial_global_config or {GC_GROUP_COL: DEFAULT_REPORT_GROUP_COL, GC_FILTER: {},
                                                      GC_DIMENSIONS: {}}

    # defaults from initial config
    init_group = initial_global_config.get(GC_GROUP_COL)
    init_filters = initial_global_config.get(GC_FILTER) or {}
    init_dims = initial_global_config.get(GC_DIMENSIONS) or {}

    init_filter_col, init_filter_op, init_filter_text = None, "in", ""
    if init_filters:
        init_filter_col, spec = next(iter(init_filters.items()))
        if isinstance(spec, dict):
            init_filter_op = spec.get("op", "in")
            init_filter_text = str(spec.get("value", "") or "")

    candidate_group_cols, candidate_filter_cols = _find_candidate_columns(df)
    available_dims = get_all_available_dimensions(df)

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
                        options=[{"label": NO_GROUPING_LABEL, "value": NO_GROUPING_COL}] +
                                [{"label": c, "value": c} for c in candidate_group_cols],
                        value=init_group,
                        multi=False,
                        clearable=True,
                        placeholder="Choose grouping column",
                        style={"width": "100%"},
                    ),
                    html.Small(
                        f"Default is `imported_path_short` (if exists). Select '{NO_GROUPING_LABEL}' to disable grouping.",
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
                        value=init_filter_col,
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
                        value=init_filter_op,
                        clearable=False,
                        style={"width": "100%", "marginTop": "4px"},
                    ),
                    dcc.Input(
                        id=GLOBAL_FILTER_TEXT_ID,
                        type="text",
                        placeholder="Value (or comma-separated list for 'in')",
                        style={"width": "100%", "marginTop": "4px"},
                        value=init_filter_text,
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
                                        value=init_dims.get(dim, "All"),
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

                    # Statistical significance toggle
                    dbc.Checkbox(
                        id=GLOBAL_SHOW_SIGNIFICANCE_ID,
                        label="Show statistical significance",
                        value=initial_global_config.get(GC_IS_SHOW_SIGNIFICANCE, False),
                        className="mb-1",
                    ),
                    html.Small(
                        "Pairwise comparisons on selected plots",
                        className="text-muted d-block mb-3",
                    ),
                    html.Hr(),

                    dbc.Row([
                        dbc.Col(
                            html.Button(
                                "Apply",
                                id=GLOBAL_APPLY_BUTTON_ID,
                                n_clicks=0,
                                className="btn btn-primary w-100",
                            ), width=6
                        ),
                        dbc.Col(
                            html.Button(
                                "Reset",
                                id=GLOBAL_RESET_BUTTON_ID,
                                n_clicks=0,
                                className="btn btn-outline-danger w-100",
                            ), width=6
                        ),
                    ], className="mt-3 g-2"),

                    html.Div(
                        # We concatenate the lists generated by the helper
                        _create_export_btn(
                            "Export current table (CSV)",
                            EXPORT_CSV_BUTTON_ID,
                            "Download a CSV file containing the data the currently displayed report is based on. Current filters applied.",
                            mt="mt-3"  # Extra margin for the first button
                        ) +
                        _create_export_btn(
                            "Export filtered project",
                            EXPORT_PROJECT_BUTTON_ID,
                            "Export a usable PixelPatrol project with current filters applied. It can be shared and opened as a fully interactive Dash app."
                        ) +
                        _create_export_btn(
                            "Save HTML Snapshot",
                            SAVE_SNAPSHOT_BUTTON_ID,
                            "Save a static html of the current view (not interactive)."
                        ),
                        className="mb-3",
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
            data=initial_global_config,
        ),
    ]

    extra_components = [
        dcc.Download(id=EXPORT_CSV_DOWNLOAD_ID),
        dcc.Download(id=EXPORT_PROJECT_DOWNLOAD_ID),
        dcc.Download(id=SAVE_SNAPSHOT_DOWNLOAD_ID),
    ]

    return sidebar_controls, stores, extra_components


# ---------- LOGIC: CENTRALIZED HELPERS ----------

def _get_dimension_columns(df: pl.DataFrame, dimensions: Dict[str, str]) -> List[str]:
    """
    Get the list of columns that match the selected dimension filters.
    Returns empty list if no dimension filters are active.
    """
    if not dimensions:
        return []

    tokens: List[str] = []
    for dim, val in dimensions.items():
        if not val or val == "All":
            continue
        token = val if val.startswith(dim) else f"{dim}{val}"
        tokens.append(f"_{token}")

    if not tokens:
        return []

    return [c for c in df.columns if all(tok in c for tok in tokens)]


def compute_filtered_row_positions(
        df: pl.DataFrame,
        global_config: Optional[Dict]
) -> Optional[List[int]]:
    """
    Computes the ROW POSITIONS of rows that match
    the global row filters AND dimension filters.

    This is optimized to:
    1. Work on a narrow subset of columns (fast)
    2. Return positions so widgets can use fast positional indexing

    Returns None if no filters are active (meaning use full df).
    Returns empty list if filters result in no matches.
    """
    global_config = global_config or {}
    filters = global_config.get(GC_FILTER) or {}
    dimensions = global_config.get(GC_DIMENSIONS) or {}

    # If no filters and no dimension constraints, return None (use full df)
    if not filters and not dimensions:
        return None

    # Determine which columns we need for filtering
    filter_cols_needed = set(filters.keys()) & set(df.columns)
    dim_cols = _get_dimension_columns(df, dimensions)

    # Build the combined mask expression
    mask = pl.lit(True)

    # 1. Apply standard value filters
    for col_name, spec in filters.items():
        if col_name in df.columns:
            mask = mask & _filter_expr(col_name, spec)

    # 2. Apply dimension filters (any of the dim columns must be non-null)
    if dim_cols:
        dim_mask = pl.any_horizontal([pl.col(c).is_not_null() for c in dim_cols])
        mask = mask & dim_mask
    elif dimensions:
        # Dimensions were specified but no matching columns found
        return []

    # Select only the columns we need, add row index, filter, extract positions
    cols_to_select = list(filter_cols_needed) + dim_cols
    if not cols_to_select:
        # No specific columns needed, just need row positions
        cols_to_select = [df.columns[0]]  # Use any column as placeholder

    # Ensure unique columns
    cols_to_select = list(dict.fromkeys(cols_to_select))

    row_positions = (
        df
        .select(cols_to_select)
        .with_row_index("__row_idx__")
        .filter(mask)
        .get_column("__row_idx__")
        .to_list()
    )

    return row_positions


def resolve_group_column(df: pl.DataFrame, global_config: Optional[Dict]) -> str:
    """Helper to safely extract the active grouping column."""
    global_config = global_config or {}

    group_col = global_config.get(GC_GROUP_COL) or DEFAULT_REPORT_GROUP_COL

    if group_col not in df.columns:
        # Fallback chain: DEFAULT_REPORT_GROUP_COL -> NO_GROUPING_COL -> first column
        if DEFAULT_REPORT_GROUP_COL in df.columns:
            group_col = DEFAULT_REPORT_GROUP_COL
        elif NO_GROUPING_COL in df.columns:
            group_col = NO_GROUPING_COL
        else:
            group_col = df.columns[0]

    return group_col


def prepare_widget_data(
        df: pl.DataFrame,
        subset_row_positions: Optional[List[int]],
        global_config: Dict,
        metric_base: Optional[str] = None
) -> Tuple[pl.DataFrame, str, Optional[str], Optional[str], Optional[List[str]]]:
    """
    The main coordinator for widgets.
    1. Slices the global `df` using `subset_row_positions` (fast positional indexing).
    2. Resolves the correct grouping column.
    3. If `metric_base` is provided, attempts to find the dimension-specific column (e.g. 'area_t0').

    Returns:
        (df_filtered, group_col, resolved_column_name, warning_message, group_order)
    """
    # 1. Filter Rows using fast positional indexing
    if subset_row_positions is not None:
        if len(subset_row_positions) == 0:
            # Empty filter result
            df_filtered = df.head(0)
        else:
            # Fast positional indexing
            df_filtered = df[subset_row_positions]
    else:
        df_filtered = df

    if df_filtered.is_empty():
        return df_filtered, "", None, "No data matches the current filters.", None

    # 2. Resolve Grouping
    group_col = resolve_group_column(df_filtered, global_config)
    df_filtered, group_col, group_order = ensure_discrete_grouping(df_filtered, group_col)

    # 3. Resolve Dimension-Specific Column (if needed)
    resolved_col = None
    warning_msg = None
    dims_selection = global_config.get(GC_DIMENSIONS, {})

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
            # Filter by metric column being non-null
            # Use the same optimized approach: compute mask on narrow df, then apply
            mask_series = df_filtered.select(pl.col(resolved_col).is_not_null()).to_series()
            keep_positions = mask_series.arg_true().to_list()
            if keep_positions:
                df_filtered = df_filtered[keep_positions]
            else:
                df_filtered = df_filtered.head(0)

            if df_filtered.is_empty():
                warning_msg = (
                    "No data matches the selected dimensions for "
                    f"metric '{metric_base}'."
                )

    return df_filtered, group_col, resolved_col, warning_msg, group_order


def apply_global_row_filters_and_grouping(
        df: pl.DataFrame,
        global_config: Optional[Dict],
        default_group_col: str = DEFAULT_REPORT_GROUP_COL,
) -> Tuple[pl.DataFrame, str]:
    """
    Apply global filters and return filtered df with group column.
    Used for exports (CSV, project).
    """
    global_config = global_config or {}

    group_col = global_config.get(GC_GROUP_COL) or default_group_col

    # ensure the chosen group_col exists; otherwise fall back to DEFAULT_GROUP_COL
    if group_col not in df.columns:
        group_col = default_group_col
    if group_col not in df.columns:
        # as a last resort pick the first column
        group_col = df.columns[0]

    # Use the optimized filter function
    row_positions = compute_filtered_row_positions(df, global_config)

    if row_positions is None:
        # No filters, use full df
        return df, group_col
    elif len(row_positions) == 0:
        # Empty result
        return df.head(0), group_col
    else:
        return df[row_positions], group_col


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