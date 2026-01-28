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


# =============================================================================
# Module-level cache for expensive computations
# =============================================================================

class _PrepareDataCache:
    """
    Module-level cache to avoid redundant computations across widgets.

    When multiple widgets fire callbacks with the same inputs, we compute once
    and return cached results for subsequent calls in the same "batch".

    The cache is invalidated when inputs change (new subset_positions or global_config).
    """

    def __init__(self):
        self._cache_key: Optional[tuple] = None
        self._cached_source_df: Optional[pl.DataFrame] = None
        self._cached_df: Optional[pl.DataFrame] = None
        self._cached_group_col: Optional[str] = None
        self._cached_group_order: Optional[List[str]] = None

    def get_or_compute(
            self,
            df: pl.DataFrame,
            subset_row_positions: Optional[List[int]],
            global_config: Dict,
    ) -> Tuple[pl.DataFrame, str, Optional[List[str]]]:
        """
        Returns (df_filtered_with_grouping, group_col, group_order).

        This is the expensive base computation that all widgets need.
        Metric resolution happens separately per widget.
        """
        # Build cache key
        cache_key = (
            tuple(subset_row_positions) if subset_row_positions else None,
            tuple(sorted((global_config or {}).items())),
        )

        if (self._cached_source_df is df and
                self._cache_key == cache_key and
                self._cached_df is not None):
            return self._cached_df, self._cached_group_col, self._cached_group_order

        # Compute fresh
        # 1. Filter rows
        if subset_row_positions is not None:
            if len(subset_row_positions) == 0:
                df_filtered = df.head(0)
            else:
                df_filtered = df[subset_row_positions]
        else:
            df_filtered = df

        if df_filtered.is_empty():
            self._cache_key = cache_key
            self._cached_df = df_filtered
            self._cached_source_df = df
            self._cached_group_col = ""
            self._cached_group_order = None
            return df_filtered, "", None

        # 2. Resolve and apply grouping
        group_col = resolve_group_column(df_filtered, global_config)
        df_filtered, group_col, group_order = ensure_discrete_grouping(df_filtered, group_col)

        # Cache results
        self._cache_key = cache_key
        self._cached_df = df_filtered
        self._cached_source_df = df
        self._cached_group_col = group_col
        self._cached_group_order = group_order

        return df_filtered, group_col, group_order

    def invalidate(self):
        """Clear the cache (called when global state changes)."""
        self._cache_key = None
        self._cached_df = None
        self._cached_source_df = None
        self._cached_group_col = None
        self._cached_group_order = None


# Global cache instance
_prepare_cache = _PrepareDataCache()


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
    except (KeyError, pl.exceptions.ComputeError, pl.exceptions.SchemaError):
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

        # accept "0" NOT "c0"
        vals_set = set(available_dims[dim])
        if val in vals_set or f"{dim}{val}" in vals_set:
            clean_dims[dim] = val
        else:
            logger.warning(
                "Global dimension selection '%s=%s' not found (known: %s); skipping.",
                dim, val, ", ".join(list(available_dims[dim])[:10]) + ("..." if len(available_dims[dim]) > 10 else "")
            )

    cfg[GC_DIMENSIONS] = clean_dims
    return cfg


def _find_candidate_columns(df: pl.DataFrame) -> tuple[list[str], list[str]]:
    if df.height == 0:
        return [], []

    schema = df.schema

    # 1. candidate columns only
    candidates = []
    for c in df.columns:
        if _DIMENSION_COL_PATTERN.search(c):
            continue

        dtype = schema.get(c)
        if dtype is None:
            continue

        if dtype.base_type() in (pl.List, pl.Struct, pl.Array, pl.Object):
            continue

        candidates.append(c)

    if not candidates:
        return [], []

    # 2. compute n_unique in one shot
    try:
        n_uniq = dict(zip(candidates, df.select([pl.col(c).n_unique().alias(c) for c in candidates]).row(0),))
    except (pl.exceptions.ComputeError, pl.exceptions.SchemaError):
        return [], []

    group_cols = []
    filter_cols = []

    for c, n_unique in n_uniq.items():
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
        **Operators:**
        - `contains` / `not_contains`: text substring
        - `eq`: exact match (number or string)
        - `gt`, `ge`, `lt`, `le`: numeric comparisons
        - `in`: comma-separated list
        """)

    # Available palettes
    palettes = sorted(
        [p for p in plt.colormaps() if not p.endswith("_r")], key=str.lower
    )
    palette_options = [{"label": p, "value": p} for p in palettes]

    # 2. Find columns valid for grouping & filtering (single efficient pass)
    group_col_options, filter_col_options = _find_candidate_columns(df)

    # 3. Add "No grouping" option for group columns
    group_col_options_with_none = [
                                      {"label": NO_GROUPING_LABEL, "value": NO_GROUPING_COL}
                                  ] + [{"label": c, "value": c} for c in group_col_options]

    filter_col_options = [{"label": c, "value": c} for c in filter_col_options]

    # 4. DIMENSION SLIDERS - inline on same row
    dim_info = get_all_available_dimensions(df)
    dim_dropdowns = []

    for dim_name in sorted(dim_info.keys()):
        indices = dim_info[dim_name]
        options = [{"label": "All", "value": "All"}] + [
            {"label": idx, "value": idx} for idx in indices
        ]
        dim_dropdowns.append(
            html.Div([
                html.Label(f"{dim_name.upper()}:", style={"fontWeight": "500", "fontSize": "12px"}),
                dcc.Dropdown(
                    id={"type": GLOBAL_DIM_FILTER_TYPE, "index": dim_name},
                    options=options,
                    value="All",
                    clearable=False,
                )
            ], style={"display": "inline-block", "width": "70px", "marginRight": "5px"})
        )

    # 5. Build initial values from initial_global_config
    init_cfg = init_global_config(df, initial_global_config)
    init_group_col = init_cfg.get(GC_GROUP_COL) or DEFAULT_REPORT_GROUP_COL
    init_dims = init_cfg.get(GC_DIMENSIONS) or {}

    # pre-select dims in dropdowns
    for dd in dim_dropdowns:
        dd_id = dd.children[1].id
        dim_name = dd_id["index"]
        if dim_name in init_dims:
            dd.children[1].value = init_dims[dim_name]

    sidebar = dbc.Card(
        dbc.CardBody([
            # --- Palette ---
            html.Label("Color Palette", style={"fontWeight": "bold"}),
            dcc.Dropdown(
                id=PALETTE_SELECTOR_ID,
                options=palette_options,
                value=default_palette_name,
                clearable=False,
                style={"marginBottom": "8px"},
            ),
            html.Hr(style={"margin": "8px 0"}),

            # --- Grouping ---
            html.Label("Group By Column", style={"fontWeight": "bold"}),
            dcc.Dropdown(
                id=GLOBAL_GROUPBY_COLS_ID,
                options=group_col_options_with_none,
                value=init_group_col,
                clearable=False,
                style={"marginBottom": "8px"},
            ),
            html.Hr(style={"margin": "8px 0"}),

            # --- Dimensions ---
            html.Label("Dimension Selection", style={"fontWeight": "bold", "marginBottom": "5px"}),
            html.Div(dim_dropdowns if dim_dropdowns else [html.Small("No dimensions found.")],
                     style={"marginBottom": "8px"}),
            html.Hr(style={"margin": "8px 0"}),

            # --- Row Filtering ---
            html.Div([
                html.Label("Filter Rows", style={"fontWeight": "bold"}),
                create_info_icon("filter-help", filter_help_md),
            ], className="d-flex align-items-center", style={"marginBottom": "5px"}),

            dcc.Dropdown(
                id=GLOBAL_FILTER_COLUMN_ID,
                options=filter_col_options,
                placeholder="Column...",
                clearable=True,
                style={"marginBottom": "5px"},
            ),
            dcc.Dropdown(
                id=GLOBAL_FILTER_OP_ID,
                options=[
                    {"label": "contains", "value": "contains"},
                    {"label": "not contains", "value": "not_contains"},
                    {"label": "equals", "value": "eq"},
                    {"label": ">", "value": "gt"},
                    {"label": ">=", "value": "ge"},
                    {"label": "<", "value": "lt"},
                    {"label": "<=", "value": "le"},
                    {"label": "in (comma-sep)", "value": "in"},
                ],
                placeholder="Operator...",
                clearable=True,
                style={"marginBottom": "5px"},
            ),
            dcc.Input(
                id=GLOBAL_FILTER_TEXT_ID,
                type="text",
                placeholder="Value...",
                debounce=True,
                style={"width": "100%", "marginBottom": "8px"},
            ),
            html.Hr(style={"margin": "8px 0"}),

            # --- Significance toggle ---
            dcc.Checklist(
                id=GLOBAL_SHOW_SIGNIFICANCE_ID,
                options=[{"label": " Show significance", "value": True}],
                value=[],
                style={"marginBottom": "10px"},
            ),

            # --- Apply / Reset on same row ---
            html.Div([
                html.Button("Apply", id=GLOBAL_APPLY_BUTTON_ID, n_clicks=0,
                            className="btn btn-primary", style={"flex": "1", "marginRight": "5px"}),
                html.Button("Reset", id=GLOBAL_RESET_BUTTON_ID, n_clicks=0,
                            className="btn btn-secondary", style={"flex": "1"}),
            ], style={"display": "flex"}),
            html.Hr(style={"margin": "10px 0"}),

            # --- Export buttons ---
            *_create_export_btn("📥 Export CSV", EXPORT_CSV_BUTTON_ID,
                                "Download filtered data as CSV"),
            *_create_export_btn("📦 Export Project", EXPORT_PROJECT_BUTTON_ID,
                                "Download filtered project as .zip"),
            *_create_export_btn("📸 Save Snapshot", SAVE_SNAPSHOT_BUTTON_ID,
                                "Save current view as image"),
        ]),
        style={"position": "sticky", "top": "10px"},
    )

    stores = [
        dcc.Store(id=GLOBAL_CONFIG_STORE_ID, data=init_cfg),
    ]

    extra = [
        dcc.Download(id=EXPORT_CSV_DOWNLOAD_ID),
        dcc.Download(id=EXPORT_PROJECT_DOWNLOAD_ID),
        dcc.Download(id=SAVE_SNAPSHOT_DOWNLOAD_ID),
    ]

    return sidebar, stores, extra


def _get_dimension_columns(df: pl.DataFrame, dimensions: Dict[str, str]) -> List[str]:
    """
    Given dimension selections like {'t': '0', 'c': '1'},
    find all columns that match ALL selected dimensions.
    """
    if not dimensions:
        return []

    matching_cols = []
    for col in df.columns:
        matches_all = True
        for dim, val in dimensions.items():
            if not val or val == "All":
                continue
            # Build the token (e.g., "_t0" or "_c1")
            token = f"_{dim}{val}" if not val.startswith(dim) else f"_{val}"
            if token not in col:
                matches_all = False
                break
        if matches_all and any(f"_{dim}" in col for dim in dimensions.keys()):
            matching_cols.append(col)

    return matching_cols


def compute_filtered_row_positions(
        df: pl.DataFrame,
        global_config: Optional[Dict],
) -> Optional[List[int]]:
    """
    Returns row POSITIONS (indices) after applying global filters.
    Returns None if no filtering needed (use full df).
    Returns [] if filter results in empty set.

    Invalidate the prepare cache when this is called,
    since it means global config has changed.
    """
    # Invalidate cache when filters change
    _prepare_cache.invalidate()

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

    Uses module-level cache for the base computation (filtering + grouping).
    Metric resolution still happens per widget since it varies.

    Returns:
        (df_filtered, group_col, resolved_column_name, warning_message, group_order)
    """
    # Use cached base computation
    df_filtered, group_col, group_order = _prepare_cache.get_or_compute(
        df, subset_row_positions, global_config
    )

    if df_filtered.is_empty():
        return df_filtered, group_col or "", None, "No data matches the current filters.", group_order

    # Metric resolution (per widget, not cached)
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
            df_filtered = df_filtered.filter(pl.col(resolved_col).is_not_null())

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