import re
from typing import List, Tuple, Dict, Optional, Sequence
import polars as pl
import numpy as np
from collections import defaultdict

from pixel_patrol_base.plugins.processors.histogram_processor import safe_hist_range

# --- Data Helpers ---

def sort_strings_alpha(values: Sequence[str]) -> List[str]:
    """
    Alphabetical, case-insensitive sort. Keeps original strings.
    """
    return sorted([v for v in values if v is not None], key=lambda x: x.lower())


def select_needed_columns(
    df: pl.DataFrame,
    cols_needed: Sequence[str],
    extra_cols: Sequence[str] | None = None,
) -> pl.DataFrame:

    cols = list(cols_needed)
    if extra_cols:
        cols.extend(extra_cols)
    cols = list(dict.fromkeys(cols))
    # keep only existing
    cols = [c for c in cols if c in df.columns]
    return df.select(cols)


def get_sortable_columns(df: pl.DataFrame) -> List[str]:
    """
    Returns a list of sortable column names (numeric, excluding dimension slices).
    """
    slice_pattern = re.compile(r'(_[a-zA-Z]\d+)+$')
    return [
        col for col in df.columns
        if (
                df[col].dtype.is_numeric() and
                not slice_pattern.search(col) and
                df[col].dtype not in [pl.List, pl.Struct, pl.Array]
        )
    ]

# --- Dim Filter Helpers ---

def get_all_available_dimensions(df: pl.DataFrame) -> Dict[str, List[str]]:
    """
        Scans all columns to find unique indices for ANY dimension (e.g. _t0, _z1).
        Returns: {'t': ['0', '1'], ...} sorted numerically.
        Only returns dimensions with >1 unique index (size 0 or 1 are ignored).
        """
    dims_found = defaultdict(set)
    # Strict regex: matches underscore + single lowercase letter + digits (e.g., _t0)
    dim_pat = re.compile(r'_([a-z])(\d+)')

    for col in df.columns:
        for m in dim_pat.finditer(col):
            d = m.group(1)  # already lowercase
            n = m.group(2)
            dims_found[d].add(n)

    sorted_dims = {}
    for k, v in dims_found.items():
        # Only return dimensions that actually allow for filtering (more than 1 slice)
        if len(v) > 1:
            sorted_dims[k] = sorted(list(v), key=int)

    return sorted_dims

def find_best_matching_column(
        columns: List[str],
        base: str,
        selections: Dict[str, str]
) -> Optional[str]:
    """
    Finds the specific column name (e.g., 'mean_intensity_t0') based on user dropdown selections.
    """
    candidates = [c for c in columns if c.startswith(base)]
    if not candidates:
        return None

    # Filter candidates to ensure they contain the selected tokens
    for dim, val in selections.items():
        if val and val != "All":
            # Global controls send indices ("0"), local legacy sends tokens ("t0")
            # We construct the token ensuring it starts with the dimension letter
            token = val
            if not token.startswith(dim):
                token = f"{dim}{val}"

            # The column matches if it contains "_t0" (for example)
            candidates = [c for c in candidates if f"_{token}" in c]

    if candidates:
        return min(candidates, key=len)
    return None

def get_dim_aware_column(
    all_columns: list[str],
    base: str,
    dims_selection: dict | None,
):
    """
    - If user did NOT filter dimensions (all 'All'): return the base metric column if it exists.
    - If user DID filter dimensions: return a matching dim-specific column if found, else None.
    """
    dims_selection = dims_selection or {}

    # True if the user selected ANY concrete dimension instead of "All"
    is_any_dim_filter = any(
        value and value != "All"
        for value in dims_selection.values()
    )

    if not is_any_dim_filter:
        return base if base in all_columns else None

    col = find_best_matching_column(all_columns, base, dims_selection)
    return col  # None means: no data for these dims


def get_all_grouping_cols(base_cols: list[str], group_col: Optional[str]) -> list[str]:
    cols = base_cols.copy()
    if group_col and group_col not in cols:
        cols.append(group_col)
    return cols


def format_selection_title(dims_selection: Dict[str, str]) -> Optional[str]:
    """
    Formats a dictionary of selections (e.g., {'t': '0'}) into a readable title string.
    """
    if not dims_selection:
        return None
    return "Filter: " + ", ".join(f"{k.upper()}={v}" for k, v in sorted(dims_selection.items()))


def parse_metric_dimension_column(
        col_name: str, supported_metrics: List[str]
) -> Optional[Tuple[str, Dict[str, int]]]:
    """
    Reverse parsing: Takes a column name and tells you what metric/dims it represents.
    Used by BaseDynamicTableWidget.
    """
    matched_metric = None
    for metric in sorted(supported_metrics, key=len, reverse=True):
        if col_name.startswith(metric):
            matched_metric = metric
            break

    if not matched_metric:
        return None

    suffix = col_name[len(matched_metric):]
    if not suffix:
        return matched_metric, {}

    dim_matches = list(re.finditer(r"_([a-zA-Z])(\d+)", suffix))
    dims = {m.group(1).lower(): int(m.group(2)) for m in dim_matches}

    if not dims and suffix:
        return None

    return matched_metric, dims


def ensure_discrete_grouping(df: pl.DataFrame, group_col: str) -> Tuple[pl.DataFrame, Optional[List[str]]]:
    """
    Checks if the grouping column is numeric. If so:
    1. Captures the sort order (so "1, 2, 10" doesn't become "1, 10, 2").
    2. Casts the column to Utf8 (String) to force Plotly to use a discrete legend instead of a colorbar.
    """

    # Check if the column is numeric
    if df[group_col].dtype.is_numeric():
        # Get unique values in correct numeric order
        order = [str(x) for x in df.select(pl.col(group_col).unique().sort()).to_series().to_list()]

        # Cast to String so Plotly treats it as a discrete category
        df = df.with_columns(pl.col(group_col).cast(pl.Utf8))
        return df, order

    return df, None


# --- Histogram Math & Aggregation Helpers ---

def compute_histogram_edges(counts, minv, maxv):
    """
    Computes edges, centers, and width for a single histogram entry.
    """
    n = counts.size
    if minv is None or maxv is None:
        edges = np.arange(n + 1).astype(float)
        return edges, edges[:-1], np.diff(edges)

    # Handle integer vs float logic safely
    if float(minv).is_integer() and float(maxv).is_integer():
        sample = np.array([int(minv), int(maxv)], dtype=np.int64)
    else:
        sample = np.array([minv, maxv], dtype=float)

    smin, _smax, max_adj = safe_hist_range(sample)
    smin_f, max_adj_f = float(smin), float(max_adj)
    width = (max_adj_f - smin_f) / float(n)

    # Create edges
    lefts = smin_f + np.arange(n, dtype=float) * width
    edges = np.concatenate([lefts, [smin_f + n * width]])
    centers = lefts
    widths = np.full(n, width, dtype=float)

    return edges, centers, widths


def rebin_histogram(counts, src_edges, tgt_edges):
    """
    Re-bins a histogram count array from source edges to target edges using CDF interpolation.
    """
    counts = np.asarray(counts, float)
    se = np.asarray(src_edges, float)
    te = np.asarray(tgt_edges, float)

    if counts.size == 0 or counts.sum() <= 0:
        return np.zeros(te.size - 1, float)

    # CDF construction
    cdf_src = np.concatenate([[0.0], np.cumsum(counts) / counts.sum()])
    # Interpolate to new edges
    cdf_t = np.interp(te, se, cdf_src, left=0.0, right=1.0)
    # Diff to get PDF
    return np.diff(cdf_t)


def aggregate_histograms_by_group(
    df: pl.DataFrame,
    group_col: str,
    hist_col: str,
    min_col: str,
    max_col: str,
    mode: str = "shape"
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Aggregates histograms per group.
    Returns: { group_name: (x_centers, y_avg_counts) }
    """
    results = {}
    groups = df.select(pl.col(group_col).unique()).to_series().to_list()

    for group_val in groups:
        # Filter logic
        # We drop nulls explicitly for the relevant columns to avoid iteration crashes
        df_group = df.filter(
            (pl.col(group_col) == group_val) &
            pl.col(hist_col).is_not_null()
        )

        if df_group.height == 0:
            continue

        # Prepare target edges
        if mode == "shape":
            # Fixed 0-255 bins
            target_edges = np.linspace(0, 255, 257)
            accumulated = np.zeros(256, dtype=float)
        else:
            # Native range logic
            # Calculate global min/max for this group to define common edges
            gmin = df_group.select(pl.col(min_col).min()).item()
            gmax = df_group.select(pl.col(max_col).max()).item()

            if gmin is None or gmax is None:
                continue

            # Create edges for the group
            n_bins = 256
            # Re-use logic from compute_histogram_edges but just for edges
            smin, _, max_adj = safe_hist_range(np.array([gmin, gmax]))
            width = (float(max_adj) - float(smin)) / float(n_bins)
            target_edges = float(smin) + np.arange(n_bins + 1, dtype=float) * width
            accumulated = np.zeros(n_bins, dtype=float)

        count_files = 0

        # Iterate and rebin
        # We select only needed columns to speed up iteration
        subset = df_group.select([hist_col, min_col, max_col])
        for row in subset.iter_rows():
            c_list, minv, maxv = row
            # Fix: Explicitly check None to avoid ValueError with numpy arrays
            if c_list is None: continue

            c_arr = np.asarray(c_list, dtype=float)
            if c_arr.size == 0 or c_arr.sum() == 0: continue

            # Get source edges for this specific image
            src_edges, _, _ = compute_histogram_edges(c_arr, minv, maxv)

            # Rebin to group common edges
            rebinned = rebin_histogram(c_arr, src_edges, target_edges)
            accumulated += rebinned
            count_files += 1

        if count_files > 0:
            avg_hist = accumulated / count_files
            if avg_hist.sum() > 0:
                avg_hist /= avg_hist.sum()

            centers = 0.5 * (target_edges[:-1] + target_edges[1:])
            results[str(group_val)] = (centers, avg_hist)

    return results
