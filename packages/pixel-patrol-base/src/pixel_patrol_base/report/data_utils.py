import re
from typing import List, Tuple, Dict, Optional, Sequence
import polars as pl
import numpy as np
from collections import defaultdict

from pixel_patrol_base.plugins.processors.histogram_processor import safe_hist_range
from pixel_patrol_base.report.constants import GROUPING_COL_PREFIX, MISSING_LABEL


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


def prettify_col_name(col: Optional[str]) -> Optional[str]:
    if col and col.startswith(GROUPING_COL_PREFIX):
        return col[len(GROUPING_COL_PREFIX):]
    return col


def ensure_discrete_grouping(df: pl.DataFrame, group_col: str) -> Tuple[pl.DataFrame, str, Optional[List[str]]]:
    """
    Converts a column to discrete string labels suitable for grouping/display.
    Applies smart shortening based on value content:
    - Floats: minimum 2 decimal places, more if needed for uniqueness
    - Paths: only distinguishing path suffixes (with ... prefix)
    - Integers: just stringify
    - Strings: shorten common prefix/suffix if long enough (with ... indicator)
    Null values become MISSING_LABEL.
    """
    new_group_col = f'{GROUPING_COL_PREFIX}{group_col}'

    # Get unique values, check for nulls
    unique_vals = df.select(pl.col(group_col).unique().sort()).to_series().to_list()
    has_nulls = None in unique_vals
    unique_non_null = [v for v in unique_vals if v is not None]

    if not unique_non_null:
        # All nulls
        df = df.with_columns(pl.lit(MISSING_LABEL).alias(new_group_col))
        return df, new_group_col, [MISSING_LABEL]

    # Build mapping from original value -> display label
    if df[group_col].dtype.is_float():
        label_map = _floats_to_categorical(unique_non_null)
    elif df[group_col].dtype.is_integer():
        label_map = {v: str(v) for v in unique_non_null}
    elif _looks_like_paths(unique_non_null):
        label_map = _paths_to_categorical(unique_non_null)
    else:
        # strings - try common prefix/suffix stripping
        label_map = _strings_to_categorical(unique_non_null)

    # Build sort order: numeric sort for numbers, alpha for strings
    if df[group_col].dtype.is_numeric():
        sorted_orig = sorted(unique_non_null)
    else:
        sorted_orig = sorted(unique_non_null, key=lambda x: str(x).lower())

    order = [label_map[v] for v in sorted_orig]
    if has_nulls:
        order.append(MISSING_LABEL)

    # Apply mapping via replace dict (convert keys to string for polars replace)
    # We need to cast to string first, then replace, then handle nulls
    str_map = {str(k): v for k, v in label_map.items()}

    df = df.with_columns(
        pl.col(group_col)
        .cast(pl.Utf8)
        .replace(str_map)
        .fill_null(MISSING_LABEL)
        .alias(new_group_col)
    )

    return df, new_group_col, order


def _floats_to_categorical(values: List[float]) -> Dict[float, str]:
    """
    Format floats for display:
    - If all whole numbers: display as integers
    - Otherwise: minimum 2 decimal places, more if needed for uniqueness
    """
    # Check if these are "integer floats" (e.g., 1.0, 2.0, 3.0)
    all_whole = all(float(v) == int(v) for v in values)

    if all_whole:
        return {v: str(int(v)) for v in values}

    # Start with 2 decimal places minimum, increase if needed for uniqueness
    for decimals in range(2, 15):
        formatted = {v: f"{v:.{decimals}f}" for v in values}
        if len(set(formatted.values())) == len(values):
            return formatted

    # Fallback - full precision
    return {v: repr(v) for v in values}


def _looks_like_paths(values: List) -> bool:
    """Check if all values appear to be file/directory paths."""
    if not values:
        return False
    str_vals = [str(v) for v in values]
    return all('/' in s or '\\' in s for s in str_vals)


def _paths_to_categorical(values: List) -> Dict:
    """
    Keep only the distinguishing suffix of paths (full folder names).
    Adds '...' prefix to indicate truncation.
    """
    from pathlib import PurePosixPath, PureWindowsPath

    str_vals = [str(v) for v in values]

    if len(str_vals) < 2:
        return {v: str(v) for v in values}

    # Detect separator
    sep = '/' if any('/' in s for s in str_vals) else '\\'
    PathClass = PurePosixPath if sep == '/' else PureWindowsPath

    # Split into parts
    parsed = [PathClass(s).parts for s in str_vals]

    if not parsed or not parsed[0]:
        return {v: str(v) for v in values}

    # Find how many trailing parts we need for uniqueness
    max_parts = max(len(p) for p in parsed)

    for n_parts in range(1, max_parts + 1):
        suffixes = [sep.join(p[-n_parts:]) if len(p) >= n_parts else sep.join(p) for p in parsed]
        if len(set(suffixes)) == len(values):
            # Check if we actually shortened anything
            result = {}
            for orig, suf, parts in zip(values, suffixes, parsed):
                if len(parts) > n_parts:
                    result[orig] = f"...{sep}{suf}"
                else:
                    result[orig] = suf
            return result

    # Fallback
    return {v: str(v) for v in values}


def _strings_to_categorical(values: List, min_strip_len: int = 10) -> Dict:
    """
    Strip common prefix/suffix from strings, but only if:
    - The common part is long enough to be worth removing (>=10 chars)
    - We indicate truncation with '...'
    """
    str_vals = [str(v) for v in values]

    if len(str_vals) < 2:
        return {v: str(v) for v in values}

    # Find common prefix/suffix
    prefix = _common_prefix(str_vals)
    suffix = _common_suffix(str_vals)

    # Only strip if the common part is substantial
    strip_prefix = len(prefix) >= min_strip_len
    strip_suffix = len(suffix) >= min_strip_len

    if not strip_prefix and not strip_suffix:
        return {v: str(v) for v in values}

    shortened = []
    for s in str_vals:
        result = s
        prefix_indicator = ""
        suffix_indicator = ""

        if strip_prefix:
            result = result[len(prefix):]
            prefix_indicator = "..."

        if strip_suffix and len(result) > len(suffix):
            result = result[:-len(suffix)]
            suffix_indicator = "..."

        final = f"{prefix_indicator}{result}{suffix_indicator}"
        # Don't allow empty or just ellipsis
        shortened.append(final if result else s)

    # Only use shortened if all still unique
    if len(set(shortened)) == len(values):
        return {orig: short for orig, short in zip(values, shortened)}

    return {v: str(v) for v in values}


def _common_prefix(strings: List[str]) -> str:
    """Find longest common prefix."""
    if not strings:
        return ""
    prefix = strings[0]
    for s in strings[1:]:
        while not s.startswith(prefix):
            prefix = prefix[:-1]
            if not prefix:
                return ""
    return prefix


def _common_suffix(strings: List[str]) -> str:
    """Find longest common suffix."""
    if not strings:
        return ""
    suffix = strings[0]
    for s in strings[1:]:
        while not s.endswith(suffix):
            suffix = suffix[1:]
            if not suffix:
                return ""
    return suffix


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

    # Get unique groups
    groups = df.select(pl.col(group_col).unique()).to_series().to_list()

    # Extract all needed columns once as lists
    all_groups = df.get_column(group_col).to_list()
    all_hists = df.get_column(hist_col).to_list()
    all_mins = df.get_column(min_col).to_list() if min_col in df.columns else [None] * len(all_groups)
    all_maxs = df.get_column(max_col).to_list() if max_col in df.columns else [None] * len(all_groups)

    # Pre-index data by group
    group_indices: Dict[str, List[int]] = defaultdict(list)
    for i, g in enumerate(all_groups):
        if all_hists[i] is not None:  # Only include rows with valid histograms
            group_indices[g].append(i)

    for group_val in groups:
        indices = group_indices.get(group_val, [])
        if not indices:
            continue

        # Prepare target edges based on mode
        if mode == "shape":
            target_edges = np.linspace(0, 255, 257)
            accumulated = np.zeros(256, dtype=float)
        else:
            # Native range: need to find global min/max for this group
            group_mins = [all_mins[i] for i in indices if all_mins[i] is not None]
            group_maxs = [all_maxs[i] for i in indices if all_maxs[i] is not None]

            if not group_mins or not group_maxs:
                continue

            gmin = min(group_mins)
            gmax = max(group_maxs)

            n_bins = 256
            smin, _, max_adj = safe_hist_range(np.array([gmin, gmax]))
            width = (float(max_adj) - float(smin)) / float(n_bins)
            target_edges = float(smin) + np.arange(n_bins + 1, dtype=float) * width
            accumulated = np.zeros(n_bins, dtype=float)

        count_files = 0

        # Process histograms for this group
        for idx in indices:
            c_list = all_hists[idx]
            minv = all_mins[idx]
            maxv = all_maxs[idx]

            if c_list is None:
                continue

            c_arr = np.asarray(c_list, dtype=float)
            if c_arr.size == 0 or c_arr.sum() == 0:
                continue

            # Get source edges
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