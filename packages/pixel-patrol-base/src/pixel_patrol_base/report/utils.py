import re
from typing import List, Tuple, Dict, Optional, Any
import polars as pl
from dash import html, dcc


# --- Data Helpers ---

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


# --- Dimension Parsing Helpers ---

def extract_dimension_tokens(columns: List[str], base: str, dims: List[str] = None) -> Dict[str, List[str]]:
    """
    Scans columns for a base metric (e.g. 'histogram') and finds available slices (e.g. t0, z1).
    Returns: {'t': ['t0', 't1'], 'z': ['z0']}
    """
    if dims is None:
        dims = ['t', 'c', 'z', 's']

    tokens = {d: set() for d in dims}
    # Matches base + one or more _L# suffixes
    pattern = re.compile(rf"^{re.escape(base)}((?:_[a-zA-Z]\d+)+)$")
    dim_pat = re.compile(r'_([a-zA-Z])(\d+)')

    for col in columns:
        m = pattern.match(col)
        if not m:
            continue
        suffix = m.group(1)
        for mm in dim_pat.finditer(suffix):
            d = mm.group(1).lower()
            n = mm.group(2)
            if d in tokens:
                tokens[d].add(f"{d}{n}")

                # Sort numerically
    sorted_tokens = {}
    for k, v in tokens.items():
        def sort_key(s):
            m = re.search(r"(\d+)$", s)
            return int(m.group(1)) if m else 0

        sorted_tokens[k] = sorted(list(v), key=sort_key)

    return sorted_tokens


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
    for dim, token in selections.items():
        if token != "All":
            # If user selected 't0', the column MUST contain '_t0'
            candidates = [c for c in candidates if f"_{token}" in c]

    if candidates:
        # Return shortest match (assumption: fewest extra dimensions is best)
        return sorted(candidates, key=len)[0]
    return None


def parse_dynamic_col(
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
        return (matched_metric, {})

    dim_matches = list(re.finditer(r"_([a-zA-Z])(\d+)", suffix))
    dims = {m.group(1).lower(): int(m.group(2)) for m in dim_matches}

    if not dims and suffix:
        return None

    return (matched_metric, dims)


# --- UI Generation Helpers ---
# These generate Dash components based on Logic.
# It is okay to keep them here to keep the Factory focused purely on "Styling".

def build_dimension_dropdown_children(
        columns: List[str],
        base: str,
        id_type: str
) -> Tuple[List[Any], List[str]]:
    """Generates the row of Dropdowns for T/C/Z/S filtering."""
    raw_tokens = extract_dimension_tokens(columns, base)

    children = []
    dims_order = []

    for dim_name in ['t', 'c', 'z', 's']:
        if dim_name in raw_tokens and raw_tokens[dim_name]:
            dims_order.append(dim_name)
            dropdown_id = {"type": id_type, "dim": dim_name}

            # Options: Label="t0", Value="t0"
            options = [{"label": "All", "value": "All"}] + [
                {"label": tok, "value": tok} for tok in raw_tokens[dim_name]
            ]

            children.append(
                html.Div(
                    [
                        html.Label(f"{dim_name.upper()} slice"),
                        dcc.Dropdown(id=dropdown_id, options=options, value="All", clearable=False),
                    ],
                    style={"display": "inline-block", "marginRight": "15px", "width": "120px"}
                )
            )

    return children, dims_order
