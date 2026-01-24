"""
Reusable statistical significance annotations for plots.

This module provides pairwise group comparisons with Bonferroni correction,
designed to be used with any Plotly figure (violin, bar, box, strip, etc.).

Usage
-----
    from pixel_patrol_base.report.stats_annotations import annotate_plot_with_significance

    fig = plot_violin(...)
    fig = annotate_plot_with_significance(
        fig=fig, df=df, value_col=y, group_col=group_col, group_order=groups
    )
"""

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import polars as pl
from scipy.stats import mannwhitneyu
import plotly.graph_objects as go

# =============================================================================
#  CONFIGURATION
# =============================================================================

SIGNIFICANCE_THRESHOLDS = [
    (0.001, "***"),
    (0.01, "**"),
    (0.05, "*"),
    (1.0, "ns"),
]


@dataclass
class PairwiseResult:
    """Result of a single pairwise comparison."""
    group_a: str
    group_b: str
    statistic: float
    p_value_raw: float
    p_value_corrected: float
    significant: bool
    symbol: str


@dataclass
class SignificanceResults:
    """Container for all significance test results."""
    results: List[PairwiseResult]
    n_comparisons: int

    def get_significant_pairs(self) -> List[PairwiseResult]:
        return [r for r in self.results if r.significant]


# =============================================================================
#  STATISTICAL TESTS
# =============================================================================

def _get_significance_symbol(p_value: float) -> str:
    for threshold, symbol in SIGNIFICANCE_THRESHOLDS:
        if p_value < threshold:
            return symbol
    return "ns"


def _apply_bonferroni(p_values: List[float], alpha: float = 0.05) -> Tuple[List[float], List[bool]]:
    if not p_values:
        return [], []
    n = len(p_values)
    corrected = [min(p * n, 1.0) for p in p_values]
    significant = [p < alpha for p in corrected]
    return corrected, significant


def compute_pairwise_significance(
        df: pl.DataFrame,
        value_col: str,
        group_col: str,
        groups: Optional[List[str]] = None,
        alpha: float = 0.05,
        min_samples: int = 3,
) -> Optional[SignificanceResults]:
    """
    Optimized: Single partition call instead of N filter calls.
    """
    if groups is None:
        groups = df[group_col].unique().drop_nulls().sort().to_list()

    # OPTIMIZATION: Single partition instead of N filters
    # Extract only needed columns first
    subset = df.select([group_col, value_col]).drop_nulls()

    # Partition once, then extract arrays
    group_data: Dict[str, np.ndarray] = {}
    valid_groups = []

    # Use group_by to get all groups at once
    grouped = subset.group_by(group_col).agg(pl.col(value_col).alias("values"))
    group_dict = {row[group_col]: row["values"] for row in grouped.iter_rows(named=True)}

    for g in groups:
        if g not in group_dict:
            continue
        data = np.array(group_dict[g])
        if len(data) >= min_samples:
            valid_groups.append(g)
            group_data[g] = data

    if len(valid_groups) < 2:
        return None

    # Rest remains the same...
    pairs = list(combinations(valid_groups, 2))
    raw_results: List[Tuple[str, str, float, float]] = []

    for group_a, group_b in pairs:
        try:
            stat, p_value = mannwhitneyu(
                group_data[group_a], group_data[group_b], alternative="two-sided"
            )
            raw_results.append((group_a, group_b, stat, p_value))
        except ValueError:
            raw_results.append((group_a, group_b, 0.0, 1.0))

    p_values_raw = [r[3] for r in raw_results]
    p_values_corrected, significant = _apply_bonferroni(p_values_raw, alpha)

    results = []
    for i, (group_a, group_b, stat, p_raw) in enumerate(raw_results):
        p_corr = p_values_corrected[i]
        results.append(PairwiseResult(
            group_a=group_a,
            group_b=group_b,
            statistic=stat,
            p_value_raw=p_raw,
            p_value_corrected=p_corr,
            significant=significant[i],
            symbol=_get_significance_symbol(p_corr),
        ))

    return SignificanceResults(results=results, n_comparisons=len(pairs))


# =============================================================================
#  PLOT ANNOTATION
# =============================================================================

def _add_bracket(
    fig: go.Figure,
    x_left: float,
    x_right: float,
    y_line: float,
    tick_height: float,
    symbol: str,
    font_size: int = 12,
) -> None:
    """Draw a single significance bracket with symbol."""
    # Left vertical tick
    fig.add_shape(
        type="line",
        x0=x_left, x1=x_left,
        y0=y_line - tick_height, y1=y_line,
        line=dict(color="black", width=1),
        xref="x", yref="y",
    )

    # Horizontal bar
    fig.add_shape(
        type="line",
        x0=x_left, x1=x_right,
        y0=y_line, y1=y_line,
        line=dict(color="black", width=1),
        xref="x", yref="y",
    )

    # Right vertical tick
    fig.add_shape(
        type="line",
        x0=x_right, x1=x_right,
        y0=y_line - tick_height, y1=y_line,
        line=dict(color="black", width=1),
        xref="x", yref="y",
    )

    # Significance symbol centered above bracket
    fig.add_annotation(
        x=(x_left + x_right) / 2,
        y=y_line + tick_height * 0.5,
        text=symbol,
        showarrow=False,
        font=dict(size=font_size, color="black"),
        xref="x", yref="y",
    )


def add_significance_annotations(
        fig: go.Figure,
        significance_results: Optional[SignificanceResults],
        group_order: List[str],
        y_max: float,
        show_ns: bool = False,
        bracket_height_frac: float = 0.04,
        bracket_gap_frac: float = 0.06,
        font_size: int = 12,
) -> go.Figure:
    """Add significance brackets to a figure."""
    if significance_results is None:
        return fig

    results_to_show = significance_results.results
    if not show_ns:
        results_to_show = [r for r in results_to_show if r.significant]

    if not results_to_show:
        return fig

    # Map group names to x-axis positions (0, 1, 2, ...)
    x_positions = {g: float(i) for i, g in enumerate(group_order)}

    # Calculate bracket dimensions
    tick_h = y_max * bracket_height_frac
    gap = y_max * bracket_gap_frac

    # Sort by span width so narrower brackets are drawn lower
    def span_width(r: PairwiseResult) -> int:
        try:
            return abs(group_order.index(r.group_b) - group_order.index(r.group_a))
        except ValueError:
            return 999

    results_to_show = sorted(results_to_show, key=span_width)

    # Start brackets above the data
    current_y = y_max + gap

    for result in results_to_show:
        if result.group_a not in x_positions or result.group_b not in x_positions:
            continue

        x_left = x_positions[result.group_a]
        x_right = x_positions[result.group_b]

        # Ensure left < right
        if x_left > x_right:
            x_left, x_right = x_right, x_left

        _add_bracket(
            fig=fig,
            x_left=x_left,
            x_right=x_right,
            y_line=current_y,
            tick_height=tick_h,
            symbol=result.symbol,
            font_size=font_size,
        )

        # Move up for next bracket
        current_y += gap + tick_h

    # Extend y-axis to show all brackets
    fig.update_layout(yaxis=dict(range=[None, current_y + gap]))
    return fig


def annotate_plot_with_significance(
        fig: go.Figure,
        df: pl.DataFrame,
        value_col: str,
        group_col: str,
        group_order: List[str],
        annotate_significance: bool = True,
        show_ns: bool = False,
        alpha: float = 0.05,
        min_samples: int = 3,
) -> go.Figure:
    """
    High-level function to add significance annotations to any grouped plot.
    """
    if not annotate_significance or len(group_order) < 2:
        return fig

    sig_results = compute_pairwise_significance(
        df=df, value_col=value_col, group_col=group_col,
        groups=group_order, alpha=alpha, min_samples=min_samples,
    )

    if sig_results is None:
        return fig

    y_data = df[value_col].drop_nulls().to_numpy()
    y_max = float(np.max(y_data)) if len(y_data) > 0 else 1.0

    return add_significance_annotations(
        fig=fig, significance_results=sig_results,
        group_order=group_order, y_max=y_max, show_ns=show_ns,
    )