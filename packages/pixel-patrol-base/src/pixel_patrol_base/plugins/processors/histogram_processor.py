import logging
from typing import Tuple

import dask.array as da
import numpy as np
import polars as pl

from pixel_patrol_base.core.contracts import ProcessResult
from pixel_patrol_base.core.record import Record
from pixel_patrol_base.core.specs import RecordSpec
from pixel_patrol_base.utils.array_utils import calculate_sliced_stats

logger = logging.getLogger(__name__)


def safe_hist_range(x: da.Array | np.ndarray) -> Tuple[float, float, float]:
    """
    Ensures the maximum is included in the last bin while having a right-bound that is strictly greater than the maximum.
    Args:
        x: Input image as array (Dask or NumPy)
    Returns (min, max, max_adj) with correct right-edge handling for histograms.
    """
    # compute min/max without pulling full arrays where possible
    if isinstance(x, da.Array):
        min_val, max_val = da.compute(x.min(), x.max())
        min_val, max_val = np.float64(min_val), np.float64(max_val)  # cast to float64 to avoid overflows
    else:
        min_val, max_val = np.float64(np.min(x)), np.float64(np.max(x))

    # If the underlying data type is uint8, set the minimum to 0 so we
    # use the full 0..255 bin range for display/processing. This ensures
    # bin-width=1 and use of all bins while remaining
    # flexible for other integer types (e.g., int16).
    try:
        dtype = x.dtype
    except Exception:
        dtype = None

    if dtype is not None and np.dtype(dtype) == np.dtype('uint8'):
        min_val, max_val, max_adj = 0.0, 255.0, 256.0
        return min_val, max_val, max_adj

    # add +1 to include the max value as its own bin for integer types
    if dtype is not None and np.issubdtype(dtype, np.integer):
        max_adj = max_val + 1.0
    else:
        # in case of a blank image, nextafter would be too small to span a range, so we need some space between min and max
        if min_val == max_val:
            max_adj = max_val + 1.0
        else:
            # make the upper bound slightly greater than max again
            max_adj = np.nextafter(max_val, np.inf)
    return min_val, max_val, max_adj


class HistogramProcessor:
    """
    Record-first processor that extracts a full hierarchy of pixel-value histograms.
    Histograms are recalculated for the full image and for every possible combination of slices.
    """

    NAME = "histogram"
    INPUT = RecordSpec(axes={"X", "Y"}, kinds={"intensity"}, capabilities={"spatial-2d"})
    OUTPUT = "features"

    OUTPUT_SCHEMA = {
        "histogram_counts": pl.List(pl.Int64),
        "histogram_min": pl.Float64,
        "histogram_max": pl.Float64,
    }
    OUTPUT_SCHEMA_PATTERNS = [
        (r"^(?:histogram)_counts_.*$", pl.List(pl.Int64)),
        (r"^(?:histogram)_min_.*$", pl.Float64),
        (r"^(?:histogram)_max_.*$", pl.Float64),
    ]

    def run(self, art: Record) -> ProcessResult:
        """
        Calculates histograms for all levels of the dimensional hierarchy using
        the generic sliced statistics calculator.
        """
        data = art.data

        if data.size == 0:
            return {
                "histogram_counts": [0] * 256,
                "histogram_min": 0.0,
                "histogram_max": 255.0
            }

        # 1. Determine Global Range for Binning
        # We must use a global range so that histogram counts from different slices
        # align to the same bins, allowing them to be summed (aggregated).
        global_min, _, global_max_adj = safe_hist_range(data)
        hist_bins = 256
        hist_range = (global_min, global_max_adj)

        # 2. Define Metric Functions (Applied to each 2D slice)

        def calc_counts(plane: np.ndarray) -> np.ndarray:
            # Returns counts as a numpy array to allow vector addition during aggregation.
            counts, _ = np.histogram(plane, bins=hist_bins, range=hist_range)
            return counts.astype(np.int64)

        def calc_min(plane: np.ndarray) -> float:
            # Use safe_hist_range to ensure we respect the uint8 override (0.0)
            # or the actual min for other types.
            min_val, _, _ = safe_hist_range(plane)
            return float(min_val)

        def calc_max(plane: np.ndarray) -> float:
            # Use safe_hist_range to ensure we respect the uint8 override (255.0)
            # or the actual max for other types.
            _, max_val, _ = safe_hist_range(plane)
            return float(max_val)

        metric_fns = {
            "histogram_counts": calc_counts,
            "histogram_min": calc_min,
            "histogram_max": calc_max
        }

        # 3. Define Aggregation Functions
        agg_fns = {
            "histogram_counts": np.sum,  # Sum counts from slices to get parent histogram
            "histogram_min": np.min,  # Parent min is the min of slice mins
            "histogram_max": np.max  # Parent max is the max of slice maxs
        }

        # 4. Calculate Statistics
        results = calculate_sliced_stats(data, art.dim_order, metric_fns, agg_fns)

        # 5. Format Output
        # Convert numpy arrays to python lists for the schema
        final_features = {}
        for key, value in results.items():
            if "counts" in key and isinstance(value, np.ndarray):
                final_features[key] = value.tolist()
            else:
                final_features[key] = value

        return final_features