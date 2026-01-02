import logging
from typing import Dict, List, Tuple, Any
from itertools import chain, combinations

import dask.array as da
import numpy as np
import polars as pl
import dask.array as da
import threading

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


def _hist_func(arr: da.Array | np.ndarray, bins: int) -> Dict[str, Any]:
    """
    Calculates a histogram on a Dask or NumPy array.
    For Dask arrays this uses Dask's histogram to avoid pulling the full chunk into memory.
    Returns a dict with:
      - "counts": numpy.ndarray (dtype=np.int64)
      - "min": Python float
      - "max": Python float
    """
    # Compute min/max and adjusted upper bound with shared helper
    min_val, max_val, max_adj_val = safe_hist_range(arr)

    if isinstance(arr, da.Array):
        # Use Dask's histogram function and compute the counts (we don't need edges to be stored)
        counts, edges = da.histogram(arr, bins=bins, range=(min_val, max_adj_val))
        computed_counts, _ = da.compute(counts, edges)
        counts_arr = np.asarray(computed_counts, dtype=np.int64)
    else:
        # NumPy path: use numpy's histogram for in-memory arrays
        counts_np, _ = np.histogram(arr, bins=bins, range=(min_val, max_adj_val))
        counts_arr = np.asarray(counts_np, dtype=np.int64)  # TODO: FIXME: test expects a list instead array

    return {"counts": counts_arr, "min": float(min_val), "max": float(max_val)}


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
        Calculates histograms for all levels of the dimensional hierarchy by using
        calculate_sliced_stats to vectorize per-slice computations instead of manual loops.
        """
        data = art.data
        dim_order = art.dim_order

        # For empty arrays, return zeroed counts and default 0..255 range so the image is visible in comparisons
        if getattr(data, "size", 0) == 0:
            return {"histogram_counts": np.array([]), "histogram_min": 0.0, "histogram_max": 0.0}


        # Metric functions operate on 2D numpy planes provided by apply_gufunc.
        # To avoid calling _hist_func three times per plane, compute once and reuse
        # results. Use thread-local storage to remain safe when Dask runs in threads.
        _thread_local = threading.local()
        def _compute_once(plane: np.ndarray):
            pid = id(plane)
            if getattr(_thread_local, "last_id", None) != pid:
                _thread_local.last_id = pid
                _thread_local.last_result = _hist_func(plane, bins=256)
            return _thread_local.last_result

        def _counts_fn(plane: np.ndarray):
            return _compute_once(plane)["counts"]

        def _min_fn(plane: np.ndarray):
            return _compute_once(plane)["min"]

        def _max_fn(plane: np.ndarray):
            return _compute_once(plane)["max"]

        metrics = {
            "histogram_counts": _counts_fn,
            "histogram_min": _min_fn,
            "histogram_max": _max_fn,
        }

        # Aggregators: sum counts per-bin, and take min/max for boundaries
        aggregators = {
            "histogram_counts": np.sum,
            "histogram_min": np.min,
            "histogram_max": np.max,
        }

        return calculate_sliced_stats(data, dim_order, metrics, aggregators)
