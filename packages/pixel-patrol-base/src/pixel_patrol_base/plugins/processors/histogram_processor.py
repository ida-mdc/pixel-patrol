import logging
from typing import Dict, List, Tuple
from itertools import chain, combinations

import numpy as np
import polars as pl
import dask.array as da

from pixel_patrol_base.core.record import Record
from pixel_patrol_base.core.contracts import ProcessResult
from pixel_patrol_base.core.specs import RecordSpec

logger = logging.getLogger(__name__)


def safe_hist_range(x: da.Array | np.ndarray) -> Tuple[float, float]:
    """
    Ensures the maximum is included in the last bin while having a right-bound that is strictly greater than the maximum.
    Args:
        x: Input image as array (Dask or NumPy)
    Returns (min, max, max_adj) with correct right-edge handling for histograms. 
    """
    if isinstance(x, da.Array):
        min, max = da.compute(x.min(), x.max())
        min, max = np.float64(min), np.float64(max) # cast to float64 to avoid overflows
    else:
        min, max = np.float64(np.min(x)), np.float64(np.max(x)) 

    # add +1 to include the max value as its own bin
    if np.issubdtype(x.dtype, np.integer):
        max_adj = max + 1
    else:
        # in case of a blank image, nextafter would be too small to span a range, so we need some space between min and max
        if min == max:
            max_adj = max + 1.
        # make the upper bound slightly greater than max again
        max_adj = np.nextafter(max, np.inf)
    return min, max, max_adj


def _dask_hist_func(dask_array: da.Array, bins: int) -> Dict[str, List]:
    """
    Calculates a histogram on a Dask array without pulling the full chunk into memory.
    Returns both counts and bin edges in a dictionary.
    """
    # For empty arrays, return zeroed counts and default 0..255 range so the image is visible in comparisons
    if dask_array.size == 0:
        zero_counts = np.zeros(bins, dtype=int).tolist()
        return {"counts": zero_counts, "min": 0.0, "max": 255.0}

    # Compute min/max efficiently with Dask
    min_val, max_val, max_adj_val = safe_hist_range(dask_array)

    # Use Dask's histogram function and compute the counts (we don't need edges to be stored)
    counts, edges = da.histogram(dask_array, bins=bins, range=(min_val, max_adj_val))
    computed_counts, _ = da.compute(counts, edges)

    return {"counts": computed_counts.tolist(), "min": min_val, "max": max_val}


class HistogramProcessor:
    """
    Record-first processor that extracts a full hierarchy of pixel-value histograms.
    Histograms are recalculated for the full image and for every possible combination of slices.
    """

    NAME = "histogram"
    INPUT = RecordSpec(axes={"X", "Y"}, kinds={"intensity"}, capabilities={"spatial-2d"})
    OUTPUT = "features"

    # Updated schema to include the full image histogram and patterns for all slice hierarchies
    # TODO: Consider smaller data types for counts and bounds to save memory
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
        Calculates histograms for all levels of the dimensional hierarchy by iterating
        through the power set of non-spatial dimensions.
        """
        final_features = {}
        data = art.data
        dim_order = art.dim_order

        non_spatial_dims = [d for d in dim_order if d not in ("Y", "X")]
        dim_map = {dim: i for i, dim in enumerate(dim_order)}

        # Generate all hierarchy levels from the power set of dimensions
        # e.g., for ['T', 'C'], generates [(), ('T',), ('C',), ('T', 'C')]
        dim_subsets = chain.from_iterable(
            combinations(non_spatial_dims, r) for r in range(len(non_spatial_dims) + 1)
        )

        for subset in dim_subsets:
            # The empty subset represents the full image histogram
            if not subset:
                hist_dict = _dask_hist_func(data, bins=256)
                final_features["histogram_counts"] = hist_dict["counts"]
                final_features["histogram_min"] = hist_dict.get("min")
                final_features["histogram_max"] = hist_dict.get("max")
                continue

            # Get the shape of the dimensions for the current hierarchy level
            subset_shape = tuple(data.shape[dim_map[d]] for d in subset)

            # Iterate through every slice in the current hierarchy level
            # e.g., for ('T', 'C'), this iterates through (t0,c0), (t0,c1), ...
            for indices in np.ndindex(subset_shape):
                slicer = [slice(None)] * data.ndim
                name_parts = []

                for dim_name, index_val in zip(subset, indices):
                    slicer[dim_map[dim_name]] = index_val
                    name_parts.append(f"{dim_name.lower()}{index_val}")

                # Extract the data chunk (as a Dask array)
                data_chunk = data[tuple(slicer)]

                # Recalculate the histogram for this specific chunk
                hist_dict = _dask_hist_func(data_chunk, bins=256)

                # Construct the final column names
                slice_suffix = "_".join(name_parts)
                final_features[f"histogram_counts_{slice_suffix}"] = hist_dict["counts"]
                final_features[f"histogram_min_{slice_suffix}"] = hist_dict.get("min")
                final_features[f"histogram_max_{slice_suffix}"] = hist_dict.get("max")

        return final_features
