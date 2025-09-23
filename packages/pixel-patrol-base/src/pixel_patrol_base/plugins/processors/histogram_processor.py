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


def _dask_hist_func(dask_array: da.Array, bins: int) -> Dict[str, List]:
    """
    Calculates a histogram on a Dask array without pulling the full chunk into memory.
    Returns both counts and bin edges in a dictionary.
    """
    if dask_array.size == 0:
        return {"counts": [], "bins": []}

    # Compute min/max efficiently with Dask
    min_val, max_val = da.compute(dask_array.min(), dask_array.max())
    min_val, max_val = float(min_val), float(max_val)

    # Guard against a degenerate range
    if min_val == max_val:
        min_val, max_val = min_val - 0.5, max_val + 0.5

    # Use Dask's histogram function and compute the result
    counts, edges = da.histogram(dask_array, bins=bins, range=(min_val, max_val))
    computed_counts, computed_edges = da.compute(counts, edges)

    return {"counts": computed_counts.tolist(), "bins": computed_edges.tolist()}


class HistogramProcessor:
    """
    Record-first processor that extracts a full hierarchy of pixel-value histograms.
    Histograms are recalculated for the full image and for every possible combination of slices.
    """

    NAME = "histogram"
    INPUT = RecordSpec(axes={"X", "Y"}, kinds={"intensity"}, capabilities={"spatial-2d"})
    OUTPUT = "features"

    # Updated schema to include the full image histogram and patterns for all slice hierarchies
    OUTPUT_SCHEMA = {
        "histogram_counts": pl.List(pl.Int64),
        "histogram_bins": pl.List(pl.Float64),
    }
    OUTPUT_SCHEMA_PATTERNS = [
        (r"^(?:histogram)_counts_.*$", pl.List(pl.Int64)),
        (r"^(?:histogram)_bins_.*$", pl.List(pl.Float64)),
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
                final_features["histogram_bins"] = hist_dict["bins"]
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
                final_features[f"histogram_bins_{slice_suffix}"] = hist_dict["bins"]

        return final_features