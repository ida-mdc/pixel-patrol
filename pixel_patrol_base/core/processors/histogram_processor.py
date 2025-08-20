import logging
from typing import Dict, Callable, Any, Optional, List, Tuple, Union
from functools import partial

import dask.array as da
import numpy as np

from pixel_patrol_base.core.image_operations_and_metadata import calculate_sliced_stats
from pixel_patrol_base.core.processor_interface import PixelPatrolProcessor

logger = logging.getLogger(__name__)


def _histogram_func(image_array: np.ndarray, bins: int, min_val: float, max_val: float) -> Dict[str, list]:
    """
    Calculates a histogram using a dynamic range and returns it as a dictionary of lists.
    """
    if image_array.size == 0:
        return {}

    hist_counts, bin_edges = np.histogram(a=image_array, bins=bins, range=(min_val, max_val))

    # Return a dictionary containing both counts and bin edges as lists
    return {"counts": hist_counts.tolist(), "bins": bin_edges.tolist()}


def _histogram_agg(
        hist_array_of_dicts: np.ndarray, axis: Optional[Tuple[int, ...]] = None
) -> Union[Dict[str, list], np.ndarray]:
    """
    Aggregates an array of histograms by summing their counts.
    Handles both partial and total aggregations correctly.
    """

    def _sum_histograms(hists_to_sum: List[Dict]) -> Dict:
        # Filter for valid histogram dictionaries
        counts_arrays = [d['counts'] for d in hists_to_sum if isinstance(d, dict) and 'counts' in d]
        if not counts_arrays:
            return {}

        # Sum the counts
        summed_counts = np.sum(np.stack(counts_arrays), axis=0)

        # Get the common bin edges from the first valid dictionary
        first_dict = next((d for d in hists_to_sum if isinstance(d, dict)), None)
        bin_edges_list = first_dict.get('bins', [])

        return {"counts": summed_counts.tolist(), "bins": bin_edges_list}

    # Case 1: Total aggregation
    if axis is None or len(axis) == hist_array_of_dicts.ndim:
        return _sum_histograms(hist_array_of_dicts.flatten().tolist())

    # Case 2: Partial aggregation
    output_shape = tuple(s for i, s in enumerate(hist_array_of_dicts.shape) if i not in axis)
    output_array = np.empty(output_shape, dtype=object)

    for output_indices in np.ndindex(output_array.shape):
        input_slice = [slice(None)] * hist_array_of_dicts.ndim
        kept_axis_counter = 0
        for i in range(hist_array_of_dicts.ndim):
            if i not in axis:
                input_slice[i] = output_indices[kept_axis_counter]
                kept_axis_counter += 1

        sub_array_to_agg = hist_array_of_dicts[tuple(input_slice)]
        output_array[output_indices] = _sum_histograms(sub_array_to_agg.flatten().tolist())

    return output_array


class HistogramProcessor(PixelPatrolProcessor):
    @property
    def name(self) -> str:
        return "HistogramProcessor"

    @property
    def description(self) -> str:
        return "Extracts pixel value histograms with a globally consistent range."

    def process(self, data: da.array, dim_order: str) -> dict:
        """
        Processes the data in two passes to ensure all histograms share a common, dynamic range.
        """
        # PASS 1: Compute the global min/max to establish a common range.
        logger.info("HistogramProcessor: Computing global data range...")
        min_val, max_val = da.compute(data.min(), data.max())

        # Cast to float BEFORE any arithmetic to avoid uint8 overflow
        min_val = float(min_val)
        max_val = float(max_val)

        # Handle degenerate or reversed ranges robustly
        if min_val == max_val:
            # symmetric expansion around the constant value
            pad = 0.5  # or 1.0 if you prefer
            min_val -= pad
            max_val += pad
        elif max_val < min_val:
            # just in case: swap
            min_val, max_val = max_val, min_val

        logger.info(f"HistogramProcessor: Global data range used: [{min_val}, {max_val}]")

        # Create specialized metric
        specialized_hist_func = partial(
            _histogram_func, bins=256, min_val=min_val, max_val=max_val
        )

        # Define the metric and aggregator functions to be used
        metric_fns = {'histogram': specialized_hist_func}
        agg_fns = {'histogram': _histogram_agg}

        # PASS 2: Run the main sliced statistic calculation.
        return calculate_sliced_stats(array=data, dim_order=dim_order, metric_fns=metric_fns, agg_fns=agg_fns)

    def get_specification(self) -> Dict[str, Any]:
        # The result is now an object (a dictionary)
        return {'histogram': object}

    def get_dynamic_specification_patterns(self) -> List[Tuple[str, Any]]:
        # All dynamic variations will also be objects
        return [(r"^(?:histogram)_[a-zA-Z]\d+(_[a-zA-Z]\d+)*$", object)]