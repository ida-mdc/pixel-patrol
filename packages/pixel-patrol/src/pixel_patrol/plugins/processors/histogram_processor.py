import logging
from functools import partial
from typing import Dict, Optional, Tuple, List, Union

import dask.array as da
import numpy as np

from pixel_patrol.plugins.array_slicing import calculate_sliced_stats
from pixel_patrol_base.core.artifact import Artifact
from pixel_patrol_base.core.contracts import ProcessResult
from pixel_patrol_base.core.specs import ArtifactSpec

logger = logging.getLogger(__name__)


def _histogram_func(image_array: np.ndarray, bins: int, min_val: float, max_val: float) -> Dict[str, list]:
    """
    Calculates a histogram using a dynamic range and returns it as a dictionary of lists.
    """
    if image_array.size == 0:
        return {}
    counts, edges = np.histogram(a=image_array, bins=bins, range=(min_val, max_val))
    return {"counts": counts.tolist(), "bins": edges.tolist()}


def _histogram_agg(
    hist_array_of_dicts: np.ndarray, axis: Optional[Tuple[int, ...]] = None
) -> Union[Dict[str, list], np.ndarray]:
    """
    Aggregates an array of histograms by summing their counts.
    Handles both partial and total aggregations correctly.
    """

    def _sum_histograms(hists_to_sum: List[Dict]) -> Dict:
        counts_arrays = [d["counts"] for d in hists_to_sum if isinstance(d, dict) and "counts" in d]
        if not counts_arrays:
            return {}
        summed_counts = np.sum(np.stack(counts_arrays), axis=0)
        first_dict = next((d for d in hists_to_sum if isinstance(d, dict)), None)
        bin_edges_list = first_dict.get("bins", []) if first_dict else []
        return {"counts": summed_counts.tolist(), "bins": bin_edges_list}

    # Total aggregation
    if axis is None or len(axis) == hist_array_of_dicts.ndim:
        return _sum_histograms(hist_array_of_dicts.flatten().tolist())

    # Partial aggregation
    output_shape = tuple(s for i, s in enumerate(hist_array_of_dicts.shape) if i not in axis)
    output_array = np.empty(output_shape, dtype=object)

    for output_indices in np.ndindex(output_array.shape):
        input_slice = [slice(None)] * hist_array_of_dicts.ndim
        kept_axis_counter = 0
        for i in range(hist_array_of_dicts.ndim):
            if i not in axis:
                input_slice[i] = output_indices[kept_axis_counter]
                kept_axis_counter += 1

        sub_array = hist_array_of_dicts[tuple(input_slice)]
        output_array[output_indices] = _sum_histograms(sub_array.flatten().tolist())

    return output_array


class HistogramProcessor:
    """
    Artifact-first processor that extracts pixel-value histograms with a globally consistent range.
    """

    # ---- Declarative plugin metadata ----
    NAME   = "histogram"
    INPUT  = ArtifactSpec(axes={"X", "Y"}, kinds={"intensity"}, capabilities={"spatial-2d"})
    OUTPUT = "features"  # returns a dict of features

    # Static & dynamic output schema for your table builder
    OUTPUT_SCHEMA = {
        "histogram": object,  # aggregated histogram (counts+bins)
    }
    OUTPUT_SCHEMA_PATTERNS = [
        (r"^(?:histogram)_[A-Za-z]\d+(?:_[A-Za-z]\d+)*$", object),  # per-slice hist keys from calculate_sliced_stats
    ]

    # ---- Execution ----
    def run(self, art: Artifact) -> ProcessResult:
        """
        Two-pass approach to use a single global range for all per-slice histograms.
        """
        data = art.data
        dim_order = art.meta.get("dim_order", "")

        # PASS 1: global min/max (Dask lazy compute)
        min_val, max_val = da.compute(data.min(), data.max())
        min_val = float(min_val)
        max_val = float(max_val)

        # guard against degenerate or reversed ranges
        if min_val == max_val:
            pad = 0.5
            min_val -= pad
            max_val += pad
        elif max_val < min_val:
            min_val, max_val = max_val, min_val

        # PASS 2: slice-wise histogram with fixed range
        specialized_hist = partial(_histogram_func, bins=256, min_val=min_val, max_val=max_val)
        metric_fns = {"histogram": specialized_hist}
        agg_fns = {"histogram": _histogram_agg}

        return calculate_sliced_stats(array=data, dim_order=dim_order, metric_fns=metric_fns, agg_fns=agg_fns)
