from itertools import combinations
from typing import Callable, Tuple, Dict, List, Any
from typing import NamedTuple

import dask.array as da
import numpy as np

NO_SLICE_AXES = ("X", "Y")

class SliceAxisSpec(NamedTuple):
    dim: str    # e.g. "T", "C" or "Z"
    idx: int   # index in dim_order
    size: int   # shape along that axis

def calculate_sliced_stats(array: da.Array, dim_order: str, metric_fns: Dict, agg_fns: Dict) -> Dict[str, Any]:
    """
    Calculates statistics on a Dask array using an efficient `apply_gufunc` approach.
    This version is updated to handle both scalar and object metric results.
    """
    if not metric_fns:
        return {}

    spatial_dims = NO_SLICE_AXES
    xy_axes = tuple(dim_order.index(d) for d in spatial_dims if d in dim_order)
    if len(xy_axes) != 2:
        print("Warning: Array does not have both X and Y dimensions. Skipping.")
        return {}

    loop_specs = [
        SliceAxisSpec(dim, i, array.shape[i])
        for i, dim in enumerate(dim_order)
        if dim not in NO_SLICE_AXES
    ]

    metric_names = list(metric_fns.keys())
    results_dask_array = _compute_all_metrics_gufunc(array, metric_fns.values(), xy_axes, len(metric_names))

    results_np_array = results_dask_array.compute()

    all_image_properties = _format_and_aggregate_results(
        results_np_array,
        loop_specs,
        metric_names,
        agg_fns
    )

    return all_image_properties


def _compute_all_metrics_gufunc(
        arr: da.Array,
        metric_fns: List[Callable[[np.ndarray], Any]], # Return type is now Any
        xy_axes: Tuple[int, int],
        num_metrics: int
) -> da.Array:
    """
    Applies multiple metric functions to each 2D slice of a Dask array.
    Updated to handle object outputs (like dictionaries for KDE).
    """

    def stats_wrapper(x_y_plane: np.ndarray) -> np.ndarray:
        results = [fn(x_y_plane) for fn in metric_fns]
        return np.array(results, dtype=object)

    return da.apply_gufunc(
        stats_wrapper,
        "(i,j)->(k)",
        arr,
        axes=[xy_axes, (-1,)],
        output_dtypes=object,
        allow_rechunk=True,
        output_sizes={'k': num_metrics},
        vectorize=True
    )


def _format_and_aggregate_results(
        results_array: np.ndarray,
        loop_specs: List[Any],
        metric_names: List[str],
        agg_fns: Dict[str, Callable]
) -> Dict[str, Any]:
    """
    Formats detailed metrics and computes aggregations.
    This version can handle object dtypes and does not force float conversion.
    """
    final_results = {}
    num_metrics = len(metric_names)

    for loop_indices in np.ndindex(results_array.shape[:-1]):
        key_suffix_parts = [f"{spec.dim.lower()}{idx}" for spec, idx in zip(loop_specs, loop_indices)]
        key_suffix = "_".join(key_suffix_parts)

        for i in range(num_metrics):
            metric_name = metric_names[i]
            result_key = f"{metric_name}_{key_suffix}" if key_suffix else metric_name
            final_results[result_key] = results_array[loop_indices + (i,)]

    loop_axes_indices = tuple(range(len(loop_specs)))

    for metric_name, agg_fn in agg_fns.items():
        if metric_name not in metric_names:
            continue

        metric_idx = metric_names.index(metric_name)
        metric_data = results_array[..., metric_idx]

        for r in range(len(loop_specs) + 1): # +1 to include full aggregation
            for axes_to_keep in combinations(loop_axes_indices, r):
                axes_to_agg_away = tuple(i for i in loop_axes_indices if i not in axes_to_keep)

                # Skip the no-aggregation case as it's handled by the per-slice logic above
                if not axes_to_agg_away:
                    continue

                agg_data = agg_fn(metric_data, axis=axes_to_agg_away)

                if hasattr(agg_data, 'compute'):
                    agg_data = agg_data.compute()

                if not axes_to_keep:
                    final_results[metric_name] = agg_data
                else:
                    kept_specs = [loop_specs[i] for i in axes_to_keep]
                    for agg_indices in np.ndindex(agg_data.shape):
                        key_suffix_parts = [f"{spec.dim.lower()}{idx}" for spec, idx in zip(kept_specs, agg_indices)]
                        key_suffix = "_".join(key_suffix_parts)
                        result_key = f"{metric_name}_{key_suffix}"
                        final_results[result_key] = agg_data[agg_indices]
    return final_results

