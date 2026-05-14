from itertools import combinations
from typing import Callable, Tuple, Dict, List, Any, Iterable, NamedTuple
import logging
import dask.array as da
import numpy as np

logger = logging.getLogger(__name__)

NO_SLICE_AXES = ("X", "Y")


class SliceAxisSpec(NamedTuple):
    dim: str    # e.g. "T", "C" or "Z"
    idx: int   # index in dim_order
    size: int   # shape along that axis


def calculate_np_array_stats(
    array: da.array,
    dim_order: str,
    registry: Dict[str, Dict[str, Callable]],
) -> List[Dict[str, Any]]:
    if array.size == 0:
        return [{"obs_level": 0, **{k: np.nan for k in registry}}]
    all_metrics = {k: v['fn'] for k, v in registry.items()}
    all_aggregators = {k: v['agg'] for k, v in registry.items() if v['agg'] is not None}
    return calculate_sliced_stats(array, dim_order, all_metrics, all_aggregators)


def calculate_sliced_stats(
    array: da.Array,
    dim_order: str,
    metric_fns: Dict,
    agg_fns: Dict,
) -> List[Dict[str, Any]]:
    """
    Calculates statistics on a Dask array using an efficient `apply_gufunc` approach.
    Returns long-format rows: each row has ``obs_level`` (0 = global, n = per-slice
    leaf) and ``dim_<letter>`` keys for fixed dimensions.
    """

    if not metric_fns:
        return []

    xy_axes = tuple(dim_order.index(d) for d in NO_SLICE_AXES if d in dim_order)
    if len(xy_axes) != 2:
        logger.warning("Array does not have both X and Y dimensions. Skipping.")
        return []

    loop_specs = [
        SliceAxisSpec(dim, i, array.shape[i])
        for i, dim in enumerate(dim_order)
        if dim not in NO_SLICE_AXES
    ]

    metric_names = list(metric_fns.keys())
    results_dask_array = _compute_all_metrics_gufunc(array, metric_fns.values(), xy_axes, len(metric_names))
    results_np_array = results_dask_array.compute()

    return _format_and_aggregate_results(results_np_array, loop_specs, metric_names, agg_fns)


def _compute_all_metrics_gufunc(
        arr: da.Array,
        metric_fns: Iterable[Callable[[np.ndarray], Any]], # Return type is now Any
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
) -> List[Dict[str, Any]]:
    """Build long-format rows from per-slice results and their aggregations.

    Each row has ``obs_level`` (number of fixed dimensions) and ``dim_<letter>``
    keys for every fixed dimension.  obs_level=0 is the global row, obs_level equal
    to len(loop_specs) is a leaf row with every non-spatial dimension fixed.
    """
    rows_by_key: Dict[tuple, Dict] = {}
    n_loop = len(loop_specs)
    loop_axes_indices = tuple(range(n_loop))

    # Leaf rows: every non-spatial dimension is fixed (obs_level = n_loop).
    for loop_indices in np.ndindex(results_array.shape[:-1]):
        key = ("leaf",) + loop_indices
        if key not in rows_by_key:
            row: Dict[str, Any] = {"obs_level": n_loop}
            for spec, idx in zip(loop_specs, loop_indices):
                row[f"dim_{spec.dim.lower()}"] = int(idx)
            rows_by_key[key] = row
        for i, metric_name in enumerate(metric_names):
            rows_by_key[key][metric_name] = results_array[loop_indices + (i,)]

    # Aggregated rows: aggregate away a subset of axes (obs_level = len(axes_to_keep)).
    for metric_name, agg_fn in agg_fns.items():
        if metric_name not in metric_names:
            continue
        metric_idx = metric_names.index(metric_name)
        metric_data = results_array[..., metric_idx]

        for r in range(n_loop + 1):
            for axes_to_keep in combinations(loop_axes_indices, r):
                axes_to_agg_away = tuple(i for i in loop_axes_indices if i not in axes_to_keep)
                if not axes_to_agg_away:
                    continue  # leaf rows already handled above

                agg_data = agg_fn(metric_data, axis=axes_to_agg_away)
                if hasattr(agg_data, 'compute'):
                    agg_data = agg_data.compute()

                if not axes_to_keep:
                    key = ("global",)
                    if key not in rows_by_key:
                        rows_by_key[key] = {"obs_level": 0}
                    rows_by_key[key][metric_name] = agg_data
                else:
                    kept_specs = [loop_specs[i] for i in axes_to_keep]
                    for agg_indices in np.ndindex(agg_data.shape):
                        key = ("agg", r) + tuple(zip([s.dim for s in kept_specs], agg_indices))
                        if key not in rows_by_key:
                            row = {"obs_level": r}
                            for spec, idx in zip(kept_specs, agg_indices):
                                row[f"dim_{spec.dim.lower()}"] = int(idx)
                            rows_by_key[key] = row
                        rows_by_key[key][metric_name] = agg_data[agg_indices]

    return list(rows_by_key.values())

