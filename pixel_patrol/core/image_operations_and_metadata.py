import logging
import time
from itertools import combinations
from pathlib import Path
from typing import Callable, Tuple, Dict, List, Any, NamedTuple

import dask.array as da
import numpy as np

from pixel_patrol.config import NO_SLICE_AXES

logger = logging.getLogger(__name__)

class SliceAxisSpec(NamedTuple):
    dim: str    # e.g. "T", "C" or "Z"
    idx: int   # index in dim_order
    size: int   # shape along that axis


def get_all_image_properties(file_path: Path, read_pixel_data: bool, loaders, processors) -> Dict:
    start_total_time = time.monotonic()

    if not file_path.exists():
        logger.warning(f"File not found: '{file_path}'. Cannot extract metadata.")
        return {}

    extracted_properties = {}

    for loader in loaders:
        loader_name = type(loader).__name__
        logger.info(f"Attempting to load '{file_path}' with loader: {loader_name}")

        da_array = None
        metadata = None
        success = False

        read_pixel_data_for_this_loader = read_pixel_data and not loader.reads_only_metadata

        start_load_time = time.monotonic()
        try:
            if read_pixel_data_for_this_loader:
                metadata, da_array = loader.read_metadata_and_data(file_path)
            else:
                metadata = loader.read_metadata(file_path)
            success = True
        except Exception as e:
            logger.info(f"Loader '{loader_name}' failed with exception, skipping: {e}")

        load_duration = time.monotonic() - start_load_time
        logger.info(f"Loading attempt with '{loader_name}' took {load_duration:.4f} seconds.")


        if success:
            loader_successful = False
            if read_pixel_data_for_this_loader:
                if da_array is not None and da_array.size > 0:
                    loader_successful = True
                    logger.info(f"Loaded Dask array shape: {da_array.shape}, dtype: {da_array.dtype}, chunks: {da_array.chunksize}")
                else:
                    logger.warning(
                        f"Loader {loader_name} for '{file_path}' returned an empty or None Dask array. Trying next loader if available.")
            else:
                if metadata:
                    loader_successful = True
                else:
                    logger.warning(
                        f"Loader {loader_name} for '{file_path}' returned empty metadata. Trying next loader if available.")

            if loader_successful:
                extracted_properties.update(metadata)

                if read_pixel_data_for_this_loader:
                    dim_order = metadata.get("dim_order", "")
                    for processor in processors:
                        processor_name = type(processor).__name__
                        logger.info(f"Starting processing with {processor_name}...")
                        start_process_time = time.monotonic()

                        features = processor.process(da_array, dim_order)

                        process_duration = time.monotonic() - start_process_time
                        logger.info(f"Processor '{processor_name}' finished in {process_duration:.4f} seconds.")
                        extracted_properties.update(features)

                total_duration = time.monotonic() - start_total_time
                logger.info(f"Successfully loaded and processed '{file_path}' with {loader_name}. Total time: {total_duration:.4f} seconds.")
                return extracted_properties  # Exit after the first successful loader

    total_duration = time.monotonic() - start_total_time
    logger.error(f"Failed to load and process '{file_path}' with any of the available loaders. Total time: {total_duration:.4f} seconds.")
    return {} # Return empty if no loader succeeded


def calculate_sliced_stats(array: da.Array, dim_order: str, all_metrics, all_aggregators) -> Dict[str, float]:
    """
    Calculates statistics on a Dask array using an efficient `apply_gufunc` approach.

    This method first computes all slice-wise metrics in a single Dask operation,
    then formats and aggregates those results in memory.
    """
    if not all_metrics:
        return {}

    spatial_dims = {"X", "Y"}
    xy_axes = tuple(dim_order.index(d) for d in spatial_dims if d in dim_order)
    if len(xy_axes) != 2:
        print("Warning: Array does not have both X and Y dimensions. Skipping.")
        return {}

    loop_specs = [
        SliceAxisSpec(dim, i, array.shape[i])
        for i, dim in enumerate(dim_order)
        if dim not in NO_SLICE_AXES
    ]

    # 2. Compute all slice-wise metrics at once using Dask
    metric_names = list(all_metrics.keys())
    # This Dask array holds the results for all metrics on all slices
    # The graph is built here, but nothing is computed yet.
    results_dask_array = _compute_all_metrics_gufunc(array, all_metrics.values(), xy_axes, len(metric_names))

    # 3. Execute the computation
    # This is the only .compute() call needed for all slice metrics.
    results_np_array = results_dask_array.compute()

    # 4. Format and aggregate the computed results in memory
    all_image_properties = _format_and_aggregate_results(
        results_np_array,
        loop_specs,
        metric_names,
        all_aggregators
    )

    return all_image_properties


def _compute_all_metrics_gufunc(
        arr: da.Array,
        metric_fns: List[Callable[[np.ndarray], float]],
        xy_axes: Tuple[int, int],
        num_metrics: int
) -> da.Array:
    """
    Applies multiple metric functions to each 2D slice of a Dask array.

    Returns a new Dask array where spatial dimensions are replaced by a
    single dimension containing the result of each metric function.
    """

    def stats_wrapper(x_y_plane: np.ndarray) -> np.ndarray:
        # This internal function runs on a single NumPy 2D slice
        # and returns a 1D array of metric results.
        results = [fn(x_y_plane) for fn in metric_fns]
        return np.array(results, dtype=np.float64)

    # The signature "(i,j)->(k)" means:
    # - Take a 2D core input with dimensions named 'i' and 'j'.
    # - Produce a 1D core output with a dimension named 'k'.
    return da.apply_gufunc(
        stats_wrapper,
        "(i,j)->(k)",
        arr,
        axes=[xy_axes, (-1,)],  # Map array's xy_axes to signature's (i,j)
        output_dtypes=np.float64,
        allow_rechunk=True,
        output_sizes={'k': num_metrics},  # Specify size of new dimension 'k'
        vectorize=True  # For performanceallow_rechunk=True
    )


def _format_and_aggregate_results(
        results_array: np.ndarray,
        loop_specs: List[Any],  # Using Any to be generic
        metric_names: List[str],
        agg_fns: Dict[str, Callable[[np.ndarray], float]]
) -> Dict[str, float]:
    """
    Formats detailed metrics and computes aggregations.
    This version includes the fix to compute Dask arrays.
    """
    final_results = {}
    num_metrics = len(metric_names)

    # --- Part 1: Format detailed, per-slice metrics (no changes) ---
    for loop_indices in np.ndindex(results_array.shape[:-1]):
        key_suffix_parts = [f"{spec.dim.lower()}{idx}" for spec, idx in zip(loop_specs, loop_indices)]
        key_suffix = "_".join(key_suffix_parts)

        for i in range(num_metrics):
            metric_name = metric_names[i]
            result_key = f"{metric_name}_{key_suffix}" if key_suffix else metric_name
            final_results[result_key] = results_array[loop_indices + (i,)]

    # --- Part 2: Compute and format all aggregations ---
    loop_axes_indices = tuple(range(len(loop_specs)))

    for metric_name, agg_fn in agg_fns.items():
        if metric_name not in metric_names:
            continue

        metric_idx = metric_names.index(metric_name)
        metric_data = results_array[..., metric_idx]

        for r in range(len(loop_specs)):
            for axes_to_keep in combinations(loop_axes_indices, r):
                axes_to_agg_away = tuple(i for i in loop_axes_indices if i not in axes_to_keep)

                agg_data = agg_fn(metric_data, axis=axes_to_agg_away)

                if hasattr(agg_data, 'compute'):
                    agg_data = agg_data.compute()

                if not axes_to_keep:
                    final_results[metric_name] = float(agg_data)
                else:
                    kept_specs = [loop_specs[i] for i in axes_to_keep]
                    for agg_indices in np.ndindex(agg_data.shape):
                        key_suffix_parts = [f"{spec.dim.lower()}{idx}" for spec, idx in zip(kept_specs, agg_indices)]
                        key_suffix = "_".join(key_suffix_parts)
                        result_key = f"{metric_name}_{key_suffix}"
                        final_results[result_key] = float(agg_data[agg_indices])

    return final_results