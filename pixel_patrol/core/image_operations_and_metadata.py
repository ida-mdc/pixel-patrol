import logging
import time
from itertools import combinations
from pathlib import Path
from typing import Callable, Tuple, Dict, List, Any, NamedTuple

import dask.array as da
import numpy as np

from pixel_patrol.config import NO_SLICE_AXES
from pixel_patrol.core.loader_interface import PixelPatrolLoader

logger = logging.getLogger(__name__)

class SliceAxisSpec(NamedTuple):
    dim: str    # e.g. "T", "C" or "Z"
    idx: int   # index in dim_order
    size: int   # shape along that axis

def get_all_image_properties(file_path: Path, read_pixel_data: bool, loader: PixelPatrolLoader, processors) -> Dict:
    start_total_time = time.monotonic()

    if not file_path.exists():
        logger.warning(f"File not found: '{file_path}'. Cannot extract metadata.")
        return {}

    extracted_properties = {}

    logger.info(f"Attempting to load '{file_path}' with loader: {loader.name}")

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
        logger.info(f"Loader '{loader.name}' failed with exception, skipping: {e}")

    load_duration = time.monotonic() - start_load_time
    logger.info(f"Loading attempt with '{loader.name}' took {load_duration:.4f} seconds.")

    if success:
        loader_successful = False
        if read_pixel_data_for_this_loader:
            if da_array is not None and da_array.size > 0:
                loader_successful = True
                logger.info(f"Loaded Dask array shape: {da_array.shape}, dtype: {da_array.dtype}, chunks: {da_array.chunksize}")
            else:
                logger.warning(
                    f"Loader {loader.name} for '{file_path}' returned an empty or None Dask array. Trying next loader if available.")
        else:
            if metadata:
                loader_successful = True
            else:
                logger.warning(
                    f"Loader {loader.name} for '{file_path}' returned empty metadata. Trying next loader if available.")

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
            logger.info(f"Successfully loaded and processed '{file_path}'. Total time: {total_duration:.4f} seconds.")
            return extracted_properties  # Exit after the first successful loader

    total_duration = time.monotonic() - start_total_time
    logger.error(f"Failed to load and process '{file_path}'. Total time: {total_duration:.4f} seconds.")
    return {} # Return empty if no loader succeeded


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
        # FIX: Use dtype=object to allow for non-float results like dictionaries
        return np.array(results, dtype=object)

    return da.apply_gufunc(
        stats_wrapper,
        "(i,j)->(k)",
        arr,
        axes=[xy_axes, (-1,)],
        # FIX: The output dtype must now be 'object'
        output_dtypes=object,
        allow_rechunk=True,
        output_sizes={'k': num_metrics},
        vectorize=True
    )


# FIX: This function is updated to handle object types and not force float conversion
def _format_and_aggregate_results(
        results_array: np.ndarray,
        loop_specs: List[Any],
        metric_names: List[str],
        agg_fns: Dict[str, Callable]
) -> Dict[str, Any]: # Return type is now Any
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
                    # FIX: Do not cast to float; store the result directly
                    final_results[metric_name] = agg_data
                else:
                    kept_specs = [loop_specs[i] for i in axes_to_keep]
                    for agg_indices in np.ndindex(agg_data.shape):
                        key_suffix_parts = [f"{spec.dim.lower()}{idx}" for spec, idx in zip(kept_specs, agg_indices)]
                        key_suffix = "_".join(key_suffix_parts)
                        result_key = f"{metric_name}_{key_suffix}"
                        # FIX: Do not cast to float; store the result directly
                        final_results[result_key] = agg_data[agg_indices]

    return final_results