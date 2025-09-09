import time
import tempfile
import shutil
import logging
from pathlib import Path
from typing import Callable, Tuple, Dict, List, Any, NamedTuple

import numpy as np
import dask.array as da
import xarray as xr

# Assuming the following imports from your provided files
# These classes/functions are not defined in this script, but are assumed to exist
# in a separate file (e.g., image_operations.py).
from image_operations_and_metadata import _column_fn_registry
from image_operations_and_metadata import STANDARD_DIM_ORDER
from image_operations_and_metadata import SliceAxisSpec
from image_operations_and_metadata import compute_hierarchical_stats
from image_operations_and_metadata import to_gray

from image_operations_and_metadata_new import calculate_sliced_stats
from image_operations_and_metadata_new import _compute_all_metrics_gufunc
from image_operations_and_metadata_new import _format_and_aggregate_results


# Set up a logger for clean output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_synthetic_xarray(
    shape: Tuple[int, ...], dim_order: str
) -> xr.DataArray:
    """Creates a synthetic xarray DataArray with specified shape and dimensions."""
    data = np.random.randint(0, 256, size=shape, dtype=np.uint8)
    coords = {dim: range(size) for dim, size in zip(dim_order, shape)}
    return xr.DataArray(data, coords=coords, dims=list(dim_order))

def numpy_benchmark(image_data: xr.DataArray, required_columns: List[str]):
    """
    Simulates the NumPy-based workflow by converting to a numpy array,
    calculating stats, and timing the process.
    """
    logging.info(f"--- Starting NumPy benchmark for image shape {image_data.shape} ---")
    start_time = time.monotonic()
    
    np_array = image_data.values
    dim_order = "".join(image_data.dims)

    # Convert to grayscale if needed, as in your original script
    s_dim_idx = dim_order.index("S") if "S" in dim_order else -1
    if s_dim_idx != -1 and np_array.shape[s_dim_idx] > 1:
        np_array = to_gray(np_array, s_dim_idx)

    # Prepare metrics and aggregators
    registry = _column_fn_registry()
    all_metrics = {
        k: v["fn"] for k, v in registry.items() if k in required_columns
    }
    all_aggregators = {
        k: v["agg"] for k, v in registry.items() if k in required_columns and v["agg"]
    }

    # The actual calculation
    results = compute_hierarchical_stats(
        np_array, dim_order, all_metrics, all_aggregators
    )
    
    duration = time.monotonic() - start_time
    logging.info(f"NumPy approach took {duration:.4f} seconds.")
    logging.info(f"Calculated {len(results)} metrics.")
    return duration


def dask_benchmark(
    image_data: xr.DataArray,
    all_metrics: Dict[str, Callable],
    all_aggregators: Dict[str, Callable]
):
    # Add this line to log the start of the benchmark
    logging.info(f"--- Starting Dask benchmark for image shape {image_data.shape} ---")

    start_time = time.monotonic()

    # Define the chunk sizes explicitly to process one slice at a time.
    y_size = image_data.sizes['Y']
    x_size = image_data.sizes['X']
    da_array = image_data.chunk({'T': 1, 'C': 1, 'Z': 1, 'Y': y_size, 'X': x_size}).data
    
    stats_df = calculate_sliced_stats(
        da_array,
        image_data.dims,
        all_metrics,
        all_aggregators=all_aggregators
    )
    
    end_time = time.monotonic()
    
    duration = end_time - start_time
    logging.info(f"Dask approach took {duration:.4f} seconds.")
    # The fix is on this line:
    logging.info(f"Calculated {len(stats_df)} metrics.")

    return duration


if __name__ == "__main__":
    test_scenarios = [
        # Original scenarios
        {"name": "Small-2D", "shape": (1, 1, 1, 512, 512), "dim_order": "TCZYX"},
        {"name": "Medium-2D", "shape": (1, 1, 1, 1024, 1024), "dim_order": "TCZYX"},
        {"name": "Large-2D", "shape": (1, 1, 1, 2048, 2048), "dim_order": "TCZYX"},
        {"name": "Huge-2D", "shape": (1, 1, 1, 4096, 4096), "dim_order": "TCZYX"},
        {"name": "Small-10 slices", "shape": (10, 1, 1, 512, 512), "dim_order": "TCZYX"},
        {"name": "Medium-10 slices", "shape": (10, 1, 1, 1024, 1024), "dim_order": "TCZYX"},
        {"name": "Large-100 slices", "shape": (100, 1, 1, 1024, 1024), "dim_order": "TCZYX"},
        {"name": "XLarge-100 slices, bigger image", "shape": (100, 1, 1, 2048, 2048), "dim_order": "TCZYX"},
        {"name": "Massive-multi-channel", "shape": (100, 10, 3, 2048, 2048), "dim_order": "TCZYX"},
    ]

    columns_to_calc = [
        "mean_intensity",
        "std_intensity",
        "laplacian_variance",
        "tenengrad",
    ]
    
    # --- Start of the change ---
    # Retrieve the function registry once
    registry = _column_fn_registry()
    
    # Create the dictionaries for metrics and aggregators
    # These will be passed to both benchmark functions
    all_metrics = {
        k: v["fn"] for k, v in registry.items() if k in columns_to_calc
    }
    all_aggregators = {
        k: v["agg"] for k, v in registry.items() if k in columns_to_calc and v["agg"]
    }
    # --- End of the change ---
    
    for scenario in test_scenarios:
        logging.info("=" * 50)
        logging.info(f"Running scenario: {scenario['name']}")
        
        try:
            xarr_img = create_synthetic_xarray(
                scenario["shape"], scenario["dim_order"]
            )
        except Exception as e:
            logging.error(f"Failed to create synthetic image for scenario '{scenario['name']}': {e}")
            continue

        try:
            # --- Start of the change ---
            # Pass the new `all_metrics` and `all_aggregators` dictionaries to dask_benchmark
            dask_time = dask_benchmark(xarr_img, all_metrics, all_aggregators)
            # --- End of the change ---
        except Exception as e:
            logging.error(f"Dask benchmark failed for '{scenario['name']}': {e}")
            dask_time = None
        
        try:
            numpy_time = numpy_benchmark(xarr_img, columns_to_calc)
        except Exception as e:
            logging.error(f"NumPy benchmark failed for '{scenario['name']}': {e}")
            numpy_time = None
        

        logging.info("-" * 50)
        logging.info(f"Summary for {scenario['name']}:")
        if dask_time is not None:
            logging.info(f"Dask time: {dask_time:.4f}s")
        if numpy_time is not None:
            logging.info(f"NumPy time: {numpy_time:.4f}s")
        
        if dask_time is not None and numpy_time is not None:
            if numpy_time > dask_time:
                logging.info(f"Dask was {(numpy_time / dask_time):.2f}x faster.")
            else:
                logging.info(f"NumPy was {(dask_time / numpy_time):.2f}x faster.")
        elif dask_time is None and numpy_time is not None:
            logging.info("Dask failed but NumPy succeeded.")
        elif dask_time is not None and numpy_time is None:
            logging.info("NumPy failed but Dask succeeded.")
        else:
            logging.info("Both benchmarks failed.")
            
        logging.info("=" * 50)