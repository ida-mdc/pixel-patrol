import fnmatch
from itertools import product
from typing import Callable, Optional
from typing import Dict, List, Tuple, Any, NamedTuple

import cv2
import numpy as np
# import pywt
import bioio_base
from bioio import BioImage
import bioio_imageio

import logging
from pathlib import Path

from pixel_patrol.config import STANDARD_DIM_ORDER, SLICE_AXES, RGB_WEIGHTS #,STATS_THAT_NEED_GRAY_SCALE

logger = logging.getLogger(__name__)


class SliceAxisSpec(NamedTuple):
    dim: str    # e.g. "T", "C" or "Z"
    idx: int   # index in dim_order
    size: int   # shape along that axis


SPRITE_SIZE = 64


def available_columns() -> List[str]:
    keys = list(_mapping_for_np_array_processing_by_column_name().keys())
    keys.extend(_mapping_for_bioimage_metadata_by_column_name())
    return keys


def _mapping_for_np_array_processing_by_column_name() -> Dict[str, Callable]:
    """
    Maps column names (requested metadata fields) to functions that compute
    statistics on a NumPy array, potentially with hierarchical aggregation.
    """
    return {
        "mean_intensity": calculate_mean,
        "median_intensity": calculate_median,
        "std_intensity": calculate_std,
        "min_intensity": calculate_min,
        "max_intensity": calculate_max,
        "laplacian_variance": calculate_variance_of_laplacian,
        "tenengrad": calculate_tenengrad,
        "brenner": calculate_brenner,
        "noise_std": calculate_noise_estimation,
        # "wavelet_energy": calculate_wavelet_energy,
        "blocking_artifacts": calculate_blocking_artifacts,
        "ringing_artifacts": calculate_ringing_artifacts,
        "thumbnail": get_thumbnail  # Note: get_thumbnail is a wrapper for generate_thumbnail
    }


def _mapping_for_bioimage_metadata_by_column_name() -> Dict[str, Callable[[BioImage], Dict]]:
    """
    Maps requested metadata fields to functions that extract them from a BioImage object.
    These functions return a dictionary with the extracted key-value pair.
    """
    # These functions return a dict, and get_all_image_properties will update the main metadata dict
    return {
        "dim_order": lambda img: {"dim_order": STANDARD_DIM_ORDER},
        "t_size": lambda img: {"t_size": img.dims.T},
        "c_size": lambda img: {"c_size": img.dims.C},
        "z_size": lambda img: {"z_size": img.dims.Z},
        "y_size": lambda img: {"y_size": img.dims.Y},
        "x_size": lambda img: {"x_size": img.dims.X},
        "s_size": lambda img: {"s_size": img.dims.S if "S" in img.dims.order else None},
        "m_size": lambda img: {"m_size": img.dims.M if "M" in img.dims.order else None},
        "n_images": lambda img: {"n_images": len(img.scenes) if hasattr(img, 'scenes') else 1},
        "dtype": lambda img: {"dtype": str(img.dtype)},
        "pixel_size_X": lambda img: {"pixel_size_X": img.physical_pixel_sizes.X if img.physical_pixel_sizes.X else 1.0},
        "pixel_size_Y": lambda img: {"pixel_size_Y": img.physical_pixel_sizes.Y if img.physical_pixel_sizes.Y else 1.0},
        "pixel_size_Z": lambda img: {"pixel_size_Z": img.physical_pixel_sizes.Z if img.physical_pixel_sizes.Z else 1.0},
        "pixel_size_t": lambda img: {"pixel_size_t": img.physical_pixel_sizes.T if img.physical_pixel_sizes.T else 1.0},
        "channel_names": lambda img: {"channel_names": img.channel_names},
        "ome_metadata": lambda img: {"ome_metadata": img.ome_metadata},
    }


def column_matches(column: str, columns_requested: List[str]) -> bool:
    """Check if column matches any entry in columns_requested (supporting wildcards)."""
    # Using fn-match for proper wildcard support as in the old code
    return any(fnmatch.fnmatch(column, pattern) for pattern in columns_requested)


def _load_bioio_image(file_path: Path) -> Any:
    """Helper to load an image, returning the image object and its type."""
    try:
        img = BioImage(file_path)
    except bioio_base.exceptions.UnsupportedFileFormatError:
        try:
            img = BioImage(file_path, reader=bioio_imageio.Reader)
        except Exception as e:
            logger.warning(f"Could not load '{file_path}' with BioImage: {e}")
            return None
    except Exception as e:
        logger.warning(f"Could not load '{file_path}' with BioImage: {e}")
        return None

    logger.info(f"Successfully loaded '{file_path}' with BioImage.")
    return img


def _extract_metadata_from_mapping(img: Any, mapping: Dict[str, Any], required_columns: List[str]):
    """
    Extracts metadata using a given mapping, handling individual column failures.
    """

    metadata: dict[str, Any] = {}

    for column_name, extractor_func in mapping.items():
        if column_matches(column_name, required_columns):
            try:
                result = extractor_func(img)
                # Ensure the result is a dictionary before updating
                if isinstance(result, dict):
                    metadata.update(result)
                else:
                    logger.warning(
                        f"Extractor for column '{column_name}' returned non-dictionary result: {result}. Skipping update."
                    )
            except Exception as e:
                logger.warning(
                    f"Failed to extract metadata for column '{column_name}' from image. Error: {e}"
                )

    return metadata


def _get_standardized_image_array(img: Any) -> np.ndarray:
    np_array = img.get_image_data()
    dim_order = img.dims.order

    if dim_order != STANDARD_DIM_ORDER:
        for i, d in enumerate(STANDARD_DIM_ORDER):
            if d not in dim_order:
                np_array = np.expand_dims(np_array, axis=i)
                dim_order = dim_order[:i] + d + dim_order[i:]
    np_array = np.transpose(np_array, [dim_order.index(d) for d in STANDARD_DIM_ORDER])
    return np_array


def _get_numpy_array_and_dim_order(img: Any) -> Tuple[np.ndarray | None, str | None]:
    """
    Extracts NumPy array and infers dim_order based on image type.
    """
    np_array = None
    dim_order = None

    try:
        np_array = img.data
        dim_order = img.dims.order
    except Exception as e:
        logger.warning(f"Could not load data from BioImage: {e}")

    return np_array, dim_order


def get_all_image_properties(file_path: Path, required_columns: List[str]) -> Dict:

    if not file_path.exists():
        logger.warning(f"File not found: '{file_path}'. Cannot extract metadata.")
        return {}
    img = _load_bioio_image(file_path)

    if img is None:
        logger.warning(f"Failed to load image from '{file_path}'. Cannot extract metadata.")
        return {}
    np_array = _get_standardized_image_array(img)

    if np_array is not None and np_array.size > 0:
        bioimage_mapping = _mapping_for_bioimage_metadata_by_column_name()
        all_image_properties = _extract_metadata_from_mapping(img, bioimage_mapping, required_columns)
        calculate_np_array_stats(required_columns, all_image_properties, np_array)
        return all_image_properties
    else:
        logger.warning(f"NumPy array for file '{file_path}' is empty. Skipping image.")
        return {}


def calculate_np_array_stats(columns: List[str], metadata: Dict, np_array: Optional[np.ndarray]):
    """
    Calculates various NumPy array-based statistics and updates the metadata dictionary.
    This integrates the logic from the old `calculate_np_array_stats` (preprocessing.py).
    """

    original_np_array = np_array

    np_array_for_stats = _maybe_gray_scale(original_np_array)

    mapping = _mapping_for_np_array_processing_by_column_name()
    for column_name, generate_statistics_func in mapping.items():
        if column_matches(column_name, columns):
            try:
                # Special handling for thumbnail: needs original array
                if column_name == "thumbnail":
                    result = generate_statistics_func(original_np_array, STANDARD_DIM_ORDER)
                else:
                    # TODO: I think this workflow is a mistake, as we do checks and process the images per stats function while they essentially all do the same thing
                    # Should probably call compute_hierarchical_stats from here and it should loop over call all the functions.
                    result = generate_statistics_func(np_array_for_stats, STANDARD_DIM_ORDER)

                if result is not None:
                    metadata.update(result)
            except Exception as e:
                logger.warning(f"Error calculating '{column_name}' for array stats: {e}")

    # Ensure general properties are added if requested and not already present
    if "num_pixels" in columns and "num_pixels" not in metadata:
        metadata["num_pixels"] = int(original_np_array.size)
    if "dtype" in columns and "dtype" not in metadata:
        metadata["dtype"] = str(original_np_array.dtype)
    if "shape" in columns and "shape" not in metadata:
        metadata["shape"] = str(original_np_array.shape)
    if "ndim" in columns and "ndim" not in metadata:
        metadata["ndim"] = original_np_array.ndim

# --- Helper functions for array processing ---

def _maybe_gray_scale(array):
    s_idx = STANDARD_DIM_ORDER.index("S")
    if array.shape[s_idx] > 1: # and any(column_matches(col, stats_cols) for col in STATS_THAT_NEED_GRAY_SCALE):
        try:
            array = to_gray(array, s_idx)
        except ValueError as e: # TODO: check!! Can we have cased of S!=1/3/4? If not, should remove the try block
            logger.warning(f"Gray conversion failed: {e}")
    return array


def to_gray(array: np.array, color_dim_idx: int) -> np.array:
    """
    Convert an image (or higher-dimensional array) to grayscale without reordering any dimensions.
    Converts to float for calculation to ensure accuracy, then converts back to original dtype.
    """

    color_dim_size = array.shape[color_dim_idx]
    original_dtype = array.dtype

    if color_dim_size not in (3, 4):
        raise ValueError(
            f"Cannot convert to grayscale for {color_dim_size} channels. "
            "Expected 3 (RGB) or 4 (RGBA)."
        )

    if np.issubdtype(original_dtype, np.integer):
        max_val = np.iinfo(original_dtype).max
    elif np.issubdtype(original_dtype, np.floating):
        max_val = np.finfo(original_dtype).max
    else:
        max_val = float(np.nanmax(array))

    arr_float = array.astype(np.float32)

    if color_dim_size == 4:
        arr_rgb = np.take(arr_float, indices=[0, 1, 2], axis=color_dim_idx)
    else:
        arr_rgb = arr_float

    weights = np.array(RGB_WEIGHTS, dtype=np.float32)

    # dot along color axis, result drops that axis
    gray_float = np.tensordot(arr_rgb, weights, axes=([color_dim_idx], [0]))
    # reinsert axis with length=1 at same position
    gray_float = np.expand_dims(gray_float, axis=color_dim_idx)

    gray_array = np.clip(gray_float, 0, max_val).astype(original_dtype)

    return gray_array


def compute_hierarchical_stats(
        arr: np.ndarray,
        dim_order: str,
        extract_metric_func: Callable[[np.ndarray], float],  # Func now takes array AND dim_order, returns float
        metric_name: str,
        agg_func: Optional[Callable[[np.ndarray], float]] = None,
        priority_order: str = "".join(SLICE_AXES),
) -> Dict[str, float]:
    """
    Computes hierarchical statistics for a multidimensional array.
    This function processes the array slices and then aggregates results.
    """
    if len(dim_order) != arr.ndim:
        raise ValueError(f"Dimension order string '{dim_order}' length ({len(dim_order)}) does not match "
                         f"the number of array dimensions ({arr.ndim}).")

    slice_axes_specs: List[SliceAxisSpec] = [
        SliceAxisSpec(dim, idx, arr.shape[idx])
        for idx, dim in enumerate(dim_order)
        if dim in SLICE_AXES
    ]

    degenerate = (
        "X" not in dim_order
        or "Y" not in dim_order
        or arr.shape[dim_order.index("X")] == 1
        or arr.shape[dim_order.index("Y")] == 1
        or not slice_axes_specs
        or all(spec.size == 1 for spec in slice_axes_specs)
        )

    if degenerate:
        slicer = tuple(
            0 if dim not in ("X", "Y") else slice(None)
            for dim in dim_order
        )
        arr_squeezed = arr[slicer]
        arr_squeezed = np.squeeze(arr_squeezed)
        entire_image_metrics = extract_metric_func(arr_squeezed)
        logging.info(f"Degenerate case detected. Calculated {metric_name} for entire image")
        return {metric_name: entire_image_metrics} if entire_image_metrics is not None else {}

    all_slices_combo_metrics = _compute_all_slices_combo_metrics(arr, slice_axes_specs, extract_metric_func, metric_name)

    if agg_func:
        all_parent_combo_agg_metrics = _compute_aggregated_combo_metrics(all_slices_combo_metrics, slice_axes_specs, agg_func, metric_name,
                                                      priority_order)
        return {**all_slices_combo_metrics, **all_parent_combo_agg_metrics}

    all_parent_combo_metrics = _compute_parent_slice_combo_metrics(arr, slice_axes_specs, extract_metric_func, metric_name)
    return {**all_slices_combo_metrics, **all_parent_combo_metrics}



def _compute_all_slices_combo_metrics(
        arr: np.ndarray,
        slice_axes_specs: List[SliceAxisSpec],
        extract_metric_func: Callable[[np.ndarray], float],  # Func takes array AND dim_order, returns float
        metric_name: str,
) -> Dict[str, float]:
    """
    Computes the lowest level statistics for each slice along non-spatial dimensions.
    """
    detailed_stats = {}
    slice_axes_ranges = [range(spec.size) for spec in slice_axes_specs]

    for idx_tuple in product(*slice_axes_ranges):
        slicer = [slice(None)] * arr.ndim

        for spec, slice_index in zip(slice_axes_specs, idx_tuple):
            dim, axis = spec.dim, spec.idx
            slicer[axis] = slice_index

        sliced_arr = arr[tuple(slicer)]

        squeezed_arr = np.squeeze(sliced_arr)
        if squeezed_arr.ndim != 2:  # Final check for functions expecting 2D
            logger.warning(f"Skipping stat for slice: Array not 2D after processing (shape: {sliced_arr.shape}).")
            continue
        stat_value = extract_metric_func(squeezed_arr)

        if stat_value is not None:
            # Construct key: e.g., "mean_C0Z1"
            key_parts = [f"{spec.dim.lower()}{idx}" for spec, idx in zip(slice_axes_specs, idx_tuple)]
            key = f"{metric_name}_" + "_".join(key_parts)  # Combine parts with no underscore if that's desired
            detailed_stats[key] = stat_value

    return detailed_stats


def _compute_aggregated_combo_metrics(
        detailed_stats: Dict[str, float],
        slice_axes_specs: List[SliceAxisSpec],
        agg_func: Callable[[np.ndarray], float],
        metric_name: str,
        priority_order: str,
) -> Dict[str, float]:
    """
    Computes aggregated statistics for each level of the hierarchy.
    """
    agg_stats = {}

    # Iterate over each non-spatial dimension and aggregate
    # This aggregates the 'detailed_stats' (e.g., C0Z0, C0Z1) into C0, C1, Z0, Z1 etc.
    for spec in slice_axes_specs:
        dim = spec.dim
        group = {}
        for key, stat_value in detailed_stats.items():
            # Extract parts like "C0", "Z1" from "func_name_C0Z1"
            key_suffix = key[len(metric_name) + 1:]  # "C0Z1"
            parts = [s for s in key_suffix.split('_') if s]  # Split by underscore if present, remove empty

            # Find the part corresponding to the current 'dim'
            dim_part = next((p for p in parts if p.lower().startswith(dim.lower())), None)

            if dim_part:
                group.setdefault(dim_part, []).append(stat_value)

        for part, values in group.items():
            agg_key = f"{metric_name}_{part}"  # e.g., "mean_C0"
            agg_stats[agg_key] = agg_func(np.array(values))

    # Compute the final aggregated statistic based on priority order
    final_key = metric_name  # This is the global stat name (e.g., "mean_intensity")
    final_values = []

    # Iterate through priority order to find a dimension to aggregate from
    for dim_char in priority_order:
        # Check if this dimension was one of our non-spatial dimensions
        if dim_char in [spec.dim for spec in slice_axes_specs]:
            dim_keys = [k for k in agg_stats.keys() if k.startswith(f"{metric_name}_{dim_char.lower()}")]
            if dim_keys:
                final_values.extend([agg_stats[k] for k in dim_keys])
                break  # Found the highest priority dimension to aggregate

    # If we collected values, compute the final global aggregated stat
    if final_values:
        agg_stats[final_key] = agg_func(np.array(final_values))
    else:
        # Fallback: if no hierarchical aggregation happened, compute global stat from all detailed stats
        if detailed_stats:
            agg_stats[final_key] = agg_func(np.array(list(detailed_stats.values())))
        else:
            agg_stats[final_key] = 0.0  # Default if no data

    return agg_stats


def _compute_parent_slice_combo_metrics(
        arr: np.ndarray,
        slice_axes_specs: List[SliceAxisSpec],
        extract_metric_func: Callable[[np.ndarray], float],  # Func takes array AND dim_order, returns float
        metric_name: str,
) -> Dict[str, float]:
    """
    Computes metrics directly on higher-level splits when no aggregation is possible.
    """
    higher_level_stats = {}

    # Iterate over each non-spatial dimension and compute metrics for higher-level splits
    # These are slices where only one non-spatial dim is fixed, and others are `slice(None)`
    for spec in slice_axes_specs:
        dim_fixed = spec.dim
        axis_fixed = spec.idx
        for idx in range(spec.size):
            slicer = [slice(None)] * arr.ndim
            slicer[axis_fixed] = idx

            sliced_arr = arr[tuple(slicer)]

            sliced_arr = np.squeeze(sliced_arr)
            stat_value = extract_metric_func(sliced_arr)

            if stat_value is not None:
                key = f"{metric_name}_{dim_fixed.lower()}{idx}"
                higher_level_stats[key] = stat_value

    global_arr = np.squeeze(arr)
    global_stat = extract_metric_func(global_arr)
    if global_stat is not None:
        higher_level_stats[metric_name] = global_stat
    else:
        logger.warning(
            f"Cannot compute global stat for {metric_name}: Array is not 2D or is empty (shape: {global_arr.shape}).")

    return higher_level_stats


###### Individual stat functions (wrappers for compute_hierarchical_stats) ######

# These now also take `dim_order` as per the updated `compute_hierarchical_stats` signature
def calculate_mean(arr: np.array, dim_order: str) -> Dict[str, float]:
    return compute_hierarchical_stats(arr, dim_order, lambda a: float(np.mean(a)) if a.size > 0 else 0.0,
                                      "mean_intensity", agg_func=np.mean)


def calculate_median(arr: np.array, dim_order: str) -> Dict[str, float]:
    return compute_hierarchical_stats(arr, dim_order, lambda a: float(np.median(a)) if a.size > 0 else 0.0,
                                      "median_intensity", agg_func=np.median)


def calculate_std(arr: np.array, dim_order: str) -> Dict[str, float]:
    return compute_hierarchical_stats(arr, dim_order, lambda a: float(np.std(a)) if a.size > 0 else 0.0,
                                      "std_intensity", agg_func=np.std)


def calculate_min(arr: np.array, dim_order: str) -> Dict[str, float]:
    return compute_hierarchical_stats(arr, dim_order, lambda a: float(np.min(a)) if a.size > 0 else 0.0,
                                      "min_intensity", agg_func=np.min)


def calculate_max(arr: np.array, dim_order: str) -> Dict[str, float]:
    return compute_hierarchical_stats(arr, dim_order, lambda a: float(np.max(a)) if a.size > 0 else 0.0,
                                      "max_intensity", agg_func=np.max)


# For focus metrics, pass the specific 2D function directly.
# The compute_hierarchical_stats will handle the slicing and ensure 2D input.
def calculate_variance_of_laplacian(arr: np.array, dim_order: str) -> Dict[str, float]:
    return compute_hierarchical_stats(arr, dim_order, _variance_of_laplacian_2d, "laplacian_variance", agg_func=np.mean)


def calculate_tenengrad(arr: np.array, dim_order: str) -> Dict[str, float]:
    return compute_hierarchical_stats(arr, dim_order, _tenengrad_2d, "tenengrad", agg_func=np.mean)


def calculate_brenner(arr: np.array, dim_order: str) -> Dict[str, float]:
    return compute_hierarchical_stats(arr, dim_order, _brenner_2d, "brenner", agg_func=np.mean)


def calculate_noise_estimation(arr: np.array, dim_order: str) -> Dict[str, float]:
    return compute_hierarchical_stats(arr, dim_order, _noise_estimation_2d, "noise_std", agg_func=np.mean)


# def calculate_wavelet_energy(arr: np.array, dim_order: str) -> Dict[str, float]:
#     return compute_hierarchical_stats(arr, dim_order, _wavelet_energy_2d, "wavelet_energy", agg_func=np.mean)


def calculate_blocking_artifacts(arr: np.ndarray, dim_order: str) -> Dict[str, float]:
    return compute_hierarchical_stats(arr, dim_order, _check_blocking_artifacts_2d, "blocking_artifacts",
                                      agg_func=np.mean)


def calculate_ringing_artifacts(arr: np.ndarray, dim_order: str) -> Dict[str, float]:
    return compute_hierarchical_stats(arr, dim_order, _check_ringing_artifacts_2d, "ringing_artifacts",
                                      agg_func=np.mean)


def get_thumbnail(arr: np.array, dim_order: str) -> Dict[str, List[List[List[int]]]]:
    """Wrapper to generate and return thumbnail as a list for JSON compatibility."""
    thumbnail_array = _generate_thumbnail_internal(arr, dim_order)
    return {"thumbnail": thumbnail_array.tolist()}  # Convert to list for Polars/JSON serialize


# --- 2D-specific stat functions (internal, called by hierarchical wrappers) ---

def _prepare_2d_image(image: np.ndarray) -> Optional[np.ndarray]:
    if image.ndim != 2 or image.size == 0:
        return None
    img2d = image.astype(np.float32)
    return img2d

def _variance_of_laplacian_2d(image: np.ndarray) -> float:
    image = _prepare_2d_image(image)
    if image is None:
        return 0.0 # TODO: check if makes sense to return zero instead of nan or None as we aggregate this value
    lap = cv2.Laplacian(image, cv2.CV_32F)
    return lap.var()


def _tenengrad_2d(image: np.ndarray) -> float:
    image = _prepare_2d_image(image)
    if image is None or np.all(image == image.flat[0]):
        return 0.0 # TODO: check if makes sense to return zero instead of nan or None as we aggregate this value
    gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    return np.mean(mag) if mag.size > 0 else 0.0


def _brenner_2d(image: np.ndarray) -> float:
    image = _prepare_2d_image(image)
    if image is None or image.shape[1] < 3:
        return 0.0 # TODO: check if makes sense to return zero instead of nan or None as we aggregate this value
    diff = image[:, 2:] - image[:, :-2]
    return np.mean(diff ** 2) if diff.size > 0 else 0.0


def _noise_estimation_2d(image: np.ndarray) -> float:
    image = _prepare_2d_image(image)
    if image is None:
        return 0.0 # TODO: check if makes sense to return zero instead of nan or None as we aggregate this value
    median = cv2.medianBlur(image, 3)
    noise = image - median
    return float(np.std(noise))


# def _wavelet_energy_2d(image: np.ndarray, dim_order: str, wavelet='db1', level=1) -> float:
#     image = _prepare_2d_image(image)
#     if image is None:
#         return 0.0
#     try:
#         co_effs = pywt.wavedec2(gray, wavelet, level=level)
#         energy = 0.0
#         for detail in co_effs[1:]:
#             for sub_band in detail:
#                 energy += np.sum(np.abs(sub_band))
#         return energy
#     except ValueError:
#         logger.warning("Error in _wavelet_energy_2d calculation. Returning 0.0.")
#         return 0.0


def _check_blocking_artifacts_2d(image: np.ndarray) -> float:
    image = _prepare_2d_image(image)
    if image is None:
        return 0.0 # TODO: check if makes sense to return zero instead of nan or None as we aggregate this value

    block_size = 8
    height, width = image.shape
    blocking_effect = 0.0
    num_boundaries = 0

    for i in range(block_size, height, block_size):
        if i < height:
            blocking_effect += np.mean(np.abs(image[i, :] - image[i - 1, :]))
            num_boundaries += 1
    for j in range(block_size, width, block_size):
        if j < width:
            blocking_effect += np.mean(np.abs(image[:, j] - image[:, j - 1]))
            num_boundaries += 1

    return blocking_effect / num_boundaries if num_boundaries > 0 else 0.0


def _check_ringing_artifacts_2d(image: np.ndarray) -> float:
    image = np.squeeze(image)
    if image.ndim != 2:
        return 0.0 # TODO: check if makes sense to return zero instead of nan or None as we aggregate this value
    if image.size == 0 or image.shape[0] < 3 or image.shape[1] < 3:
        return 0.0 # TODO: check if makes sense to return zero instead of nan or None as we aggregate this value

    normalized_image = (cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))
    edges = cv2.Canny(normalized_image, 50, 150)
    if np.sum(edges) == 0:
        return 0.0

    kernel = np.ones((3, 3), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)
    edge_neighborhood = dilated_edges - edges

    if np.sum(edge_neighborhood > 0) == 0:
        return 0.0

    ringing_variance = np.var(image[edge_neighborhood > 0])
    return float(ringing_variance)


def _generate_thumbnail_internal(np_array: np.array, dim_order: str) -> np.array:
    """
    Internal function to generate a thumbnail (NumPy array) without direct dict wrapping.
    """
    if np_array is None or np_array.size == 0:
        return np.array([])

    arr_to_process = np_array.copy()
    current_dim_order = dim_order

    i = 0
    while arr_to_process.ndim > 2 and i < len(current_dim_order):
        dim = current_dim_order[i]
        if dim not in ["X", "Y", "C"]:  # Reduce non-spatial, non-channel dimensions
            center_index = arr_to_process.shape[i] // 2
            arr_to_process = np.take(arr_to_process, indices=center_index, axis=i)
            current_dim_order = current_dim_order.replace(dim, "")
            # Do not increment i, as the array shrinks and the next dim is now at current i
        else:
            i += 1

    if arr_to_process.ndim == 3 and "C" in current_dim_order:
        try:
            arr_to_process, _ = to_gray(arr_to_process, current_dim_order)
        except ValueError as e:
            logger.warning(f"Could not convert to grayscale for thumbnail: {e}. Keeping original channels.")
            # Fallback for failed grayscale: take first channel or average
            c_idx = current_dim_order.index("C")
            if arr_to_process.shape[c_idx] > 0:
                arr_to_process = np.take(arr_to_process, indices=0, axis=c_idx)

    if arr_to_process.ndim > 2:
        logger.warning(f"Thumbnail: Array still multi-dimensional after reduction ({arr_to_process.ndim}D). "
                       f"Taking mean along remaining non-XY dimensions.")
        while arr_to_process.ndim > 2:
            arr_to_process = np.mean(arr_to_process, axis=0)

    if arr_to_process.dtype != np.uint8:
        min_val = np_array.min()
        max_val = np_array.max()

        if max_val == min_val:
            normalized_array = np.zeros_like(arr_to_process, dtype=np.uint8)
        else:
            normalized_array = (arr_to_process - min_val) / (max_val - min_val) * 255
            normalized_array = np.clip(normalized_array, 0, 255).astype(np.uint8)
    else:
        normalized_array = arr_to_process

    try:
        if normalized_array.ndim == 3 and normalized_array.shape[0] == 1:
            normalized_array = np.squeeze(normalized_array, axis=0)

        thumbnail = cv2.resize(
            normalized_array,
            (SPRITE_SIZE, SPRITE_SIZE),
            interpolation=cv2.INTER_LANCZOS4
        )

        return thumbnail
    except TypeError as e:
        logger.error(
            f"Error converting array to PIL Image or resizing for thumbnail: {e}. Array shape: {normalized_array.shape}, dtype: {normalized_array.dtype}")
        return np.array([])
