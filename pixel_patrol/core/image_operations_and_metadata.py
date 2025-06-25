import fnmatch
from itertools import product
from typing import Callable, Optional
from typing import Dict, List, Tuple, Any

import bioio_base
import cv2
import numpy as np
# import pywt
from PIL import Image
from bioio import BioImage

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

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
    # These functions return a dict, and extract_image_metadata will update the main metadata dict
    return {
        "dim_order": lambda img: {"dim_order": img.dims.order},
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


def _mapping_for_png_metadata_by_column_name() -> Dict[str, Callable[[Image.Image], Dict]]:
    """
    Maps requested metadata fields to functions that extract them from a PIL Image object.
    """
    # These functions return a dict, and extract_image_metadata will update the main metadata dict
    return {
        "dim_order": lambda img: {"dim_order": "XYC" if len(img.getbands()) > 1 else "XY"},
        "t_size": lambda img: {"t_size": 1},
        "c_size": lambda img: {"c_size": len(img.getbands())},  # Number of channels
        "z_size": lambda img: {"z_size": 1},
        "y_size": lambda img: {"y_size": img.height},
        "x_size": lambda img: {"x_size": img.width},
        "s_size": lambda img: {"s_size": 1},
        "m_size": lambda img: {"m_size": 1},
        "n_images": lambda img: {"n_images": 1},
        "dtype": lambda img: {"dtype": str(img.mode)},  # PIL mode as dtype
        "pixel_size_X": lambda img: {"pixel_size_X": 1.0},  # Default if not found
        "pixel_size_Y": lambda img: {"pixel_size_Y": 1.0},  # Default if not found
        "pixel_size_Z": lambda img: {"pixel_size_Z": 1.0},  # Default if not found
        "pixel_size_t": lambda img: {"pixel_size_t": 1.0},  # Default if not found
        "channel_names": lambda img: {"channel_names": list(img.getbands())},  # PIL getbands() for channel names
    }


def column_matches(column: str, columns_requested: List[str]) -> bool:
    """Check if column matches any entry in columns_requested (supporting wildcards)."""
    # Using fnmatch for proper wildcard support as in the old code
    return any(fnmatch.fnmatch(column, pattern) for pattern in columns_requested)


def _load_image(file_path: Path) -> Tuple[Any, str | None]:
    """Helper to load an image, returning the image object and its type."""
    try:
        img = BioImage(file_path)
        logger.debug(f"Successfully loaded '{file_path}' as BioImage.")
        print(f"Image loaded from {file_path} with dimensions: {img.dims.order}, shape: {img.data.shape}")
        return img, "bioimage"
    except bioio_base.exceptions.UnsupportedFileFormatError:
        logger.debug(f"'{file_path}' is not a BioImage, attempting to load with PIL.")
        try:
            img = Image.open(file_path)
            logger.debug(f"Successfully loaded '{file_path}' with PIL.")
            return img, "pil"
        except Exception as e:
            logger.warning(f"Could not load '{file_path}' with PIL: {e}")
            return None, None
    except Exception as e:
        logger.warning(f"Could not load '{file_path}' as BioImage: {e}")
        return None, None

def _extract_metadata_from_mapping(
    img: Any,
    mapping: Dict[str, Any], # Changed from Any to avoid circular imports if BioImage is not defined
    required_columns: List[str],
    metadata: Dict
):
    """
    Extracts metadata using a given mapping, handling individual column failures.
    """
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



def _get_numpy_array_and_dim_order(img: Any, img_type: str, required_columns: List[str], metadata: Dict) -> Tuple[np.ndarray | None, str | None]:
    """
    Extracts NumPy array and infers dim_order based on image type.
    """
    np_array = None
    dim_order = None

    if img_type == "bioimage":
        try:
            np_array = img.data
            dim_order = img.dims.order
        except Exception as e:
            logger.warning(f"Could not load data from BioImage: {e}")
    elif img_type == "pil":
        try:
            np_array = np.array(img)
            # Basic inference for PIL images if dim_order is required and not already from mapping
            if "dim_order" in required_columns:
                if np_array.ndim == 3 and np_array.shape[-1] in [3, 4]:
                    inferred_dim_order = "XYC"
                elif np_array.ndim == 2:
                    inferred_dim_order = "XY"
                else:
                    inferred_dim_order = ""  # Cannot infer
                metadata["dim_order"] = inferred_dim_order
            dim_order = metadata.get("dim_order") # Use the inferred or previously extracted dim_order
        except Exception as e:
            logger.warning(f"Could not convert PIL Image to NumPy array: {e}")

    return np_array, dim_order


def extract_image_metadata(file_path: Path, required_columns: List[str]) -> Dict:
    if not file_path.exists():
        logger.warning(f"File not found: '{file_path}'. Cannot extract metadata.")
        return {}
    img, img_type = _load_image(file_path)
    if img is None:
        logger.error(f"Failed to load image from '{file_path}'. Cannot extract metadata.")
        return {}

    metadata = {}

    if img_type == "bioimage":
        bioimage_mapping = _mapping_for_bioimage_metadata_by_column_name()
        _extract_metadata_from_mapping(img, bioimage_mapping, required_columns, metadata)
    elif img_type == "pil":
        pil_mapping = _mapping_for_png_metadata_by_column_name()
        _extract_metadata_from_mapping(img, pil_mapping, required_columns, metadata)

    np_array, dim_order = _get_numpy_array_and_dim_order(img, img_type, required_columns, metadata)

    if np_array is not None and np_array.size > 0:
        calculate_np_array_stats(required_columns, metadata, np_array, dim_order)
    elif np_array is not None and np_array.size == 0:
        logger.warning(f"NumPy array for file '{file_path}' is empty. Skipping array stats.")

    return metadata


def calculate_np_array_stats(columns: List[str], metadata: Dict, np_array: Optional[np.ndarray],
                             dim_order: Optional[str]):
    """
    Calculates various NumPy array-based statistics and updates the metadata dictionary.
    This integrates the logic from the old `calculate_np_array_stats` (preprocessing.py).
    """
    if np_array is None or np_array.size == 0:
        return

    original_np_array = np_array
    original_dim_order = dim_order

    stats_that_need_grayscale = [
        "mean_intensity", "median_intensity", "std_intensity", "min_intensity", "max_intensity",
        "laplacian_variance", "tenengrad", "brenner", "noise_std",
        "wavelet_energy", "blocking_artifacts", "ringing_artifacts"
    ]

    # Determine if grayscale conversion is needed for intensity/focus stats
    needs_grayscale = False
    if original_dim_order and "C" in original_dim_order:
        c_index = original_dim_order.index("C")
        if original_np_array.ndim > c_index and original_np_array.shape[c_index] > 1:
            if any(column_matches(col, columns) for col in stats_that_need_grayscale):
                needs_grayscale = True

    np_array_for_stats = original_np_array
    dim_order_for_stats = original_dim_order

    if needs_grayscale:
        try:
            gray_array, gray_dim_order = to_gray(original_np_array, original_dim_order)
            np_array_for_stats = gray_array
            dim_order_for_stats = gray_dim_order
        except ValueError as e:
            logger.warning(
                f"Grayscale conversion failed for stats: {e}. Attempting to run stats on original array where possible.")
            # Fallback to original array if grayscale conversion fails,
            # but stats that need 2D will fail or return 0.0 later.

    mapping = _mapping_for_np_array_processing_by_column_name()
    for column_name, func in mapping.items():
        if column_matches(column_name, columns):
            try:
                # Special handling for thumbnail: needs original array
                if column_name == "thumbnail":
                    result = func(original_np_array, original_dim_order)
                # For stats requiring 2D input (after grayscale conversion)
                elif column_name in stats_that_need_grayscale:
                    if np_array_for_stats.ndim != 2:
                        logger.warning(
                            f"Skipping {column_name}: Requires 2D (grayscale) array, got shape {np_array_for_stats.shape}.")
                        continue
                    result = func(np_array_for_stats, dim_order_for_stats)
                else:
                    # Other numerical stats can operate on the possibly-grayscale array
                    result = func(np_array_for_stats, dim_order_for_stats)

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
        metadata["ndim"] = int(original_np_array.ndim)
    if "dim_order" in columns and "dim_order" not in metadata:
        metadata["dim_order"] = original_dim_order if original_dim_order else (
            "YX" if original_np_array.ndim == 2 else "CZYX")


# --- Helper functions for array processing (moved directly from old preprocessing.py) ---

def to_gray(image: np.array, dim_order: str) -> tuple[np.array, str]:
    """
    Convert an image (or higher-dimensional array) to grayscale without reordering any dimensions.
    Converts to float for calculation to ensure accuracy, then converts back to original dtype.
    """
    if "C" not in dim_order:
        return image, dim_order

    c_index = dim_order.index("C")
    n_channels = image.shape[c_index]

    if n_channels == 1:
        gray = np.squeeze(image, axis=c_index)
        new_dim_order = dim_order.replace("C", "")
        return gray, new_dim_order

    if n_channels < 3:
        raise ValueError(
            f"Cannot convert to grayscale for {n_channels} channels. Expected 1, 3 (RGB/BGR), or 4 (RGBA).")

    # Store original dtype and its maximum value for proper scaling back
    original_dtype = image.dtype
    if np.issubdtype(original_dtype, np.integer):
        # For integer types (like uint8), get the max representable value (e.g., 255 for uint8)
        max_val_original_dtype = np.iinfo(original_dtype).max
    else:
        # For float types, assume the range is 0-1, so max value is 1.0
        max_val_original_dtype = 1.0

    # Convert image to float for accurate grayscale conversion.
    # It's crucial to perform the weighted sum in floating point.
    image_float = image.astype(np.float32)

    # Move channel axis to the end for easier dot product
    image_transposed_for_dot = np.moveaxis(image_float, c_index, -1)

    # Standard grayscale conversion weights (luminosity method)
    # Ensure weights are float32 to avoid truncation
    weights = np.array([0.2989, 0.5870, 0.1140], dtype=np.float32)

    if n_channels == 4:
        # For RGBA, use only the RGB channels for grayscale conversion
        gray_float = np.dot(image_transposed_for_dot[..., :3], weights)
    else:
        gray_float = np.dot(image_transposed_for_dot, weights)

    # Normalize and convert back to the original data type.
    # The grayscale_float result will be in a range roughly corresponding to the input's max value.
    # Clip to ensure values are within the valid range before casting to the original integer type.
    gray = np.clip(gray_float, 0, max_val_original_dtype).astype(original_dtype)

    new_dim_order = dim_order.replace("C", "")
    return gray, new_dim_order


def compute_hierarchical_stats(
        arr: np.ndarray,
        dim_order: str,
        func: Callable[[np.ndarray, str], float],  # Func now takes array AND dim_order, returns float
        func_name: str,
        agg_func: Optional[Callable[[np.ndarray], float]] = None,
        priority_order: str = "CTZ",
) -> Dict[str, float]:
    """
    Computes hierarchical statistics for a multi-dimensional array.
    This function processes the array slices and then aggregates results.
    """
    if len(dim_order) != arr.ndim:
        # Adjusting the error message to be more specific to array dimensions
        raise ValueError(f"Dimension order string '{dim_order}' length ({len(dim_order)}) does not match "
                         f"the number of array dimensions ({arr.ndim}).")
    if "X" not in dim_order or "Y" not in dim_order:
        # Warn instead of error if XY are not present, as some images might be non-spatial data
        logger.warning(
            "Dim order does not contain 'X' or 'Y' spatial dimensions. Hierarchical stats might not be meaningful.")
        # If no spatial dimensions, treat as a single slice for global stat
        global_stat = func(arr, dim_order)  # Pass dim_order to func
        return {func_name: global_stat} if global_stat is not None else {}

    # Identify non-spatial dimensions: (letter, axis_index)
    non_spatial = [(dim, i) for i, dim in enumerate(dim_order) if dim not in ["X", "Y"]]

    # If there are no non-spatial dimensions, compute the metric directly on the entire array (assuming 2D)
    if not non_spatial:
        if arr.ndim == 2:  # Ensure it's a 2D array if no non-spatial dimensions
            global_stat = func(arr, dim_order)  # Pass dim_order to func
            return {func_name: global_stat} if global_stat is not None else {}
        else:
            logger.warning(
                f"Array has {arr.ndim} dimensions but no non-spatial dims specified. Cannot compute 2D stat.")
            return {}  # Cannot compute 2D stat on non-2D array

    # Compute the lowest level statistics (leaves of the tree)
    detailed_stats = _compute_lowest_level_stats(arr, dim_order, non_spatial, func, func_name)

    # If no aggregation is required (e.g., for direct per-slice metrics)
    if agg_func is None:
        higher_level_stats = _compute_higher_level_stats(arr, dim_order, non_spatial, func, func_name)
        return {**detailed_stats, **higher_level_stats}

    # Compute aggregated statistics for each level of the hierarchy
    agg_stats = _compute_aggregated_stats(detailed_stats, non_spatial, agg_func, func_name, priority_order)

    # Combine results
    result = {**detailed_stats, **agg_stats}
    return result


def _compute_lowest_level_stats(
        arr: np.ndarray,
        dim_order: str,
        non_spatial: List[tuple],
        func: Callable[[np.ndarray, str], float],  # Func takes array AND dim_order, returns float
        func_name: str,
) -> Dict[str, float]:
    """
    Computes the lowest level statistics for each slice along non-spatial dimensions.
    """
    detailed_stats = {}
    non_spatial_ranges = [range(arr.shape[i]) for _, i in non_spatial]

    for idx_tuple in product(*non_spatial_ranges):
        slicer = [slice(None)] * arr.ndim
        current_slice_dim_order = list(dim_order)  # Start with full dim_order

        for (dim, axis), idx in zip(non_spatial, idx_tuple):
            slicer[axis] = idx
            current_slice_dim_order[axis] = ''  # Mark as removed

        # Remove empty strings from current_slice_dim_order to get actual slice dim_order
        slice_dim_order_str = "".join([d for d in current_slice_dim_order if d != ''])

        sliced_arr = arr[tuple(slicer)]

        # If a channel dimension exists in the slice and it's a singleton (e.g., C=1), squeeze it
        if "C" in slice_dim_order_str:
            c_idx_in_slice = slice_dim_order_str.find("C")
            if sliced_arr.ndim > c_idx_in_slice and sliced_arr.shape[c_idx_in_slice] == 1:
                sliced_arr = np.squeeze(sliced_arr, axis=c_idx_in_slice)
                slice_dim_order_str = slice_dim_order_str.replace("C", "")

        # Ensure the array slice is 2D for computation functions expecting 2D
        if sliced_arr.ndim > 2 and "C" in slice_dim_order_str:
            # Try to convert to grayscale 2D if multiple channels are left
            try:
                sliced_arr, slice_dim_order_str = to_gray(sliced_arr, slice_dim_order_str)
            except ValueError:
                logger.warning(f"Could not convert slice to 2D grayscale for lowest level stats. Skipping stat.")
                continue  # Skip if we can't get it to 2D
        elif sliced_arr.ndim > 2 and "C" not in slice_dim_order_str:
            # If still >2D and no C, try averaging remaining non-XY dims
            original_ndim = sliced_arr.ndim
            while sliced_arr.ndim > 2:
                sliced_arr = np.mean(sliced_arr, axis=0)  # Average along first non-XY axis
                slice_dim_order_str = slice_dim_order_str[1:]  # Remove first char
            logger.warning(
                f"Lowest level stats: Reduced {original_ndim}D slice to 2D by averaging. Final shape: {sliced_arr.shape}")

        if sliced_arr.ndim != 2:  # Final check for functions expecting 2D
            logger.warning(f"Skipping stat for slice: Array not 2D after processing (shape: {sliced_arr.shape}).")
            continue

        stat_value = func(sliced_arr, slice_dim_order_str)  # Pass slice dim_order

        if stat_value is not None:
            # Construct key: e.g., "mean_C0Z1"
            key_parts = []
            for (dim_char, _), idx in zip(non_spatial, idx_tuple):
                key_parts.append(f"{dim_char.lower()}{idx}")
            key = f"{func_name}_" + "".join(key_parts)  # Combine parts with no underscore if that's desired
            detailed_stats[key] = stat_value

    return detailed_stats


def _compute_higher_level_stats(
        arr: np.ndarray,
        dim_order: str,
        non_spatial: List[tuple],
        func: Callable[[np.ndarray, str], float],  # Func takes array AND dim_order, returns float
        func_name: str,
) -> Dict[str, float]:
    """
    Computes metrics directly on higher-level splits when no aggregation is possible.
    """
    higher_level_stats = {}

    # Iterate over each non-spatial dimension and compute metrics for higher-level splits
    # These are slices where only one non-spatial dim is fixed, and others are `slice(None)`
    for dim_fixed, axis_fixed in non_spatial:
        for idx in range(arr.shape[axis_fixed]):
            slicer = [slice(None)] * arr.ndim
            slicer[axis_fixed] = idx

            slice_dim_order_list = list(dim_order)
            slice_dim_order_list[axis_fixed] = ''  # Mark fixed dim as removed for slice's dim_order
            slice_dim_order_str = "".join([d for d in slice_dim_order_list if d != ''])

            sliced_arr = arr[tuple(slicer)]

            # Similar dimension reduction logic as in _compute_lowest_level_stats
            if "C" in slice_dim_order_str:
                c_idx_in_slice = slice_dim_order_str.find("C")
                if sliced_arr.ndim > c_idx_in_slice and sliced_arr.shape[c_idx_in_slice] == 1:
                    sliced_arr = np.squeeze(sliced_arr, axis=c_idx_in_slice)
                    slice_dim_order_str = slice_dim_order_str.replace("C", "")

            if sliced_arr.ndim > 2 and "C" in slice_dim_order_str:
                try:
                    sliced_arr, slice_dim_order_str = to_gray(sliced_arr, slice_dim_order_str)
                except ValueError:
                    logger.warning(f"Could not convert slice to 2D grayscale for higher-level stats. Skipping stat.")
                    continue
            elif sliced_arr.ndim > 2 and "C" not in slice_dim_order_str:
                original_ndim = sliced_arr.ndim
                while sliced_arr.ndim > 2:
                    sliced_arr = np.mean(sliced_arr, axis=0)
                    slice_dim_order_str = slice_dim_order_str[1:]
                logger.warning(
                    f"Higher level stats: Reduced {original_ndim}D slice to 2D by averaging. Final shape: {sliced_arr.shape}")

            if sliced_arr.ndim != 2:
                logger.warning(
                    f"Skipping stat for higher-level slice: Array not 2D after processing (shape: {sliced_arr.shape}).")
                continue

            stat_value = func(sliced_arr, slice_dim_order_str)  # Pass slice dim_order

            if stat_value is not None:
                key = f"{func_name}_{dim_fixed.lower()}{idx}"
                higher_level_stats[key] = stat_value

    # Compute the metric on the entire array (global stat)
    # Ensure the main array is reduced to 2D for the global stat if needed
    global_arr = arr.copy()
    global_dim_order = dim_order

    if "C" in global_dim_order:
        c_idx = global_dim_order.find("C")
        if global_arr.ndim > c_idx and global_arr.shape[c_idx] > 1:
            try:
                global_arr, global_dim_order = to_gray(global_arr, global_dim_order)
            except ValueError:
                logger.warning("Could not convert global array to 2D grayscale for global stat. Skipping.")
                global_arr = np.array([])  # Ensure it's empty to skip stat

    if global_arr.ndim > 2:
        original_ndim = global_arr.ndim
        while global_arr.ndim > 2:
            global_arr = np.mean(global_arr, axis=0)
            global_dim_order = global_dim_order[1:]
        logger.warning(
            f"Global stat: Reduced {original_ndim}D array to 2D by averaging. Final shape: {global_arr.shape}")

    if global_arr.ndim == 2 and global_arr.size > 0:  # Ensure it's a 2D array and not empty
        global_stat = func(global_arr, global_dim_order)
        if global_stat is not None:
            higher_level_stats[func_name] = global_stat
    else:
        logger.warning(
            f"Cannot compute global stat for {func_name}: Array is not 2D or is empty (shape: {global_arr.shape}).")

    return higher_level_stats


def _compute_aggregated_stats(
        detailed_stats: Dict[str, float],
        non_spatial: List[tuple],
        agg_func: Callable[[np.ndarray], float],
        func_name: str,
        priority_order: str,
) -> Dict[str, float]:
    """
    Computes aggregated statistics for each level of the hierarchy.
    """
    agg_stats = {}

    # Iterate over each non-spatial dimension and aggregate
    # This aggregates the 'detailed_stats' (e.g., C0Z0, C0Z1) into C0, C1, Z0, Z1 etc.
    for dim, axis in non_spatial:
        group = {}
        for key, stat_value in detailed_stats.items():
            # Extract parts like "C0", "Z1" from "func_name_C0Z1"
            key_suffix = key[len(func_name) + 1:]  # "C0Z1"
            parts = [s for s in key_suffix.split('_') if s]  # Split by underscore if present, remove empty

            # Find the part corresponding to the current 'dim'
            dim_part = next((p for p in parts if p.lower().startswith(dim.lower())), None)

            if dim_part:
                group.setdefault(dim_part, []).append(stat_value)

        for part, values in group.items():
            agg_key = f"{func_name}_{part}"  # e.g., "mean_C0"
            agg_stats[agg_key] = agg_func(np.array(values))

    # Compute the final aggregated statistic based on priority order
    final_key = func_name  # This is the global stat name (e.g., "mean_intensity")
    final_values = []

    # Iterate through priority order to find a dimension to aggregate from
    for dim_char in priority_order:
        # Check if this dimension was one of our non-spatial dimensions
        if dim_char in [d for d, _ in non_spatial]:
            # Collect all aggregated stats for this dimension (e.g., mean_C0, mean_C1)
            dim_keys = [k for k in agg_stats.keys() if k.startswith(f"{func_name}_{dim_char.lower()}")]
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


# Individual stat functions (wrappers for compute_hierarchical_stats)
# These now also take `dim_order` as per the updated `compute_hierarchical_stats` signature
def calculate_mean(arr: np.array, dim_order: str) -> Dict[str, float]:
    return compute_hierarchical_stats(arr, dim_order, lambda a, d: float(np.mean(a)) if a.size > 0 else 0.0,
                                      "mean_intensity", agg_func=np.mean)


def calculate_median(arr: np.array, dim_order: str) -> Dict[str, float]:
    return compute_hierarchical_stats(arr, dim_order, lambda a, d: float(np.median(a)) if a.size > 0 else 0.0,
                                      "median_intensity", agg_func=np.median)


def calculate_std(arr: np.array, dim_order: str) -> Dict[str, float]:
    return compute_hierarchical_stats(arr, dim_order, lambda a, d: float(np.std(a)) if a.size > 0 else 0.0,
                                      "std_intensity", agg_func=np.std)


def calculate_min(arr: np.array, dim_order: str) -> Dict[str, float]:
    return compute_hierarchical_stats(arr, dim_order, lambda a, d: float(np.min(a)) if a.size > 0 else 0.0,
                                      "min_intensity", agg_func=np.min)


def calculate_max(arr: np.array, dim_order: str) -> Dict[str, float]:
    return compute_hierarchical_stats(arr, dim_order, lambda a, d: float(np.max(a)) if a.size > 0 else 0.0,
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
    return {"thumbnail": thumbnail_array.tolist()}  # Convert to list for Polars/JSON serializability


# --- 2D-specific stat functions (internal, called by hierarchical wrappers) ---
# These functions expect a 2D numpy array and don't need dim_order as much,
# but are kept for consistency with the signature of func in compute_hierarchical_stats.

def _variance_of_laplacian_2d(image: np.ndarray, dim_order: str) -> float:
    if image.ndim != 2:
        return 0.0  # Should be handled by calling function
    gray = image.astype(np.float64)
    if np.all(gray == gray.flat[0]) or gray.size == 0:
        return 0.0
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return lap.var()


def _tenengrad_2d(image: np.ndarray, dim_order: str) -> float:
    if image.ndim != 2:
        return 0.0
    gray = image.astype(np.float64)
    if np.all(gray == gray.flat[0]) or gray.size == 0:
        return 0.0
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    return np.mean(mag) if mag.size > 0 else 0.0


def _brenner_2d(image: np.ndarray, dim_order: str) -> float:
    if image.ndim != 2:
        return 0.0
    gray = image.astype(np.float32)
    if gray.size == 0 or gray.shape[1] < 3:
        return 0.0
    diff = gray[:, 2:] - gray[:, :-2]
    return np.mean(diff ** 2) if diff.size > 0 else 0.0


def _noise_estimation_2d(image: np.ndarray, dim_order: str) -> float:
    if image.ndim != 2:
        return 0.0
    gray = image.astype(np.float32)
    if gray.size == 0:
        return 0.0
    median = cv2.medianBlur(gray, 3)
    noise = gray - median
    return float(np.std(noise))


# def _wavelet_energy_2d(image: np.ndarray, dim_order: str, wavelet='db1', level=1) -> float:
#     if image.ndim != 2:
#         return 0.0
#     gray = np.float32(image)
#     if gray.size == 0:
#         return 0.0
#     try:
#         coeffs = pywt.wavedec2(gray, wavelet, level=level)
#         energy = 0.0
#         for detail in coeffs[1:]:
#             for subband in detail:
#                 energy += np.sum(np.abs(subband))
#         return energy
#     except ValueError:
#         logger.warning("Error in _wavelet_energy_2d calculation. Returning 0.0.")
#         return 0.0


def _check_blocking_artifacts_2d(gray: np.ndarray, dim_order: str) -> float:
    if gray.ndim != 2:
        return 0.0
    gray = np.float32(gray)
    if gray.size == 0:
        return 0.0

    block_size = 8
    height, width = gray.shape
    blocking_effect = 0.0
    num_boundaries = 0

    for i in range(block_size, height, block_size):
        if i < height:
            blocking_effect += np.mean(np.abs(gray[i, :] - gray[i - 1, :]))
            num_boundaries += 1
    for j in range(block_size, width, block_size):
        if j < width:
            blocking_effect += np.mean(np.abs(gray[:, j] - gray[:, j - 1]))
            num_boundaries += 1

    return blocking_effect / num_boundaries if num_boundaries > 0 else 0.0


def _check_ringing_artifacts_2d(gray: np.ndarray, dim_order: str) -> float:
    if gray.ndim != 2:
        return 0.0
    if gray.size == 0 or gray.shape[0] < 3 or gray.shape[1] < 3:
        return 0.0

    normalized_gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    edges = cv2.Canny(normalized_gray, 50, 150)
    if np.sum(edges) == 0:
        return 0.0

    kernel = np.ones((3, 3), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)
    edge_neighborhood = dilated_edges - edges

    if np.sum(edge_neighborhood > 0) == 0:
        return 0.0

    ringing_variance = np.var(gray[edge_neighborhood > 0])
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
            current_dim_order = current_dim_order.replace("C", "")

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

        img = Image.fromarray(normalized_array)
        img = img.resize((SPRITE_SIZE, SPRITE_SIZE), Image.LANCZOS)
        return np.array(img)
    except TypeError as e:
        logger.error(
            f"Error converting array to PIL Image or resizing for thumbnail: {e}. Array shape: {normalized_array.shape}, dtype: {normalized_array.dtype}")
        return np.array([])
