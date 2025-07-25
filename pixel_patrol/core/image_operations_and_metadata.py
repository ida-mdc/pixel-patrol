import fnmatch
from itertools import product, combinations
from typing import Callable, Tuple ,Dict, List, Any, NamedTuple, Optional

import logging
from pathlib import Path

import cv2
import numpy as np
# import pywt
import bioio_base
from bioio import BioImage
import bioio_imageio

from pixel_patrol.config import STANDARD_DIM_ORDER, SLICE_AXES, RGB_WEIGHTS, SPRITE_SIZE

logger = logging.getLogger(__name__)

class SliceAxisSpec(NamedTuple):
    dim: str    # e.g. "T", "C" or "Z"
    idx: int   # index in dim_order
    size: int   # shape along that axis


def _column_fn_registry() -> Dict[str, Dict[str, Callable]]:
    return {
        "mean_intensity": {
            "fn": lambda a: float(np.mean(a)) if a.size else np.nan,
            "agg": np.mean,
        },
        "median_intensity": {
            "fn": lambda a: float(np.median(a)) if a.size else np.nan,
            "agg": np.median,
        },
        "std_intensity": {
            "fn": lambda a: float(np.std(a)) if a.size else np.nan,
            "agg": np.mean,
        },
        "min_intensity": {
            "fn": lambda a: float(np.min(a)) if a.size else np.nan,
            "agg": np.min,
        },
        "max_intensity": {
            "fn": lambda a: float(np.max(a)) if a.size else np.nan,
            "agg": np.max,
        },
        "laplacian_variance": {"fn": _variance_of_laplacian_2d, "agg": np.mean},
        "tenengrad": {"fn": _tenengrad_2d, "agg": np.mean},
        "brenner": {"fn": _brenner_2d, "agg": np.mean},
        "noise_std": {"fn": _noise_estimation_2d, "agg": np.mean},
        "blocking_artifacts": {"fn": _check_blocking_artifacts_2d, "agg": np.mean},
        "ringing_artifacts": {"fn": _check_ringing_artifacts_2d, "agg": np.mean},
        "thumbnail": {"fn": _generate_thumbnail, "agg": None},  # No aggregation needed
        "histogram": {"fn": lambda a: _histogram_func(a) if a.size else np.nan, "agg": _histogram_agg},
    }


def _mapping_for_bioimage_metadata_by_column_name() -> (
    Dict[str, Callable[[BioImage], Dict]]
):
    """
    Maps requested metadata fields to functions that extract them from a BioImage object.
    These functions return a dictionary with the extracted key-value pair.
    """
    return {
        "n_images": lambda img: {
            "n_images": len(img.scenes) if hasattr(img, "scenes") else 1
        },
        "pixel_size_X": lambda img: {
            "pixel_size_X": getattr(img.physical_pixel_sizes, "X", None)
        },
        "pixel_size_Y": lambda img: {
            "pixel_size_Y": getattr(img.physical_pixel_sizes, "Y", None)
        },
        "pixel_size_Z": lambda img: {
            "pixel_size_Z": getattr(img.physical_pixel_sizes, "Z", None)
        },
        "pixel_size_t": lambda img: {
            "pixel_size_t": getattr(img.physical_pixel_sizes, "T", None)
        },
        "channel_names": lambda img: {"channel_names": img.channel_names},
        "ome_metadata": lambda img: {"ome_metadata": img.ome_metadata},
    }


def available_columns() -> List[str]:
    keys = list(_column_fn_registry().keys())
    keys.extend(_mapping_for_bioimage_metadata_by_column_name())
    # TODO: probably hard coded should be removed
    keys.extend(["dim_order", "t_size", "c_size", "z_size", "y_size", "x_size", "s_size", "ndim", "shape", "dtype", "num_pixels"])
    return keys


def is_column_matches(column: str, columns_requested: List[str]) -> bool:
    """Check if column matches any entry in columns_requested (supporting wildcards)."""
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
    for key, fn in mapping.items():
        if is_column_matches(key, required_columns):
            try:
                result = fn(img)
                if isinstance(result, dict):
                    metadata.update(result)
                else:
                    logger.warning(
                        f"Extractor for column '{key}' returned non-dictionary result: {result}. Skipping update."
                    )
            except Exception as e:
                logger.warning(
                    f"Failed to extract metadata for column '{key}' from image. Error: {e}"
                )

    return metadata

def _get_standardized_array_and_dim_metadata(np_array: Any, dim_order: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Retrieves the image data, standardizes its STANDARD_DIM_ORDER,
    and extracts all dimension-related metadata from the standardized array.

    Returns:
        A tuple containing:
        - np_array: The standardized NumPy array.
        - dim_metadata: A dictionary with dimension-related metadata (t_size, c_size, etc., ndim, shape, dtype).
    """

    if dim_order != STANDARD_DIM_ORDER:
        # Pad missing dimensions with size 1
        for i, d in enumerate(STANDARD_DIM_ORDER):
            if d not in dim_order:
                np_array = np.expand_dims(np_array, axis=i)
                dim_order = dim_order[:i] + d + dim_order[i:]
        # Transpose to the standard dimension order
        np_array = np.transpose(np_array, [dim_order.index(d) for d in STANDARD_DIM_ORDER])

    # Extract dimension-related metadata from the standardized array
    dim_metadata: Dict[str, Any] = {"dim_order": STANDARD_DIM_ORDER, "shape": str(np_array.shape),
                                    "ndim": np_array.ndim, "dtype": str(np_array.dtype),
                                    "num_pixels": int(np_array.size)}

    # Extract individual dimension sizes from the standardized shape
    shape_map = dict(zip(STANDARD_DIM_ORDER, np_array.shape))
    dim_metadata["t_size"] = shape_map.get("T")
    dim_metadata["c_size"] = shape_map.get("C")
    dim_metadata["z_size"] = shape_map.get("Z")
    dim_metadata["y_size"] = shape_map.get("Y")
    dim_metadata["x_size"] = shape_map.get("X")
    dim_metadata["s_size"] = shape_map.get("S")

    return np_array, dim_metadata


def get_all_image_properties(file_path: Path, required_columns: List[str]) -> Dict:

    if not file_path.exists():
        logger.warning(f"File not found: '{file_path}'. Cannot extract metadata.")
        return {}
    img = _load_bioio_image(file_path)

    if img is None:
        logger.warning(f"Failed to load image from '{file_path}'. Cannot extract metadata.")
        return {}
    np_array = img.get_image_data() # TODO: test speed with .compute() at the end and without it

    if np_array is not None and np_array.size > 0:
        np_array, all_image_properties = _get_standardized_array_and_dim_metadata(np_array, img.dims.order)
        bioimage_mapping = _mapping_for_bioimage_metadata_by_column_name()
        all_image_properties.update(_extract_metadata_from_mapping(img, bioimage_mapping, required_columns))

        calculate_np_array_stats(required_columns, all_image_properties, np_array)
        return all_image_properties
    else:
        logger.warning(f"NumPy array for file '{file_path}' is empty. Skipping image.")
        return {}


def _maybe_gray_scale(array):
    s_idx = STANDARD_DIM_ORDER.index("S")
    if array.shape[s_idx] > 1:
        try:
            array = to_gray(array, s_idx)
        except ValueError as e: # TODO: check!! Can we have cased of S!=1/3/4? If not, should remove the try block
            logger.warning(f"Gray conversion failed: {e}")
    return array

# TODO: Func is too long and complex. Improve!
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

    if color_dim_size == 4: # TODO: we assume RGBA - need to check all types of S dim.
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


def calculate_np_array_stats(columns: List[str], all_image_properties: Dict[str, Any], array: np.ndarray) -> None:
    gray_arr = _maybe_gray_scale(array)
    registry = _column_fn_registry()
    all_metrics = {k: v['fn'] for k, v in registry.items() if k in columns and k != 'thumbnail'}
    all_aggregators = {k: v['agg'] for k, v in registry.items() if k in columns and v['agg'] is not None}
    if all_metrics:
        results = compute_hierarchical_stats(gray_arr, STANDARD_DIM_ORDER, all_metrics, all_aggregators)
        for key, value in results.items():
            all_image_properties[key] = value
    if 'thumbnail' in columns:
        try:
            all_image_properties['thumbnail'] = _generate_thumbnail(array, STANDARD_DIM_ORDER)
        except (ValueError, IOError) as e:
            logger.warning(f"Error generating thumbnail - {e}.")


def compute_hierarchical_stats(arr: np.ndarray, dim_order: str, metrics_fns: Dict[str, Callable[[np.ndarray], Any]], agg_fns: Dict[str, Callable[[np.ndarray], Any]]) -> Dict[str, float]:
    if len(dim_order) != arr.ndim:
        raise ValueError(f"Dimension order string '{dim_order}' length ({len(dim_order)}) does not match "
                         f"the number of array dimensions ({arr.ndim}).")

    slice_axes_specs = [
        SliceAxisSpec(dim, idx, arr.shape[idx])
        for idx, dim in enumerate(dim_order)
        if dim in SLICE_AXES
    ]

    if "X" not in dim_order or "Y" not in dim_order or arr.shape[dim_order.index("X")] == 1 or arr.shape[
        dim_order.index("Y")] == 1:
        logger.warning("Image lacks 2D spatial dimensions or has 1-pixel X/Y. Skipping.")
        return {}

    elif not slice_axes_specs or all(spec.size == 1 for spec in slice_axes_specs):
        logger.info("Image has only one sliceable instance. Calculating global metrics for the 2D plane.")
        arr_squeezed = np.squeeze(arr)
        return {name: fn(arr_squeezed) for name, fn in metrics_fns.items()}

    results = {}

    detailed_metrics = _compute_all_slices_combo_metrics(arr, slice_axes_specs, metrics_fns)
    results.update(detailed_metrics)

    if agg_fns:
        aggregated_metrics = _compute_aggregated_combo_metrics(detailed_metrics, slice_axes_specs, agg_fns)
        results.update(aggregated_metrics)

    no_agg = {m: fn for m, fn in metrics_fns.items() if m not in agg_fns}
    if no_agg:
        parent_metrics = _compute_parent_slice_combo_metrics(arr, slice_axes_specs, no_agg)
        results.update(parent_metrics)

    return results


def _compute_all_slices_combo_metrics(arr: np.ndarray,
                                      slice_axes_specs: List[SliceAxisSpec],
                                      metrics_fns: Dict[str, Callable]
                                      ) -> Dict[str, Any]:

    detailed_stats = {}
    slice_axes_ranges = [range(spec.size) for spec in slice_axes_specs]

    for idx_tuple in product(*slice_axes_ranges):
        slicer = [slice(None)] * arr.ndim
        key_suffix = []

        for spec, slice_idx in zip(slice_axes_specs, idx_tuple):
            slicer[spec.idx] = slice_idx
            key_suffix.append(f"{spec.dim.lower()}{slice_idx}")

        sliced_arr = np.squeeze(arr[tuple(slicer)])

        if sliced_arr.ndim != 2:  # Final check for functions expecting 2D
            logger.warning(f"Skipping stat for slice: Array not 2D after processing (shape: {sliced_arr.shape}).")
            continue

        for metric_name, fn in metrics_fns.items():
            detailed_stats[f"{metric_name}_{'_'.join(key_suffix)}"] = fn(sliced_arr)

    return detailed_stats


def _compute_aggregated_combo_metrics(
        detailed_stats: Dict[str, float],
        slice_axes_specs: List[SliceAxisSpec],
        agg_funcs: Dict[str, Callable[[np.ndarray], float]],
) -> Dict[str, float]:
    agg_stats = {}

    # 1) drop any axis that can't slice
    active = [s for s in slice_axes_specs if s.size > 1]
    axes   = [spec.dim.lower()       for spec in active]
    n      = len(axes)

    # 2) parse per‐metric entries
    parsed: Dict[str, List[Tuple[Dict[str,str], float]]] = {}

    axis_prefixes = {spec.dim.lower()[0] for spec in slice_axes_specs}
    for key, val in detailed_stats.items():
        parts = key.split("_")
        split_i = next(
            (i for i, p in enumerate(parts) if p[0].lower() in axis_prefixes and p[1:].isdigit()),
            len(parts))
        metric = "_".join(parts[:split_i])
        suffix = "_".join(parts[split_i:])

        parts = suffix.split("_")                 # ["t0","c1","z2"]
        dmap  = {p[0].lower(): p for p in parts}  # {'t':"t0", 'c':"c1", 'z':"z2"}
        parsed.setdefault(metric, []).append((dmap, val))

    # 3) for each metric, build all parent aggregates
    for metric, entries in parsed.items():
        agg = agg_funcs[metric]

        # a) every subset of axes size 1..n-1
        for r in range(1, n):
            for dims in combinations(axes, r):
                bucket: Dict[str, List[float]] = {}
                for dmap, val in entries:
                    if all(d in dmap for d in dims):
                        combo_key = "_".join(dmap[d] for d in dims)
                        bucket.setdefault(combo_key, []).append(val)
                for combo_key, vals in bucket.items():
                    result = agg(np.array(vals))
                    agg_stats[f"{metric}_{combo_key}"] = result

        # b) global
        all_vals = [val for _, val in entries]
        result = agg(np.array(all_vals))
        agg_stats[metric] = result

    return agg_stats


def _compute_parent_slice_combo_metrics(
        arr: np.ndarray,
        slice_axes_specs: List[SliceAxisSpec],
        metrics_fns: Dict[str, Callable[[np.ndarray], float]],
) -> Dict[str, float]:
    parent_stats = {}
    # only axes with more than one slice
    active = [s for s in slice_axes_specs if s.size > 1]
    n = len(active)

    # all actual slice‐combos of size 1..n−1
    for r in range(1, n):
        for specs in combinations(active, r):
            ranges = [range(s.size) for s in specs]
            for idxs in product(*ranges):
                slicer = [slice(None)] * arr.ndim
                parts = []
                for spec, idx in zip(specs, idxs):
                    slicer[spec.idx] = idx
                    parts.append(f"{spec.dim.lower()}{idx}")
                sub = np.squeeze(arr[tuple(slicer)])
                for name, fn in metrics_fns.items():
                    parent_stats[f"{name}_{'_'.join(parts)}"] = fn(sub)

    # global
    whole = np.squeeze(arr)
    for name, fn in metrics_fns.items():
        parent_stats[name] = fn(whole)

    return parent_stats


def _prepare_2d_image(image: np.ndarray) -> Optional[np.ndarray]:
    if image.ndim != 2 or image.size == 0: # TODO: Remove cols (and maybe cols that are all nans) - look into polars check of column empty
        return None
    img2d = image.astype(np.float32)
    return img2d

def _variance_of_laplacian_2d(image: np.ndarray) -> float:
    image = _prepare_2d_image(image)
    if image is None:
        return np.nan
    lap = cv2.Laplacian(image, cv2.CV_32F)
    return lap.var()


def _tenengrad_2d(image: np.ndarray) -> float:
    image = _prepare_2d_image(image)
    if image is None or np.all(image == image.flat[0]):
        return np.nan
    gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    return np.mean(mag) if mag.size > 0 else 0.0


def _brenner_2d(image: np.ndarray) -> float:
    image = _prepare_2d_image(image)
    if image is None or image.shape[1] < 3:
        return np.nan
    diff = image[:, 2:] - image[:, :-2]
    return np.mean(diff ** 2) if diff.size > 0 else 0.0


def _noise_estimation_2d(image: np.ndarray) -> float:
    image = _prepare_2d_image(image)
    if image is None:
        return np.nan
    median = cv2.medianBlur(image, 3)
    noise = image - median
    return float(np.std(noise))


def _check_blocking_artifacts_2d(image: np.ndarray) -> float:
    image = _prepare_2d_image(image)
    if image is None:
        return np.nan

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
    if image.ndim != 2 or image.size == 0 or image.shape[0] < 3 or image.shape[1] < 3:
        return np.nan
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

# TODO: decide how we create thumbnail for images with S (probably call to gray) and C
def _generate_thumbnail(np_array: np.array, dim_order: str) -> np.array:
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
        if dim not in ["X", "Y"]:  # Reduce non-spatial, non-channel dimensions
            center_index = arr_to_process.shape[i] // 2
            arr_to_process = np.take(arr_to_process, indices=center_index, axis=i)
            current_dim_order = current_dim_order.replace(dim, "")
            # Do not increment i, as the array shrinks and the next dim is now at current i
        else:
            i += 1

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


def _histogram_func(image_array: np.ndarray, bins: int = 256) -> np.ndarray:
    """
    Calculate histogram of an array using numpy's histogram function, but ensure it uses 256 bins.
    """
    hist, _ = np.histogram(a=image_array, bins=bins, range=(0, image_array.max()))
    return hist


def _histogram_agg(hist_list: list[np.ndarray], bins: int = 256) -> np.ndarray:
    """
    Aggregate a list of histograms by summing them.
    """
    if len(hist_list) == 0:
        return np.zeros(bins, dtype=int)
    return np.sum(np.stack(hist_list), axis=0)


def calculate_histogram(
    arr: np.ndarray, dim_order: str, bins: int = 256
) -> Dict[str, np.ndarray]:
    """
    Calculate the histogram of an array across all dimensions and return it as a dictionary.
    Uses 256 bins and accumulates the histogram across all dimensions.
    """
    return compute_hierarchical_stats(
        arr,
        dim_order,
        lambda a, d: _histogram_func(a, bins),  # Accepts both array and dim_order
        "histogram",
        agg_func=_histogram_agg,
    )
