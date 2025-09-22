import logging
from typing import Dict, Callable, Any, Optional, List, Tuple

import cv2
import dask.array as da
import numpy as np

from pixel_patrol_base.utils.array_utils import calculate_sliced_stats
from pixel_patrol_base.core.record import Record
from pixel_patrol_base.core.specs import RecordSpec

logger = logging.getLogger(__name__)


def _column_fn_registry() -> Dict[str, Dict[str, Callable]]:
    return {
        "laplacian_variance": {"fn": _variance_of_laplacian_2d, "agg": da.mean},
        "tenengrad": {"fn": _tenengrad_2d, "agg": da.mean},
        "brenner": {"fn": _brenner_2d, "agg": da.mean},
        "noise_std": {"fn": _noise_estimation_2d, "agg": da.mean},
        "blocking_records": {"fn": _check_blocking_records_2d, "agg": da.mean},
        "ringing_records": {"fn": _check_ringing_records_2d, "agg": da.mean},
    }


def calculate_np_array_stats(array: da.array, dim_order: str) -> Dict[str, float]:
    registry = _column_fn_registry()
    all_metrics = {k: v["fn"] for k, v in registry.items()}
    all_aggregators = {k: v["agg"] for k, v in registry.items() if v["agg"] is not None}
    return calculate_sliced_stats(array, dim_order, all_metrics, all_aggregators)


def _prepare_2d_image(image: np.ndarray) -> Optional[np.ndarray]:
    if image.ndim != 2 or image.size == 0 or image.dtype == bool:
        return None
    return image.astype(np.float32)


def _variance_of_laplacian_2d(image: np.ndarray) -> float:
    image = _prepare_2d_image(image)
    if image is None:
        return float(np.nan)
    lap = cv2.Laplacian(image, cv2.CV_32F)
    return float(lap.var())


def _tenengrad_2d(image: np.ndarray) -> float:
    image = _prepare_2d_image(image)
    if image is None or np.all(image == image.flat[0]):
        return float(np.nan)
    gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    return float(np.mean(mag)) if mag.size > 0 else 0.0


def _brenner_2d(image: np.ndarray) -> float:
    image = _prepare_2d_image(image)
    if image is None:
        return float(np.nan)
    diff = image[:, 2:] - image[:, :-2]
    return float(np.mean(diff ** 2)) if diff.size > 0 else 0.0


def _noise_estimation_2d(image: np.ndarray) -> float:
    image = _prepare_2d_image(image)
    if image is None:
        return float(np.nan)
    median = cv2.medianBlur(image, 3)
    noise = image - median
    return float(np.std(noise))


def _check_blocking_records_2d(image: np.ndarray) -> float:
    image = _prepare_2d_image(image)
    if image is None:
        return float(np.nan)

    block_size = 8
    height, width = image.shape
    blocking_effect = 0.0
    num_boundaries = 0

    for i in range(block_size, height, block_size):
        if i < height:
            blocking_effect += float(np.mean(np.abs(image[i, :] - image[i - 1, :])))
            num_boundaries += 1
    for j in range(block_size, width, block_size):
        if j < width:
            blocking_effect += float(np.mean(np.abs(image[:, j] - image[:, j - 1])))
            num_boundaries += 1

    return blocking_effect / num_boundaries if num_boundaries > 0 else 0.0


def _check_ringing_records_2d(image: np.ndarray) -> float:
    image = _prepare_2d_image(image)
    if image is None:
        return float(np.nan)
    normalized_image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
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


class QualityMetricsProcessor:
    """
    Extracts image quality metrics (tenengrad, brenner, noise, etc.) from XY slices.
    """

    # Declarative plugin metadata
    NAME = "quality-metrics"
    INPUT = RecordSpec(axes={"X", "Y"}, kinds={"intensity"}, capabilities={"spatial-2d"})
    OUTPUT = "features"  # or "record" if this produced another image

    # Table schema (static + dynamic)
    OUTPUT_SCHEMA: Dict[str, Any] = {name: float for name in _column_fn_registry().keys()}
    # e.g. tenengrad_C0_Z3, brenner_T5, etc.
    OUTPUT_SCHEMA_PATTERNS: List[Tuple[str, Any]] = [
        (rf"^(?:{name})_[a-zA-Z]\d+(_[a-zA-Z]\d+)*$", float)
        for name in _column_fn_registry().keys()
    ]

    def run(self, art: Record) -> Dict[str, float]:
        dim_order = art.dim_order
        return calculate_np_array_stats(art.data, dim_order)
