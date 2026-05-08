"""Histogram edge helpers shared by report aggregation and processors."""

from __future__ import annotations

from typing import Tuple

import dask.array as da
import numpy as np


def safe_hist_range(x: da.Array | np.ndarray) -> Tuple[float, float, float]:
    """
    Return ``(min, max, max_adj)`` so the last bin includes ``max`` with a right edge strictly above ``max``.

    Works with Dask arrays (computes only min/max) or NumPy.
    """
    if isinstance(x, da.Array):
        min_val, max_val = da.compute(da.nanmin(x), da.nanmax(x))
        min_val, max_val = np.float64(min_val), np.float64(max_val)
    else:
        xa = np.asarray(x)
        if xa.size == 0 or not np.any(np.isfinite(xa)):
            return 0.0, 0.0, 1.0
        min_val, max_val = np.float64(np.nanmin(xa)), np.float64(np.nanmax(xa))

    if not np.isfinite(min_val) or not np.isfinite(max_val):
        return 0.0, 0.0, 1.0

    try:
        dtype = x.dtype
    except AttributeError:
        dtype = None

    if dtype is not None and np.dtype(dtype) == np.dtype("uint8"):
        return 0.0, 255.0, 256.0

    if dtype is not None and np.issubdtype(dtype, np.integer):
        max_adj = max_val + 1.0
    else:
        if min_val == max_val:
            max_adj = max_val + 1.0
        else:
            max_adj = np.nextafter(max_val, np.inf)
    return float(min_val), float(max_val), float(max_adj)
