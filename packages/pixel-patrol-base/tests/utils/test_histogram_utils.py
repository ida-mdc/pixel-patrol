import numpy as np

from pixel_patrol_base.utils.histogram_utils import safe_hist_range


def test_safe_hist_range_uint8():
    x = np.array([0, 255], dtype=np.uint8)
    mn, mx, adj = safe_hist_range(x)
    assert mn == 0.0 and mx == 255.0 and adj == 256.0


def test_safe_hist_range_float():
    x = np.array([0.0, 1.0], dtype=np.float32)
    mn, mx, adj = safe_hist_range(x)
    assert mn == 0.0 and mx == 1.0
    assert adj > mx


def test_safe_hist_range_all_nan():
    x = np.full((3, 3), np.nan)
    mn, mx, adj = safe_hist_range(x)
    assert mn == 0.0 and mx == 0.0 and adj == 1.0
