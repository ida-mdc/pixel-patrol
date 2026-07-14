import numpy as np

from pixel_patrol_base.core.processing import _extract_image_meta
from pixel_patrol_base.core.record import record_from


def _record(meta, shape=(2, 3), dim_order="YX"):
    arr = np.zeros(shape, dtype="uint8")
    meta = {**meta, "dim_order": dim_order}
    return record_from(arr, meta, kind="intensity")


def test_dim_names_survives():
    record = _record({"dim_names": ["row", "col"]})
    meta = _extract_image_meta(record)
    assert meta["dim_names"] == ["row", "col"]


def test_dim_coordinate_keys_are_dropped():
    record = _record({"dim_y": 5, "dim_x": 7})
    meta = _extract_image_meta(record)
    assert "dim_y" not in meta
    assert "dim_x" not in meta


def test_shape_and_num_pixels_are_dropped():
    record = _record({"shape": [2, 3], "num_pixels": 6})
    meta = _extract_image_meta(record)
    assert "shape" not in meta
    assert "num_pixels" not in meta


def test_dim_order_is_recomputed_not_dropped():
    record = _record({}, dim_order="YX")
    meta = _extract_image_meta(record)
    assert meta["dim_order"] == "YX"
