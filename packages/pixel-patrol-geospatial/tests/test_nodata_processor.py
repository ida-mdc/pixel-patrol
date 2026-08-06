import dask.array as da
import numpy as np

from pixel_patrol_base.core.record import record_from

from pixel_patrol_geospatial.nodata_processor import NoDataCountProcessor


def _make_record(arr, nodata_value):
    return record_from(arr, {"dim_order": "CYX", "nodata_value": nodata_value}, kind="intensity")


def test_no_nodata_value_returns_zero():
    arr = np.ones((1, 4, 4), dtype=np.uint8)
    record = _make_record(arr, nodata_value=None)
    result = NoDataCountProcessor().run_chunk(record)
    assert result["nodata_count"] == 0


def test_integer_nodata_counts_matches():
    arr = np.array([[[0, 1], [0, 2]]], dtype=np.uint8)
    record = _make_record(arr, nodata_value=0)
    result = NoDataCountProcessor().run_chunk(record)
    assert result["nodata_count"] == 2


def test_integer_nodata_no_match_returns_zero():
    arr = np.ones((1, 3, 3), dtype=np.uint8)
    record = _make_record(arr, nodata_value=255)
    result = NoDataCountProcessor().run_chunk(record)
    assert result["nodata_count"] == 0


def test_nan_nodata_counts_nans():
    arr = np.array([[[float("nan"), 1.0], [2.0, float("nan")]]], dtype=np.float32)
    record = _make_record(arr, nodata_value=float("nan"))
    result = NoDataCountProcessor().run_chunk(record)
    assert result["nodata_count"] == 2


def test_dask_array_gives_same_result_as_numpy():
    np_arr = np.array([[[255, 1], [255, 4]]], dtype=np.uint8)
    dask_arr = da.from_array(np_arr, chunks=(1, 2, 2))

    np_record = _make_record(np_arr, nodata_value=255)
    dask_record = _make_record(dask_arr, nodata_value=255)

    proc = NoDataCountProcessor()
    assert proc.run_chunk(np_record)["nodata_count"] == proc.run_chunk(dask_record)["nodata_count"]
