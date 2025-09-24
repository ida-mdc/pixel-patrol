# packages/pixel-patrol-base/tests/test_dims.py
import pytest
from pixel_patrol_base.core.record import _infer_dim_order, _infer_dim_names

class StubArr:
    def __init__(self, shape):
        self.shape = shape
        self.ndim = len(shape)

def test_fallback_order_and_names():
    a = StubArr((5, 6))
    meta = {}
    order = _infer_dim_order(a, meta)
    names = _infer_dim_names(order, meta)
    assert order == "AB"
    assert names == ["dimA", "dimB"]

def test_meta_order_preserved_and_names_from_order():
    a = StubArr((2, 3, 4, 5, 6))
    meta = {"dim_order": "TCZYX", "ndim": 5}
    order = _infer_dim_order(a, meta)
    names = _infer_dim_names(order, meta)
    assert order == "TCZYX"
    assert names == ["t", "c", "z", "y", "x"]

def test_meta_names_preferred_when_length_matches():
    a = StubArr((3, 4, 5))
    meta = {"dim_names": ["time", "channel", "z"]}
    order = _infer_dim_order(a, {})             # no order in meta → fallback "ABC"
    names = _infer_dim_names(order, meta)       # length matches → keep meta names
    assert order == "ABC"
    assert names == ["time", "channel", "z"]

def test_meta_names_ignored_on_length_mismatch():
    a = StubArr((3, 4, 5))
    meta = {"dim_names": ["t", "c"]}  # too short → ignore
    order = _infer_dim_order(a, {})
    names = _infer_dim_names(order, meta)
    assert names == [f"dim{c}" for c in order]

def test_invalid_meta_order_falls_back_and_names_follow():
    a = StubArr((3, 4))
    meta = {"dim_order": "TimeChannel", "ndim": 2}  # not single-letter
    order = _infer_dim_order(a, meta)
    names = _infer_dim_names(order, {})
    assert order == "AB"
    assert names == ["dimA", "dimB"]

def test_no_order_numeric_fallback():
    # call names with empty order → should use meta['ndim'] → dim1, dim2, dim3
    names = _infer_dim_names("", {"ndim": 3})
    assert names == ["dim1", "dim2", "dim3"]

def test_meta_names_wrong_types_ignored():
    a = StubArr((3, 4, 5))
    meta = {"dim_names": ["t", 1, "z"]}  # invalid → ignore
    order = _infer_dim_order(a, {})
    names = _infer_dim_names(order, meta)
    assert order == "ABC"
    assert names == ["dimA", "dimB", "dimC"]

def test_meta_order_nonalpha_ignored():
    a = StubArr((2, 3))
    meta = {"dim_order": "T1", "ndim": 2}  # non-alpha → ignore
    order = _infer_dim_order(a, meta)
    names = _infer_dim_names(order, {})
    assert order == "AB"
    assert names == ["dimA", "dimB"]
