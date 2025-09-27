import pytest
from pixel_patrol_base.core.record import _infer_dim_order, _infer_dim_names
from pixel_patrol_base.core.record import record_from

class StubArr:
    def __init__(self, shape):
        self.shape = shape
        self.ndim = len(shape)

def test_fallback_order_and_names():
    a = StubArr((5, 6))
    meta = {}
    record = record_from(a, meta)
    order = record.dim_order
    names = record.dim_names
    assert order == "AB"
    assert names == ["dimA", "dimB"]


def test_meta_order_preserved_and_names_from_order():
    a = StubArr((2, 3, 4, 5, 6))
    meta = {"dim_order": "TCZYX", "ndim": 5}
    order = _infer_dim_order(meta)
    names = _infer_dim_names(order, meta)
    assert order == "TCZYX"
    assert names == ["T", "C", "Z", "Y", "X"]

def test_meta_names_preferred_when_length_matches():
    a = StubArr((3, 4, 5))
    meta = {"dim_names": ["time", "channel", "z"]}
    record = record_from(a, meta)
    order = record.dim_order
    names = record.dim_names
    assert order == "ABC"
    assert names == ["time", "channel", "z"]

def test_meta_names_ignored_on_length_mismatch():
    a = StubArr((3, 4, 5))
    meta = {"dim_names": ["t", "c"]}  # too short → ignore
    record = record_from(a, meta)
    order = record.dim_order
    names = record.dim_names
    assert order == "ABC"
    assert names == [f"dim{c}" for c in 'ABC']

def test_invalid_meta_order_falls_back_and_names_follow():
    a = StubArr((3, 4))
    meta = {"dim_order": "TimeChannel", "ndim": 2}  # not single-letter
    record = record_from(a, meta)
    order = record.dim_order
    names = record.dim_names
    assert order == "AB"
    assert names == ["dimA", "dimB"]


def test_meta_names_wrong_types_ignored():
    a = StubArr((3, 4, 5))
    meta = {"dim_names": ["t", 1, "z"]}
    record = record_from(a, meta)
    names = record.dim_names
    order = record.dim_order
    assert order == "ABC"
    assert names == ["dimA", "dimB", "dimC"]

def test_meta_order_non_alpha():
    a = StubArr((2, 3))
    meta = {"dim_order": "T1", "ndim": 2}  # non-alpha → ignore
    record = record_from(a, meta)
    names = record.dim_names
    order = record.dim_order
    assert order == "T1"
    assert names == ["T", "1"]
