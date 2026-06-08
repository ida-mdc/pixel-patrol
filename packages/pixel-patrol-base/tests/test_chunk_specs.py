"""Unit tests for _resolve_leaf_block_shape and _compute_memory_chunk_specs."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from pixel_patrol_base.core.contracts import FileInfo
from pixel_patrol_base.core.processing import (
    _compute_memory_chunk_specs,
    _resolve_leaf_block_shape,
)

_DUMMY_PATH = Path("/mock/file")


def _specs(shape, dim_order, mb_per_task, leaf_block_shape=None, dtype=np.float32):
    info = FileInfo(shape=shape, dtype=dtype, dim_order=dim_order)
    return _compute_memory_chunk_specs(_DUMMY_PATH, info, mb_per_task, leaf_block_shape)


# ── _resolve_leaf_block_shape ─────────────────────────────────────────────────

def test_resolve_defaults():
    block = _resolve_leaf_block_shape(("Z", "C", "T", "S", "Y", "X"), None)
    for dim in ("Z", "C", "T", "S"):
        assert block[dim] == 1
    for dim in ("Y", "X"):
        assert block[dim] == -1


def test_resolve_user_spec_overrides():
    block = _resolve_leaf_block_shape(("Z", "Y", "X"), {"Z": -1, "X": 16})
    assert block["Z"] == -1
    assert block["X"] == 16
    assert block["Y"] == -1


def test_resolve_unknown_user_spec_key_is_ignored():
    block = _resolve_leaf_block_shape(("Y", "X"), {"Q": 5})
    assert "Q" not in block
    assert block["Y"] == -1


# ── _compute_memory_chunk_specs ───────────────────────────────────────────────

def test_file_fits_in_budget_returns_none():
    assert _specs((64, 64), ("Y", "X"), 1.0) is None


def test_2d_yx_no_slice_size_splits_by_budget():
    # Geometric distribution: both Y and X share the reduction equally → 2×2 chunks.
    specs = _specs((256, 256), ("Y", "X"), 0.1)
    assert specs is not None
    assert len(specs) == 4
    assert all(s.slices[0] != slice(None) for s in specs)  # Y split
    assert all(s.slices[1] != slice(None) for s in specs)  # X split


def test_3d_zyx_no_slice_size_all_dims_split():
    # Z (leaf=1, processed first), then Y and X (leaf=-1) all share the reduction.
    specs = _specs((2, 256, 256), ("Z", "Y", "X"), 0.1)
    assert specs is not None
    assert len(specs) > 1
    assert all(s.slices[0] != slice(None) for s in specs)  # Z split
    assert all(s.slices[1] != slice(None) for s in specs)  # Y split
    assert all(s.slices[2] != slice(None) for s in specs)  # X split


def test_must_split_two_dims_when_one_is_not_enough():
    # Very tight budget: Z splits first (leaf=1), then Y and X (leaf=-1).
    specs = _specs((2, 512, 512), ("Z", "Y", "X"), 0.001)
    assert specs is not None
    assert any(s.slices[0] != slice(None) for s in specs)  # Z split
    assert any(s.slices[1] != slice(None) for s in specs)  # Y split
    assert any(s.slices[2] != slice(None) for s in specs)  # X split


def test_non_divisible_large_image_with_slice_size():
    # Originally failing: non-divisible dims with slice_size caused OOM
    specs = _specs((40, 53638, 62366), ("Z", "Y", "X"), 512.0, {"X": 1024, "Y": 1024},
                   dtype=np.uint16)
    assert specs is not None
    assert len(specs) > 1
    budget_bytes = int(512 * 1024 * 1024)
    for s in specs:
        elems = 1
        for size, slc in zip((40, 53638, 62366), s.slices):
            elems *= size if slc == slice(None) else slc.stop - slc.start
        assert elems * 2 <= budget_bytes
    for s in specs:
        if s.slices[1] != slice(None):
            assert s.slices[1].start % 1024 == 0
        if s.slices[2] != slice(None):
            assert s.slices[2].start % 1024 == 0


def test_each_chunk_fits_in_budget():
    budget_mb = 0.05
    budget_bytes = int(budget_mb * 1024 * 1024)
    specs = _specs((4, 192, 300), ("Z", "Y", "X"), budget_mb, {"Y": 32})
    assert specs is not None
    for s in specs:
        elems = 1
        for size, slc in zip((4, 192, 300), s.slices):
            elems *= size if slc == slice(None) else slc.stop - slc.start
        assert elems * 4 <= budget_bytes


def test_cartesian_product_no_duplicate_combos():
    specs = _specs((6, 64, 64), ("Z", "Y", "X"), 0.001)
    assert specs is not None
    assert len({s.slices for s in specs}) == len(specs)


def test_chunk_spec_metadata():
    shape = (2, 256, 128)
    dim_order = ("Z", "Y", "X")
    specs = _specs(shape, dim_order, 0.05, {"Y": 32, "Z": 1})
    assert specs is not None
    for s in specs:
        assert s.image_shape == shape
        assert s.dim_order == dim_order


def test_1d_array_splits():
    specs = _specs((1000,), ("Z",), 1 / 1024, {"Z": 100})
    assert specs is not None
    assert sum(s.slices[0].stop - s.slices[0].start for s in specs) == 1000


# ── exact slice values ────────────────────────────────────────────────────────
# Each test pins the expected slice boundaries to catch regressions.

def test_exact_2d_no_slice_size():
    # Y and X share the 2.5× reduction equally → 2 strips each, 4 chunks of 128×128.
    specs = _specs((256, 256), ("Y", "X"), 0.1)
    assert {s.slices for s in specs} == {
        (slice(0,   128), slice(0,   128)),
        (slice(0,   128), slice(128, 256)),
        (slice(128, 256), slice(0,   128)),
        (slice(128, 256), slice(128, 256)),
    }


def test_exact_2d_aligned_single_dim():
    # Y (leaf=32, tier 1) handles the full 4× reduction alone - 4 strips of 64 rows.
    # X (leaf=-1, tier 2) is never reached because tier 1 already meets the budget.
    specs = _specs((256, 256), ("Y", "X"), 1 / 16, {"Y": 32})
    assert [s.slices for s in specs] == [
        (slice(0,   64),  slice(None)),
        (slice(64,  128), slice(None)),
        (slice(128, 192), slice(None)),
        (slice(192, 256), slice(None)),
    ]


def test_exact_two_constrained_dims_split_fairly():
    # Y and X both constrained (leaf=32); 4x reduction needed → each splits into 2.
    specs = _specs((512, 512), ("Y", "X"), 0.25, {"Y": 32, "X": 32})
    assert [s.slices for s in specs] == [
        (slice(0, 256),   slice(0, 256)),
        (slice(0, 256),   slice(256, 512)),
        (slice(256, 512), slice(0, 256)),
        (slice(256, 512), slice(256, 512)),
    ]


def test_exact_3d_z_and_y_both_split():
    # Z (leaf=1 default) and Y (leaf=32) are both constrained.
    # Geometric distribution: Y → chunk 128 (2 strips), Z → chunk 1 (3 slices). X full.
    specs = _specs((3, 256, 256), ("Z", "Y", "X"), 0.2, {"Y": 32})
    assert {s.slices for s in specs} == {
        (slice(0, 1), slice(0,   128), slice(None)),
        (slice(0, 1), slice(128, 256), slice(None)),
        (slice(1, 2), slice(0,   128), slice(None)),
        (slice(1, 2), slice(128, 256), slice(None)),
        (slice(2, 3), slice(0,   128), slice(None)),
        (slice(2, 3), slice(128, 256), slice(None)),
    }
