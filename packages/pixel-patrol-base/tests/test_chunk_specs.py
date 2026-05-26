"""Unit tests for _resolve_leaf_block_shape and _compute_memory_chunk_specs."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pytest

from pixel_patrol_base.core.processing import (
    MemoryChunkSpec,
    _compute_memory_chunk_specs,
    _resolve_leaf_block_shape,
)
from _processing_mocks import MockEntry, MockLoader, capture_warnings


def _make(shape, dim_order, dtype=np.float32):
    path = Path("/mock/file.npy")
    loader = MockLoader({str(path): MockEntry(shape=shape, dtype=dtype, dim_order=dim_order)})
    return path, loader


def _specs(path, loader, mb_per_task, leaf_block_shape=None):
    return _compute_memory_chunk_specs(path, loader.read_header(path), mb_per_task, leaf_block_shape)


def _dim_slices(specs: List[MemoryChunkSpec], dim: str) -> List[Any]:
    return [s.slices[list(s.dim_order).index(dim)] for s in specs]


def _is_full(slc: Any) -> bool:
    return slc == slice(None)


def _check_multiples(specs: List[MemoryChunkSpec], block: Dict[str, int]) -> None:
    for s in specs:
        for dim, slc in zip(s.dim_order, s.slices):
            if _is_full(slc):
                continue
            blk = block.get(dim)
            if blk is None or blk == -1:
                continue
            assert slc.start % blk == 0, f"dim {dim}: start {slc.start} not a multiple of {blk}"


def _check_origins(specs: List[MemoryChunkSpec]) -> None:
    for s in specs:
        for i, (dim, slc) in enumerate(zip(s.dim_order, s.slices)):
            expected = 0 if _is_full(slc) else slc.start
            assert s.origin[i] == expected, f"origin mismatch for dim {dim}"


# ── _resolve_leaf_block_shape ─────────────────────────────────────────────────

def test_resolve_xy_default_to_minus1():
    block = _resolve_leaf_block_shape(("Y", "X"), None)
    assert block["Y"] == -1
    assert block["X"] == -1


def test_resolve_non_xy_default_to_1():
    block = _resolve_leaf_block_shape(("Z", "C", "T", "S", "Y", "X"), None)
    for dim in ("Z", "C", "T", "S"):
        assert block[dim] == 1
    for dim in ("Y", "X"):
        assert block[dim] == -1


def test_resolve_user_spec_overrides():
    block = _resolve_leaf_block_shape(("Z", "Y", "X"), {"Z": 4, "X": 16})
    assert block["Z"] == 4
    assert block["X"] == 16
    assert block["Y"] == -1


def test_resolve_pin_non_xy_to_minus1():
    block = _resolve_leaf_block_shape(("Z", "Y", "X"), {"Z": -1})
    assert block["Z"] == -1


def test_resolve_unknown_user_spec_key_is_ignored():
    block = _resolve_leaf_block_shape(("Y", "X"), {"Q": 5})
    assert "Q" not in block
    assert block["Y"] == -1


def test_resolve_all_dims_explicitly_set():
    block = _resolve_leaf_block_shape(("Z", "Y", "X"), {"Z": 2, "Y": 64, "X": 32})
    assert block == {"Z": 2, "Y": 64, "X": 32}


# ── _compute_memory_chunk_specs ───────────────────────────────────────────────

def test_file_fits_in_budget_returns_none():
    path, loader = _make((64, 64), ("Y", "X"))
    assert _specs(path, loader, 1.0) is None


def test_all_dims_pinned_returns_none():
    path, loader = _make((512, 512), ("Y", "X"))
    assert _specs(path, loader, 0.1, {"Y": -1, "X": -1}) is None


def test_2d_yx_splits_y():
    path, loader = _make((256, 256), ("Y", "X"))
    specs = _specs(path, loader, 0.1, {"Y": 32})
    assert specs is not None
    assert len(specs) == 3
    for slc in _dim_slices(specs, "X"):
        assert _is_full(slc)
    _check_multiples(specs, {"Y": 32})
    assert sorted(slc.start for slc in _dim_slices(specs, "Y")) == [0, 96, 192]
    assert [slc.stop - slc.start for slc in _dim_slices(specs, "Y")] == [96, 96, 64]
    _check_origins(specs)


def test_non_divisible_dim_skips_file_with_warning():
    path, loader = _make((100, 512), ("Y", "X"))
    with capture_warnings() as warnings:
        specs = _specs(path, loader, 0.1, {"Y": 32})
    assert specs is None
    assert any("could not be split" in w for w in warnings)


def test_3d_zyx_y_split_alone_fits_z_stays_full():
    path, loader = _make((2, 128, 128), ("Z", "Y", "X"))
    specs = _specs(path, loader, 0.1, {"Y": 32})
    assert specs is not None
    for slc in _dim_slices(specs, "Z"):
        assert _is_full(slc)
    assert len(specs) == 2


def test_3d_zyx_y_strip_exceeds_budget_splits_both():
    path, loader = _make((4, 256, 256), ("Z", "Y", "X"))
    specs = _specs(path, loader, 0.1, {"Y": 32, "Z": 1})
    assert specs is not None
    assert len(specs) == 16
    assert len({s.slices[0].start for s in specs if not _is_full(s.slices[0])}) > 1
    assert len({s.slices[1].start for s in specs if not _is_full(s.slices[1])}) > 1
    _check_multiples(specs, {"Y": 32, "Z": 1})
    _check_origins(specs)
    combos = {(s.slices[0].start, s.slices[1].start) for s in specs}
    assert len(combos) == 16


def test_4d_zcyx_y_split_alone_fits_c_and_z_stay_full():
    path, loader = _make((2, 3, 96, 96), ("Z", "C", "Y", "X"))
    specs = _specs(path, loader, 0.1, {"Y": 32})
    assert specs is not None
    assert len(specs) == 3
    for slc in _dim_slices(specs, "Z"):
        assert _is_full(slc)
    for slc in _dim_slices(specs, "C"):
        assert _is_full(slc)


def test_z_only_splittable_xy_pinned():
    path, loader = _make((10, 128, 128), ("Z", "Y", "X"))
    specs = _specs(path, loader, 0.1, {"Z": 1, "Y": -1, "X": -1})
    assert specs is not None
    assert len(specs) == 10
    for slc in _dim_slices(specs, "Y"):
        assert _is_full(slc)
    for slc in _dim_slices(specs, "X"):
        assert _is_full(slc)


def test_single_block_exceeds_budget_falls_back_to_one_block():
    path, loader = _make((64, 64), ("Y", "X"))
    specs = _specs(path, loader, 1 / 1024, {"Y": 32})
    assert specs is not None
    y_sizes = [slc.stop - slc.start for slc in _dim_slices(specs, "Y")]
    assert sum(y_sizes) == 64


def test_block_size_larger_than_dim_returns_none_with_warning():
    path, loader = _make((2, 256, 256), ("Z", "Y", "X"))
    with capture_warnings() as warnings:
        specs = _specs(path, loader, 0.1, {"Z": 100, "Y": -1, "X": -1})
    assert specs is None
    assert any("could not be split" in w for w in warnings)


def test_non_divisible_dim_in_z_skips_file_with_warning():
    path, loader = _make((5, 128, 128), ("Z", "Y", "X"))
    with capture_warnings() as warnings:
        specs = _specs(path, loader, 0.1, {"Z": 2})
    assert specs is None
    assert any("could not be split" in w for w in warnings)


def test_non_divisible_dim_not_needed_no_skip_no_warning():
    path, loader = _make((4, 5, 64, 64), ("Z", "C", "Y", "X"))
    with capture_warnings() as warnings:
        specs = _specs(path, loader, 0.1, {"Z": 1, "C": 2, "Y": -1, "X": -1})
    assert specs is not None
    assert len(specs) == 4
    for slc in _dim_slices(specs, "C"):
        assert _is_full(slc)
    assert not any("could not be split" in w for w in warnings)


def test_largest_splittable_non_divisible_smaller_divisible_sufficient():
    path, loader = _make((9, 8, 32, 32), ("Z", "T", "Y", "X"))
    with capture_warnings() as warnings:
        specs = _specs(path, loader, 0.1, {"Z": 4, "T": 2, "Y": -1, "X": -1})
    assert specs is not None
    assert len(specs) == 4
    for slc in _dim_slices(specs, "Z"):
        assert _is_full(slc)
    assert not any("could not be split" in w for w in warnings)


def test_all_splittable_dims_non_divisible_warns():
    path, loader = _make((3, 5, 64, 64), ("Z", "C", "Y", "X"))
    with capture_warnings() as warnings:
        specs = _specs(path, loader, 0.001, {"Z": 2, "C": 3, "Y": -1, "X": -1})
    assert specs is None
    nondiv = [w for w in warnings if "could not be split" in w]
    assert len(nondiv) == 1


def test_non_divisible_dim_warned_only_when_it_blocks_budget():
    path, loader = _make((6, 5, 32, 32), ("Z", "C", "Y", "X"))
    with capture_warnings() as warnings:
        specs = _specs(path, loader, 10 / 1024, {"Z": 2, "C": 3, "Y": -1, "X": -1})
    assert specs is not None
    assert len(specs) == 3
    for slc in _dim_slices(specs, "C"):
        assert _is_full(slc)
    assert not any("could not be split" in w for w in warnings)


def test_multi_dim_block_multiples_always_hold():
    path, loader = _make((4, 192, 300), ("Z", "Y", "X"))
    specs = _specs(path, loader, 0.05, {"Z": 2, "Y": 64})
    assert specs is not None
    _check_multiples(specs, {"Z": 2, "Y": 64})


def test_cartesian_product_no_gaps_no_overlaps():
    path, loader = _make((6, 64, 64), ("Z", "Y", "X"))
    specs = _specs(path, loader, 0.001, {"Z": 2, "Y": 16})
    assert specs is not None
    combos = {
        (s.slices[0] if _is_full(s.slices[0]) else s.slices[0].start,
         s.slices[1] if _is_full(s.slices[1]) else s.slices[1].start)
        for s in specs
    }
    assert len(combos) == len(specs)


def test_chunk_spec_metadata_origin_shape_dimorder():
    shape = (2, 256, 128)
    dim_order = ("Z", "Y", "X")
    path, loader = _make(shape, dim_order)
    specs = _specs(path, loader, 0.05, {"Y": 64, "Z": 1})
    assert specs is not None
    assert len(specs) == 8
    for s in specs:
        assert s.image_shape == shape
        assert s.dim_order == dim_order
    _check_origins(specs)
    target = [s for s in specs if s.slices[0] == slice(1, 2) and s.slices[1] == slice(128, 192)]
    assert len(target) == 1
    assert target[0].origin == (1, 128, 0)


def test_1d_z_only_file():
    path, loader = _make((1000,), ("Z",))
    specs = _specs(path, loader, 1 / 1024, {"Z": 100})
    assert specs is not None
    z_coverage = sum(
        1000 if _is_full(slc) else slc.stop - slc.start
        for slc in _dim_slices(specs, "Z")
    )
    assert z_coverage == 1000
