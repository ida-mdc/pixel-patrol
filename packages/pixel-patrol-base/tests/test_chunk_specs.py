"""Unit tests for memory-chunking: _resolve_leaf_block_shape, _compute_memory_chunk_specs,
and _execute_container_task's use of them to chunk an oversized container sub-image."""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, List, Tuple

import numpy as np

from pixel_patrol_base.core.processing import (
    BatchTask,
    ContainerTask,
    MemoryChunkResult,
    _IndexedPath,
    _compute_memory_chunk_specs,
    _execute_batch_task,
    _execute_container_task,
    _resolve_leaf_block_shape,
)
from pixel_patrol_base.core.processing_config import ProcessingConfig
from pixel_patrol_base.core.record import Record, record_from
from _processing_mocks import MockMemoryProcessor

_DUMMY_PATH = Path("/mock/file")


def _specs(shape, dim_order, mb_per_task, leaf_block_shape=None, dtype=np.float32):
    return _compute_memory_chunk_specs(_DUMMY_PATH, dim_order, shape, dtype, mb_per_task, leaf_block_shape)


# ── _resolve_leaf_block_shape ─────────────────────────────────────────────────

def test_resolve_defaults():
    block = _resolve_leaf_block_shape("ZCTSYX", None)
    for dim in ("Z", "C", "T", "S"):
        assert block[dim] == 1
    for dim in ("Y", "X"):
        assert block[dim] == -1


def test_resolve_user_spec_overrides():
    block = _resolve_leaf_block_shape("ZYX", {"Z": -1, "X": 16})
    assert block["Z"] == -1
    assert block["X"] == 16
    assert block["Y"] == -1


def test_resolve_unknown_user_spec_key_is_ignored():
    block = _resolve_leaf_block_shape("YX", {"Q": 5})
    assert "Q" not in block
    assert block["Y"] == -1


# ── _compute_memory_chunk_specs ───────────────────────────────────────────────

def test_file_fits_in_budget_returns_none():
    assert _specs((64, 64), "YX", 1.0) is None


def test_2d_yx_no_slice_size_splits_by_budget():
    # Geometric distribution: both Y and X share the reduction equally → 2×2 chunks.
    specs = _specs((256, 256), "YX", 0.1)
    assert specs is not None
    assert len(specs) == 4
    assert all(s.slices[0] != slice(None) for s in specs)  # Y split
    assert all(s.slices[1] != slice(None) for s in specs)  # X split


def test_3d_zyx_no_slice_size_all_dims_split():
    # Z (leaf=1, processed first), then Y and X (leaf=-1) all share the reduction.
    specs = _specs((2, 256, 256), "ZYX", 0.1)
    assert specs is not None
    assert len(specs) > 1
    assert all(s.slices[0] != slice(None) for s in specs)  # Z split
    assert all(s.slices[1] != slice(None) for s in specs)  # Y split
    assert all(s.slices[2] != slice(None) for s in specs)  # X split


def test_must_split_two_dims_when_one_is_not_enough():
    # Very tight budget: Z splits first (leaf=1), then Y and X (leaf=-1).
    specs = _specs((2, 512, 512), "ZYX", 0.001)
    assert specs is not None
    assert any(s.slices[0] != slice(None) for s in specs)  # Z split
    assert any(s.slices[1] != slice(None) for s in specs)  # Y split
    assert any(s.slices[2] != slice(None) for s in specs)  # X split


def test_non_divisible_large_image_with_slice_size():
    # Originally failing: non-divisible dims with slice_size caused OOM
    specs = _specs((40, 53638, 62366), "ZYX", 512.0, {"X": 1024, "Y": 1024},
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
    specs = _specs((4, 192, 300), "ZYX", budget_mb, {"Y": 32})
    assert specs is not None
    for s in specs:
        elems = 1
        for size, slc in zip((4, 192, 300), s.slices):
            elems *= size if slc == slice(None) else slc.stop - slc.start
        assert elems * 4 <= budget_bytes


def test_cartesian_product_no_duplicate_combos():
    specs = _specs((6, 64, 64), "ZYX", 0.001)
    assert specs is not None
    assert len({s.slices for s in specs}) == len(specs)


def test_chunk_spec_metadata():
    shape = (2, 256, 128)
    dim_order = "ZYX"
    specs = _specs(shape, dim_order, 0.05, {"Y": 32, "Z": 1})
    assert specs is not None
    for s in specs:
        assert s.dim_order == dim_order


def test_1d_array_splits():
    specs = _specs((1000,), "Z", 1 / 1024, {"Z": 100})
    assert specs is not None
    assert sum(s.slices[0].stop - s.slices[0].start for s in specs) == 1000


# ── exact slice values ────────────────────────────────────────────────────────
# Each test pins the expected slice boundaries to catch regressions.

def test_exact_2d_no_slice_size():
    # Y and X share the 2.5× reduction equally → 2 strips each, 4 chunks of 128×128.
    specs = _specs((256, 256), "YX", 0.1)
    assert {s.slices for s in specs} == {
        (slice(0,   128), slice(0,   128)),
        (slice(0,   128), slice(128, 256)),
        (slice(128, 256), slice(0,   128)),
        (slice(128, 256), slice(128, 256)),
    }


def test_exact_2d_aligned_single_dim():
    # Y (leaf=32, tier 1) handles the full 4× reduction alone - 4 strips of 64 rows.
    # X (leaf=-1, tier 2) is never reached because tier 1 already meets the budget.
    specs = _specs((256, 256), "YX", 1 / 16, {"Y": 32})
    assert [s.slices for s in specs] == [
        (slice(0,   64),  slice(None)),
        (slice(64,  128), slice(None)),
        (slice(128, 192), slice(None)),
        (slice(192, 256), slice(None)),
    ]


def test_exact_two_constrained_dims_split_fairly():
    # Y and X both constrained (leaf=32); 4x reduction needed → each splits into 2.
    specs = _specs((512, 512), "YX", 0.25, {"Y": 32, "X": 32})
    assert [s.slices for s in specs] == [
        (slice(0, 256),   slice(0, 256)),
        (slice(0, 256),   slice(256, 512)),
        (slice(256, 512), slice(0, 256)),
        (slice(256, 512), slice(256, 512)),
    ]


def test_exact_3d_z_and_y_both_split():
    # Z (leaf=1 default) and Y (leaf=32) are both constrained.
    # Geometric distribution: Y → chunk 128 (2 strips), Z → chunk 1 (3 slices). X full.
    specs = _specs((3, 256, 256), "ZYX", 0.2, {"Y": 32})
    assert {s.slices for s in specs} == {
        (slice(0, 1), slice(0,   128), slice(None)),
        (slice(0, 1), slice(128, 256), slice(None)),
        (slice(1, 2), slice(0,   128), slice(None)),
        (slice(1, 2), slice(128, 256), slice(None)),
        (slice(2, 3), slice(0,   128), slice(None)),
        (slice(2, 3), slice(128, 256), slice(None)),
    }


# ── _execute_container_task: chunking an oversized container sub-image ────────

class _SumProcessor(MockMemoryProcessor):
    """Sums real chunk pixel values, so splitting/reassembly can be checked for correctness."""

    def __init__(self):
        super().__init__("sum", {"pixel_sum": 0})

    def run_chunk(self, record: Record) -> dict:
        return {"pixel_sum": int(np.asarray(record.data).sum())}

    def get_aggregation(self, name: str):
        return lambda rows, g_dims: sum(r["pixel_sum"] for r in rows)


class _FakeContainerLoader:
    """Minimal loader yielding pre-built sub-image arrays for a ContainerTask."""

    def __init__(self, arrays: List[np.ndarray], dim_order: str = "ZYX") -> None:
        self._arrays = arrays
        self._dim_order = dim_order

    def load_range(self, file_path: Path, start: int, stop: int) -> Iterator[Tuple[str, Record]]:
        for i in range(start, stop):
            yield str(i), record_from(self._arrays[i], {"dim_order": self._dim_order}, kind="intensity")


def _sums(chunk_group: List[MemoryChunkResult]) -> int:
    return sum(cr.chunk_rows["sum"]["pixel_sum"] for cr in chunk_group)


def test_small_sub_image_stays_a_single_chunk():
    small = np.arange(2 * 4 * 4, dtype=np.uint8).reshape(2, 4, 4)
    loader = _FakeContainerLoader([small])
    config = ProcessingConfig(mb_per_task=100 / (1024 * 1024))  # budget = 100 bytes
    task = ContainerTask(file_index=0, file_path="mock.lmdb", image_slice=(0, 1))

    results = _execute_container_task(task, loader, [_SumProcessor()], config)

    assert len(results) == 1
    assert len(results[0]) == 1
    assert _sums(results[0]) == int(small.sum())


def test_oversized_sub_image_is_chunked_and_reassembles_correctly():
    big = np.arange(20 * 4 * 4, dtype=np.uint8).reshape(20, 4, 4)
    loader = _FakeContainerLoader([big])
    config = ProcessingConfig(mb_per_task=100 / (1024 * 1024))  # budget = 100 bytes; big = 320 bytes
    task = ContainerTask(file_index=0, file_path="mock.lmdb", image_slice=(0, 1))

    results = _execute_container_task(task, loader, [_SumProcessor()], config)

    assert len(results) == 1
    assert len(results[0]) > 1  # actually split into multiple memory chunks
    assert _sums(results[0]) == int(big.sum())  # no pixels lost or double-counted


def test_mixed_batch_only_chunks_the_oversized_sub_image():
    small = np.arange(2 * 4 * 4, dtype=np.uint8).reshape(2, 4, 4)
    big   = np.arange(20 * 4 * 4, dtype=np.uint8).reshape(20, 4, 4)
    loader = _FakeContainerLoader([small, big])
    config = ProcessingConfig(mb_per_task=100 / (1024 * 1024))
    task = ContainerTask(file_index=0, file_path="mock.lmdb", image_slice=(0, 2))

    results = _execute_container_task(task, loader, [_SumProcessor()], config)

    assert len(results) == 2
    assert len(results[0]) == 1
    assert len(results[1]) > 1
    assert _sums(results[0]) == int(small.sum())
    assert _sums(results[1]) == int(big.sum())


class _PoisonArray:
    """Looks like a lazy dask array (has .compute) but raises when materialized."""
    shape = (4, 4)
    dtype = np.dtype("uint8")

    def __getitem__(self, key):
        return self

    def compute(self, **kw):
        raise RuntimeError("simulated decode failure")


class _OneBadSubImageLoader:
    """Yields good sub-images except index 1, which fails to decode."""

    def load_range(self, file_path: Path, start: int, stop: int) -> Iterator[Tuple[str, Record]]:
        for i in range(start, stop):
            if i == 1:
                yield str(i), Record(data=_PoisonArray(), dim_order="YX", dim_names=["Y", "X"],
                                     kind="intensity", meta={"dim_order": "YX"}, capabilities=set())
            else:
                arr = np.full((4, 4), i, dtype=np.uint8)
                yield str(i), record_from(arr, {"dim_order": "YX"}, kind="intensity")


def test_one_bad_sub_image_does_not_discard_its_siblings():
    task = ContainerTask(file_index=0, file_path="mock.lmdb", image_slice=(0, 3))
    config = ProcessingConfig(mb_per_task=100)

    results = _execute_container_task(task, _OneBadSubImageLoader(), [_SumProcessor()], config)

    assert len(results) == 2  # sub-images 0 and 2 survive; only 1 is lost
    assert _sums(results[0]) == 0        # sub-image 0: filled with 0
    assert _sums(results[1]) == 4 * 4 * 2  # sub-image 2: filled with 2


class _OneBadFileLoader:
    """Loads good files except 'bad.tif', which fails to decode."""

    def load(self, file_path: Path) -> Record:
        if "bad" in str(file_path):
            return Record(data=_PoisonArray(), dim_order="YX", dim_names=["Y", "X"],
                         kind="intensity", meta={"dim_order": "YX"}, capabilities=set())
        arr = np.full((4, 4), 7, dtype=np.uint8)
        return record_from(arr, {"dim_order": "YX"}, kind="intensity")


def test_one_bad_file_does_not_discard_its_batch_siblings():
    task = BatchTask(files=(
        _IndexedPath(file_index=0, file_path="a.tif"),
        _IndexedPath(file_index=1, file_path="bad.tif"),
        _IndexedPath(file_index=2, file_path="c.tif"),
    ))
    config = ProcessingConfig(mb_per_task=100)

    results = _execute_batch_task(task, _OneBadFileLoader(), [_SumProcessor()], config)

    assert len(results) == 2  # a.tif and c.tif survive; only bad.tif is lost
    assert {r.file_index for r in results} == {0, 2}
