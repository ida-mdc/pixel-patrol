"""Tests for _rollup: aggregation, obs_level assignment, active dim detection, power-set grouping."""
from __future__ import annotations

import pytest

from pixel_patrol_base.core.processing import MemoryChunkResult, _rollup
from _processing_mocks import MockLeafProcessor, MockMemoryProcessor


def _result(file_index=0, child_id=None, chunk_rows=None, leaf_rows=None):
    return MemoryChunkResult(
        file_index=file_index,
        child_id=child_id,
        chunk_rows=chunk_rows or {},
        leaf_rows=leaf_rows or [],
        image_meta={},
    )


def _leaf_proc(name="p", **cols):
    return MockLeafProcessor(name, cols if cols else {"metric": 1.0})


def _mem_proc(name="m", **cols):
    return MockMemoryProcessor(name, cols if cols else {"summary": 1.0})


def _rows_at(result, level):
    return [r for r in result if r["obs_level"] == level]


# ── no leaf rows ─────────────────────────────────────────────────────────────

def test_empty_no_leaf_rows_no_processors_returns_nothing():
    result = _rollup([_result()], processors=[])
    assert len(result) == 0


def test_empty_memory_proc_col_in_global_row():
    mem_proc = _mem_proc("m", thumbnail=b"")
    result = _rollup([_result(chunk_rows={"m": {"thumbnail": b"img"}})], processors=[mem_proc])
    assert len(result) == 1
    assert "thumbnail" in result[0]


# ── single leaf row (no active dims) ─────────────────────────────────────────

def test_single_leaf_row_no_active_dims():
    leaf_rows = [{"dim_z": 0, "num_pixels": 100, "Z_size": 1}]
    result = _rollup([_result(leaf_rows=leaf_rows)], processors=[])
    assert len(result) == 1
    assert result[0]["obs_level"] == 0
    assert result[0]["num_pixels"] == 100


# ── one active dim ────────────────────────────────────────────────────────────

def test_one_active_dim_obs_levels():
    leaf_rows = [{"dim_z": i, "num_pixels": 10} for i in range(3)]
    result = _rollup([_result(leaf_rows=leaf_rows)], processors=[])
    assert {r["obs_level"] for r in result} == {0, 1}


def test_one_active_dim_global_row_count():
    leaf_rows = [{"dim_z": i, "num_pixels": 10} for i in range(5)]
    result = _rollup([_result(leaf_rows=leaf_rows)], processors=[])
    assert len(_rows_at(result, 0)) == 1


def test_one_active_dim_individual_count():
    leaf_rows = [{"dim_z": i, "num_pixels": 10} for i in range(4)]
    result = _rollup([_result(leaf_rows=leaf_rows)], processors=[])
    assert len(_rows_at(result, 1)) == 4


def test_one_active_dim_global_num_pixels():
    leaf_rows = [{"dim_z": 0, "num_pixels": 100}, {"dim_z": 1, "num_pixels": 200}]
    result = _rollup([_result(leaf_rows=leaf_rows)], processors=[])
    assert _rows_at(result, 0)[0]["num_pixels"] == 300


def test_one_active_dim_global_row_dim_is_none():
    leaf_rows = [{"dim_z": 0, "num_pixels": 10}, {"dim_z": 1, "num_pixels": 10}]
    result = _rollup([_result(leaf_rows=leaf_rows)], processors=[])
    assert _rows_at(result, 0)[0].get("dim_z") is None


def test_one_active_dim_individual_rows_carry_coords():
    leaf_rows = [{"dim_z": 2, "num_pixels": 10}, {"dim_z": 5, "num_pixels": 10}]
    result = _rollup([_result(leaf_rows=leaf_rows)], processors=[])
    z_vals = sorted(r["dim_z"] for r in _rows_at(result, 1))
    assert z_vals == [2, 5]


# ── two active dims ───────────────────────────────────────────────────────────

def test_two_active_dims_row_counts():
    leaf_rows = [{"dim_z": z, "dim_c": c, "num_pixels": 10} for z in range(2) for c in range(2)]
    result = _rollup([_result(leaf_rows=leaf_rows)], processors=[])
    assert len(_rows_at(result, 0)) == 1
    assert len(_rows_at(result, 1)) == 4
    assert len(_rows_at(result, 2)) == 4


def test_two_active_dims_per_z_aggregation():
    leaf_rows = [
        {"dim_z": 0, "dim_c": 0, "num_pixels": 10},
        {"dim_z": 0, "dim_c": 1, "num_pixels": 20},
        {"dim_z": 1, "dim_c": 0, "num_pixels": 30},
        {"dim_z": 1, "dim_c": 1, "num_pixels": 40},
    ]
    result = _rollup([_result(leaf_rows=leaf_rows)], processors=[])
    per_z = [r for r in _rows_at(result, 1) if r.get("dim_z") is not None and r.get("dim_c") is None]
    assert len(per_z) == 2
    assert sorted(r["num_pixels"] for r in per_z) == [30, 70]


def test_two_active_dims_per_c_aggregation():
    leaf_rows = [
        {"dim_z": 0, "dim_c": 0, "num_pixels": 10},
        {"dim_z": 0, "dim_c": 1, "num_pixels": 20},
        {"dim_z": 1, "dim_c": 0, "num_pixels": 30},
        {"dim_z": 1, "dim_c": 1, "num_pixels": 40},
    ]
    result = _rollup([_result(leaf_rows=leaf_rows)], processors=[])
    per_c = [r for r in _rows_at(result, 1) if r.get("dim_c") is not None and r.get("dim_z") is None]
    assert len(per_c) == 2
    assert sorted(r["num_pixels"] for r in per_c) == [40, 60]


# ── degenerate dims ───────────────────────────────────────────────────────────

def test_degenerate_dim_not_active():
    leaf_rows = [{"dim_z": 0, "dim_c": 0, "num_pixels": 10}, {"dim_z": 1, "dim_c": 0, "num_pixels": 20}]
    result = _rollup([_result(leaf_rows=leaf_rows)], processors=[])
    assert {r["obs_level"] for r in result} == {0, 1}
    assert len(_rows_at(result, 1)) == 2


# ── leaf metric aggregation ───────────────────────────────────────────────────

def test_leaf_metric_in_global_row():
    leaf_rows = [{"dim_z": 0, "num_pixels": 1, "intensity": 5.0}, {"dim_z": 1, "num_pixels": 1, "intensity": 9.0}]
    proc = _leaf_proc("p", intensity=0.0)
    result = _rollup([_result(leaf_rows=leaf_rows)], processors=[proc])
    assert "intensity" in _rows_at(result, 0)[0]


def test_leaf_metric_in_individual_rows():
    leaf_rows = [{"dim_z": 0, "num_pixels": 1, "val": 3.0}, {"dim_z": 1, "num_pixels": 1, "val": 7.0}]
    proc = _leaf_proc("p", val=0.0)
    result = _rollup([_result(leaf_rows=leaf_rows)], processors=[proc])
    ind = sorted(_rows_at(result, 1), key=lambda r: r["dim_z"])
    assert ind[0]["val"] == 3.0
    assert ind[1]["val"] == 7.0


# ── memory processor columns ──────────────────────────────────────────────────

def test_memory_proc_col_only_in_global_row():
    leaf_rows = [{"dim_z": 0, "num_pixels": 10}, {"dim_z": 1, "num_pixels": 10}]
    mem_proc = _mem_proc("m", summary=0.0)
    result = _rollup([_result(chunk_rows={"m": {"summary": 42.0}}, leaf_rows=leaf_rows)], processors=[mem_proc])
    assert "summary" in _rows_at(result, 0)[0]
    for row in _rows_at(result, 1):
        assert "summary" not in row


def test_memory_proc_multiple_spatial_chunks():
    chunk1 = _result(chunk_rows={"m": {"patch": b"a"}}, leaf_rows=[{"dim_z": 0, "num_pixels": 10}])
    chunk2 = _result(chunk_rows={"m": {"patch": b"b"}}, leaf_rows=[{"dim_z": 1, "num_pixels": 10}])
    mem_proc = _mem_proc("m", patch=b"")
    result = _rollup([chunk1, chunk2], processors=[mem_proc])
    assert _rows_at(result, 0)[0]["patch"] == b"a"


def test_memory_proc_missing_from_some_chunks_still_aggregates():
    chunk1 = _result(chunk_rows={"m": {"v": 1.0}}, leaf_rows=[{"dim_z": 0, "num_pixels": 10}])
    chunk2 = _result(chunk_rows={}, leaf_rows=[{"dim_z": 1, "num_pixels": 10}])
    mem_proc = _mem_proc("m", v=0.0)
    result = _rollup([chunk1, chunk2], processors=[mem_proc])
    assert _rows_at(result, 0)[0]["v"] == 1.0


# ── multiple spatial chunks pooled ───────────────────────────────────────────

def test_spatial_chunks_leaf_rows_pooled():
    chunk1 = _result(leaf_rows=[{"dim_z": 0, "num_pixels": 100}])
    chunk2 = _result(leaf_rows=[{"dim_z": 1, "num_pixels": 200}])
    result = _rollup([chunk1, chunk2], processors=[])
    assert _rows_at(result, 0)[0]["num_pixels"] == 300
    assert len(_rows_at(result, 1)) == 2


# ── X/Y full-extent-by-default: memory-chunk offsets are not user tiling ─────

def test_xy_varying_from_memory_chunking_not_active_without_slice_size():
    # Simulate two memory chunks at different X positions (memory management split,
    # not user-requested tiling). Without leaf_block_shape, X/Y should be ignored.
    chunk1 = _result(leaf_rows=[{"dim_z": 0, "dim_x": 0,    "dim_y": 0, "num_pixels": 100}])
    chunk2 = _result(leaf_rows=[{"dim_z": 0, "dim_x": 2048, "dim_y": 0, "num_pixels": 100}])
    result = _rollup([chunk1, chunk2], processors=[])
    assert {r["obs_level"] for r in result} == {0}
    assert _rows_at(result, 0)[0]["num_pixels"] == 200


def test_xy_active_when_explicitly_in_leaf_block_shape():
    # User explicitly requested X/Y tiling — variation should create active dims.
    chunk1 = _result(leaf_rows=[{"dim_z": 0, "dim_x": 0,    "dim_y": 0, "num_pixels": 100}])
    chunk2 = _result(leaf_rows=[{"dim_z": 0, "dim_x": 2048, "dim_y": 0, "num_pixels": 100}])
    result = _rollup([chunk1, chunk2], processors=[], leaf_block_shape={"X": 2048, "Y": -1})
    assert 1 in {r["obs_level"] for r in result}


# ── spatial Y/X split with active non-spatial dims (the core bug) ─────────────

def test_spatial_y_split_active_z_produces_one_row_per_z():
    # Z=2, Y split into 2 halves for memory management → 4 leaf blocks.
    # Y is excluded from active_dims (not user-tiled).
    # obs_level=1 must have exactly 2 rows (one per Z), not 4 raw leaf blocks.
    chunks = [
        _result(leaf_rows=[{"dim_z": 0, "dim_y": 0,    "num_pixels": 100}]),
        _result(leaf_rows=[{"dim_z": 0, "dim_y": 1024, "num_pixels": 150}]),
        _result(leaf_rows=[{"dim_z": 1, "dim_y": 0,    "num_pixels": 200}]),
        _result(leaf_rows=[{"dim_z": 1, "dim_y": 1024, "num_pixels": 250}]),
    ]
    result = _rollup(chunks, processors=[])
    assert {r["obs_level"] for r in result} == {0, 1}
    assert len(_rows_at(result, 1)) == 2


def test_spatial_y_split_active_z_aggregates_num_pixels():
    # Same scenario: num_pixels for each Z row must sum both Y halves.
    chunks = [
        _result(leaf_rows=[{"dim_z": 0, "dim_y": 0,    "num_pixels": 100}]),
        _result(leaf_rows=[{"dim_z": 0, "dim_y": 1024, "num_pixels": 150}]),
        _result(leaf_rows=[{"dim_z": 1, "dim_y": 0,    "num_pixels": 200}]),
        _result(leaf_rows=[{"dim_z": 1, "dim_y": 1024, "num_pixels": 250}]),
    ]
    result = _rollup(chunks, processors=[])
    by_z = {r["dim_z"]: r["num_pixels"] for r in _rows_at(result, 1)}
    assert by_z[0] == 250   # 100 + 150
    assert by_z[1] == 450   # 200 + 250


def test_spatial_xy_split_active_z_c_produces_one_row_per_zc_combo():
    # Z=2, C=2, Y and X each split in half → 2*2*2*2 = 16 leaf blocks.
    # Only Z and C are active; each (Z, C) pair must produce exactly one obs_level=2 row.
    chunks = []
    pixels = {}
    for z in range(2):
        for c in range(2):
            for y in (0, 512):
                for x in (0, 512):
                    px = (z + 1) * (c + 1) * 10
                    chunks.append(_result(leaf_rows=[{
                        "dim_z": z, "dim_c": c, "dim_y": y, "dim_x": x, "num_pixels": px,
                    }]))
                    pixels[(z, c)] = pixels.get((z, c), 0) + px
    result = _rollup(chunks, processors=[])
    level2 = _rows_at(result, 2)
    assert len(level2) == 4   # one per (Z, C) combo, not 16
    by_zc = {(r["dim_z"], r["dim_c"]): r["num_pixels"] for r in level2}
    for (z, c), expected in pixels.items():
        assert by_zc[(z, c)] == expected


def test_spatial_split_leaf_metric_aggregated_not_first_row():
    # Z=2 with Y split into 2 halves per Z slice → 4 leaf blocks.
    # get_aggregation must be called with both Y-fragment rows for each Z,
    # not just the first fragment emitted raw. The key observable: 2 rows
    # at obs_level=1 (one per Z), each carrying a val produced by aggregation.
    proc = _leaf_proc("p", val=0.0)
    chunks = [
        _result(leaf_rows=[{"dim_z": 0, "dim_y": 0,    "num_pixels": 1, "val": 10.0}]),
        _result(leaf_rows=[{"dim_z": 0, "dim_y": 1024, "num_pixels": 1, "val": 20.0}]),
        _result(leaf_rows=[{"dim_z": 1, "dim_y": 0,    "num_pixels": 1, "val": 30.0}]),
        _result(leaf_rows=[{"dim_z": 1, "dim_y": 1024, "num_pixels": 1, "val": 40.0}]),
    ]
    result = _rollup(chunks, processors=[proc])
    level1 = _rows_at(result, 1)
    assert len(level1) == 2
    assert {r["dim_z"] for r in level1} == {0, 1}
