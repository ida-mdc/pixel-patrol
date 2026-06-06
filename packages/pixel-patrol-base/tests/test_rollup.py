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
    # User explicitly requested X tiling — dim_x is active, dim_y (spatial, not tiled) is not.
    # Each X tile may have multiple Y fragments (memory splits); those must be collapsed.
    chunks = [
        _result(leaf_rows=[{"dim_z": 0, "dim_x": 0,   "dim_y": 0,   "num_pixels": 100}]),
        _result(leaf_rows=[{"dim_z": 0, "dim_x": 512, "dim_y": 0,   "num_pixels": 200}]),
        _result(leaf_rows=[{"dim_z": 0, "dim_x": 0,   "dim_y": 512, "num_pixels": 300}]),
        _result(leaf_rows=[{"dim_z": 0, "dim_x": 512, "dim_y": 512, "num_pixels": 400}]),
    ]
    result = _rollup(chunks, processors=[], leaf_block_shape={"X": 512})
    assert len(_rows_at(result, 0)) == 1
    assert _rows_at(result, 0)[0]["num_pixels"] == 1000  # 100+200+300+400
    # Only dim_x is active: 2 unique X values → 2 obs_level=1 rows.
    assert len(_rows_at(result, 1)) == 2
    by_x = {r["dim_x"]: r["num_pixels"] for r in _rows_at(result, 1)}
    assert by_x[0]   == 400   # 100 (y=0) + 300 (y=512)
    assert by_x[512] == 600   # 200 (y=0) + 400 (y=512)


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


def test_global_row_has_all_active_dims_none():
    # obs_level=0 must have every active dim set to None regardless of how many active dims there are.
    leaf_rows = [
        {"dim_z": z, "dim_c": c, "num_pixels": 10}
        for z in range(2) for c in range(2)
    ]
    result = _rollup([_result(leaf_rows=leaf_rows)], processors=[])
    global_row = _rows_at(result, 0)[0]
    assert global_row.get("dim_z") is None
    assert global_row.get("dim_c") is None


# ── three active dims: full power-set coverage ────────────────────────────────

def test_three_active_dims_obs_levels_and_counts():
    # Z=2, C=2, T=2 → 8 leaf blocks.
    # Power-set over 3 dims:
    #   obs_level=1: C(3,1)=3 combos × 2 groups each = 6 rows
    #   obs_level=2: C(3,2)=3 combos × 4 groups each = 12 rows
    #   obs_level=3: 2×2×2 = 8 rows
    leaf_rows = [
        {"dim_z": z, "dim_c": c, "dim_t": t, "num_pixels": 1}
        for z in range(2) for c in range(2) for t in range(2)
    ]
    result = _rollup([_result(leaf_rows=leaf_rows)], processors=[])
    assert {r["obs_level"] for r in result} == {0, 1, 2, 3}
    assert len(_rows_at(result, 0)) == 1
    assert len(_rows_at(result, 1)) == 6
    assert len(_rows_at(result, 2)) == 12
    assert len(_rows_at(result, 3)) == 8


def test_three_active_dims_per_z_aggregation():
    # Z=2, C=2, T=2. Each Z slice has 4 leaf blocks (2C×2T).
    # Per-Z rows at obs_level=1 must aggregate over all (C, T) combos.
    leaf_rows = [
        {"dim_z": z, "dim_c": c, "dim_t": t, "num_pixels": (z + 1) * 10}
        for z in range(2) for c in range(2) for t in range(2)
    ]
    result = _rollup([_result(leaf_rows=leaf_rows)], processors=[])
    per_z = [
        r for r in _rows_at(result, 1)
        if r.get("dim_z") is not None and r.get("dim_c") is None and r.get("dim_t") is None
    ]
    assert len(per_z) == 2
    by_z = {r["dim_z"]: r["num_pixels"] for r in per_z}
    assert by_z[0] == 4 * 10   # 4 blocks × 10 each
    assert by_z[1] == 4 * 20   # 4 blocks × 20 each


def test_three_active_dims_per_zc_aggregation():
    # At obs_level=2, per-(Z,C) rows aggregate over T only.
    leaf_rows = [
        {"dim_z": z, "dim_c": c, "dim_t": t, "num_pixels": (z + 1) * (c + 1) * 10}
        for z in range(2) for c in range(2) for t in range(2)
    ]
    result = _rollup([_result(leaf_rows=leaf_rows)], processors=[])
    per_zc = [
        r for r in _rows_at(result, 2)
        if r.get("dim_z") is not None and r.get("dim_c") is not None and r.get("dim_t") is None
    ]
    assert len(per_zc) == 4   # (Z,C) ∈ {(0,0),(0,1),(1,0),(1,1)}
    by_zc = {(r["dim_z"], r["dim_c"]): r["num_pixels"] for r in per_zc}
    # Each (Z,C) combo has 2 T values; num_pixels = (z+1)*(c+1)*10 per block.
    assert by_zc[(0, 0)] == 2 * 1 * 1 * 10   # 20
    assert by_zc[(0, 1)] == 2 * 1 * 2 * 10   # 40
    assert by_zc[(1, 0)] == 2 * 2 * 1 * 10   # 40
    assert by_zc[(1, 1)] == 2 * 2 * 2 * 10   # 80


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


# ── *_size correctness per obs_level ─────────────────────────────────────────

def test_size_fields_correct_per_obs_level():
    # Z=2, Y split into 2 halves (memory management). image_meta carries full sizes.
    # obs_level=0 (global): all dims at full-image size.
    # obs_level=1 (per-Z): Z_size=1 (one slice), Y_size and X_size at full-image size.
    image_meta = {"Z_size": 2, "Y_size": 1024, "X_size": 2048}
    def _result_with_meta(leaf_rows):
        return MemoryChunkResult(
            file_index=0, child_id=None, chunk_rows={}, leaf_rows=leaf_rows,
            image_meta=image_meta,
        )
    chunks = [
        _result_with_meta([{"dim_z": 0, "dim_y": 0,    "Z_size": 1, "Y_size": 512, "X_size": 2048, "num_pixels": 512  * 2048}]),
        _result_with_meta([{"dim_z": 0, "dim_y": 512,  "Z_size": 1, "Y_size": 512, "X_size": 2048, "num_pixels": 512  * 2048}]),
        _result_with_meta([{"dim_z": 1, "dim_y": 0,    "Z_size": 1, "Y_size": 512, "X_size": 2048, "num_pixels": 512  * 2048}]),
        _result_with_meta([{"dim_z": 1, "dim_y": 512,  "Z_size": 1, "Y_size": 512, "X_size": 2048, "num_pixels": 512  * 2048}]),
    ]
    result = _rollup(chunks, processors=[])
    global_row = _rows_at(result, 0)[0]
    assert global_row["Z_size"] == 2    # full image
    assert global_row["Y_size"] == 1024
    assert global_row["X_size"] == 2048
    for row in _rows_at(result, 1):
        assert row["Z_size"] == 1       # one slice
        assert row["Y_size"] == 1024    # full extent (both Y halves aggregated)
        assert row["X_size"] == 2048
