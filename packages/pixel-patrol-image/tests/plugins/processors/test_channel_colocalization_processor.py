"""Tests for ChannelColocalizationProcessor."""

from itertools import combinations

import dask.array as da
import numpy as np
import pytest

from pixel_patrol_base.core.record import record_from
from pixel_patrol_image.plugins.processors.channel_colocalization_processor import (
    ChannelColocalizationProcessor,
)

_ARRAY_METRICS = ("coloc_pearson_r", "coloc_ssim", "coloc_ssim_luminance",
                  "coloc_ssim_contrast", "coloc_ssim_structure")


@pytest.fixture
def proc():
    return ChannelColocalizationProcessor()


def _global(result):
    return next(r for r in result if r["obs_level"] == 0)


def _at_depth(result, depth):
    return [r for r in result if r["obs_level"] == depth]


def _n_pairs(n_c):
    return n_c * (n_c - 1) // 2


# ---------------------------------------------------------------------------
# Basic structure
# ---------------------------------------------------------------------------

def test_returns_list(proc):
    data = da.from_array(np.ones((2, 8, 8), dtype=np.float32), chunks=(2, 8, 8))
    result = proc.run(record_from(data, {"dim_order": "CYX"}))
    assert isinstance(result, list)


def test_single_channel_returns_empty_list(proc):
    data = da.from_array(np.ones((1, 8, 8), dtype=np.float32), chunks=(1, 8, 8))
    assert proc.run(record_from(data, {"dim_order": "CYX"})) == []


def test_no_c_axis_returns_empty_list(proc):
    data = da.from_array(np.ones((8, 8), dtype=np.float32), chunks=(8, 8))
    assert proc.run(record_from(data, {"dim_order": "YX"})) == []


def test_global_row_obs_level_zero(proc):
    data = da.from_array(np.ones((2, 16, 16), dtype=np.float32), chunks=(2, 16, 16))
    result = proc.run(record_from(data, {"dim_order": "CYX"}))
    assert _global(result)["obs_level"] == 0


def test_two_channels_one_pair(proc):
    data = da.from_array(np.ones((2, 16, 16), dtype=np.float32), chunks=(2, 16, 16))
    g = _global(proc.run(record_from(data, {"dim_order": "CYX"})))
    assert g["coloc_n_channels"] == 2
    for m in _ARRAY_METRICS:
        assert len(g[m]) == _n_pairs(2)  # == 1


def test_three_channels_three_pairs(proc):
    data = da.from_array(np.ones((3, 16, 16), dtype=np.float32), chunks=(3, 16, 16))
    g = _global(proc.run(record_from(data, {"dim_order": "CYX"})))
    assert g["coloc_n_channels"] == 3
    for m in _ARRAY_METRICS:
        assert len(g[m]) == _n_pairs(3)  # == 3


def test_all_array_metrics_present(proc):
    data = da.from_array(np.random.default_rng(0).random((2, 16, 16)).astype(np.float32),
                         chunks=(2, 16, 16))
    g = _global(proc.run(record_from(data, {"dim_order": "CYX"})))
    for m in _ARRAY_METRICS:
        assert m in g
        assert isinstance(g[m], np.ndarray)
        assert g[m].dtype == np.float32


# ---------------------------------------------------------------------------
# Rollup tree structure
# ---------------------------------------------------------------------------

def test_czyx_rollup_has_per_z_rows(proc, monkeypatch):
    monkeypatch.setenv("PIXEL_PATROL_STATS_TILE_SIZE", "8")
    data = da.from_array(np.ones((2, 3, 8, 8), dtype=np.float32), chunks=(2, 1, 8, 8))
    result = proc.run(record_from(data, {"dim_order": "CZYX"}))
    per_z = [r for r in result if "dim_z" in r and r.get("dim_y") is None and r.get("dim_x") is None
             and r["obs_level"] == 1]
    assert len(per_z) == 3
    assert {r["dim_z"] for r in per_z} == {0, 1, 2}


def test_czyx_rollup_per_z_rows_have_coloc_metrics(proc, monkeypatch):
    monkeypatch.setenv("PIXEL_PATROL_STATS_TILE_SIZE", "8")
    rng = np.random.default_rng(10)
    c0 = rng.random((3, 8, 8)).astype(np.float32)
    c1 = rng.random((3, 8, 8)).astype(np.float32)
    data = da.from_array(np.stack([c0, c1]), chunks=(2, 1, 8, 8))
    result = proc.run(record_from(data, {"dim_order": "CZYX"}))
    per_z = [r for r in result if "dim_z" in r and r.get("dim_y") is None and r["obs_level"] == 1]
    for row in per_z:
        assert len(row["coloc_pearson_r"]) == 1
        assert np.isfinite(row["coloc_pearson_r"][0])


def test_tile_rows_have_coloc_metrics(proc, monkeypatch):
    monkeypatch.setenv("PIXEL_PATROL_STATS_TILE_SIZE", "4")
    rng = np.random.default_rng(11)
    data = da.from_array(rng.random((2, 8, 8)).astype(np.float32), chunks=(2, 4, 4))
    result = proc.run(record_from(data, {"dim_order": "CYX"}))
    tiles = [r for r in result if r.get("dim_y") is not None and r.get("dim_x") is not None]
    assert len(tiles) == 4  # 2×2 tiles
    for row in tiles:
        assert len(row["coloc_pearson_r"]) == 1
        assert np.isfinite(row["coloc_pearson_r"][0])


def test_coloc_n_channels_present_in_all_rows(proc, monkeypatch):
    monkeypatch.setenv("PIXEL_PATROL_STATS_TILE_SIZE", "8")
    data = da.from_array(np.ones((3, 8, 8), dtype=np.float32), chunks=(3, 8, 8))
    result = proc.run(record_from(data, {"dim_order": "CYX"}))
    for row in result:
        assert row["coloc_n_channels"] == 3


def test_czyx_rollup_tree_row_count(proc, monkeypatch):
    """CZYX C=2, Z=3, one XY tile: 3 leaf + 1 (dim_y,dim_x) + 3 per-Z + 1 global = 8."""
    monkeypatch.setenv("PIXEL_PATROL_STATS_TILE_SIZE", "256")
    data = da.from_array(np.ones((2, 3, 16, 16), dtype=np.float32), chunks=(2, 1, 16, 16))
    result = proc.run(record_from(data, {"dim_order": "CZYX"}))
    assert len(result) == 8


def test_cyx_tile_rows_aggregate_to_global(proc, monkeypatch):
    """Global Pearson r is the mean of per-tile values."""
    monkeypatch.setenv("PIXEL_PATROL_STATS_TILE_SIZE", "4")
    rng = np.random.default_rng(12)
    c0 = rng.random((8, 8)).astype(np.float32)
    c1 = rng.random((8, 8)).astype(np.float32)
    data = da.from_array(np.stack([c0, c1]), chunks=(2, 4, 4))
    result = proc.run(record_from(data, {"dim_order": "CYX"}))
    tiles = [r for r in result if r.get("dim_y") is not None and r.get("dim_x") is not None]
    tile_rs = [r["coloc_pearson_r"][0] for r in tiles if np.isfinite(r["coloc_pearson_r"][0])]
    g = _global(result)
    assert abs(g["coloc_pearson_r"][0] - float(np.mean(tile_rs))) < 1e-5


# ---------------------------------------------------------------------------
# Pearson r correctness (checked on global row)
# ---------------------------------------------------------------------------

def test_identical_channels_pearson_one(proc):
    rng = np.random.default_rng(1)
    ch = rng.random((16, 16)).astype(np.float32)
    data = da.from_array(np.stack([ch, ch]), chunks=(2, 16, 16))
    assert abs(_global(proc.run(record_from(data, {"dim_order": "CYX"})))["coloc_pearson_r"][0] - 1.0) < 1e-4


def test_opposite_channels_pearson_minus_one(proc):
    rng = np.random.default_rng(2)
    ch = rng.random((16, 16)).astype(np.float32)
    data = da.from_array(np.stack([ch, -ch]), chunks=(2, 16, 16))
    assert abs(_global(proc.run(record_from(data, {"dim_order": "CYX"})))["coloc_pearson_r"][0] - (-1.0)) < 1e-4


def test_independent_channels_pearson_near_zero(proc):
    rng = np.random.default_rng(3)
    c1 = rng.random((64, 64)).astype(np.float32)
    c2 = rng.random((64, 64)).astype(np.float32)
    data = da.from_array(np.stack([c1, c2]), chunks=(2, 64, 64))
    assert abs(_global(proc.run(record_from(data, {"dim_order": "CYX"})))["coloc_pearson_r"][0]) < 0.3


def test_pair_order_matches_combinations(proc):
    """Values at pair index (ci,cj) should match manually computed r (single-tile images)."""
    rng = np.random.default_rng(4)
    n_c = 3
    channels = [rng.random((32, 32)).astype(np.float32) for _ in range(n_c)]
    data = da.from_array(np.stack(channels), chunks=(n_c, 32, 32))
    g = _global(proc.run(record_from(data, {"dim_order": "CYX"})))

    for pi, (ci, cj) in enumerate(combinations(range(n_c), 2)):
        c1, c2 = channels[ci].ravel(), channels[cj].ravel()
        expected_r = float(np.corrcoef(c1, c2)[0, 1])
        assert abs(g["coloc_pearson_r"][pi] - expected_r) < 0.05, \
            f"pair ({ci},{cj}) idx={pi}: got {g['coloc_pearson_r'][pi]:.3f}, expected {expected_r:.3f}"


# ---------------------------------------------------------------------------
# Intensity invariance
# ---------------------------------------------------------------------------

def test_pearson_invariant_to_additive_offset(proc):
    rng = np.random.default_rng(5)
    c1 = rng.random((32, 32)).astype(np.float32)
    c2 = rng.random((32, 32)).astype(np.float32)
    base   = da.from_array(np.stack([c1, c2]),          chunks=(2, 32, 32))
    offset = da.from_array(np.stack([c1 + 1000., c2]),  chunks=(2, 32, 32))
    r_base   = _global(proc.run(record_from(base,   {"dim_order": "CYX"})))["coloc_pearson_r"][0]
    r_offset = _global(proc.run(record_from(offset, {"dim_order": "CYX"})))["coloc_pearson_r"][0]
    assert abs(r_base - r_offset) < 1e-4


def test_pearson_invariant_to_multiplicative_scale(proc):
    rng = np.random.default_rng(6)
    c1 = rng.random((32, 32)).astype(np.float32)
    c2 = rng.random((32, 32)).astype(np.float32)
    base  = da.from_array(np.stack([c1, c2]),          chunks=(2, 32, 32))
    scale = da.from_array(np.stack([c1 * 500., c2]),   chunks=(2, 32, 32))
    r_base  = _global(proc.run(record_from(base,  {"dim_order": "CYX"})))["coloc_pearson_r"][0]
    r_scale = _global(proc.run(record_from(scale, {"dim_order": "CYX"})))["coloc_pearson_r"][0]
    assert abs(r_base - r_scale) < 1e-4


# ---------------------------------------------------------------------------
# SSIM structure ≈ Pearson r
# ---------------------------------------------------------------------------

def test_ssim_structure_close_to_pearson_r(proc):
    rng = np.random.default_rng(7)
    c1 = rng.random((64, 64)).astype(np.float32)
    c2 = 0.7 * c1 + 0.3 * rng.random((64, 64)).astype(np.float32)
    data = da.from_array(np.stack([c1, c2]), chunks=(2, 64, 64))
    g = _global(proc.run(record_from(data, {"dim_order": "CYX"})))
    assert abs(g["coloc_ssim_structure"][0] - g["coloc_pearson_r"][0]) < 0.05


# ---------------------------------------------------------------------------
# Multi-dimensional inputs
# ---------------------------------------------------------------------------

def test_czyx_input(proc):
    data = da.from_array(np.ones((2, 3, 16, 16), dtype=np.float32), chunks=(2, 3, 8, 8))
    result = proc.run(record_from(data, {"dim_order": "CZYX"}))
    assert isinstance(result, list) and len(result) > 0
    assert _global(result)["coloc_n_channels"] == 2


def test_tcyx_input(proc):
    data = da.from_array(np.ones((4, 2, 16, 16), dtype=np.float32), chunks=(4, 2, 8, 8))
    result = proc.run(record_from(data, {"dim_order": "TCYX"}))
    assert isinstance(result, list) and len(result) > 0
    assert _global(result)["coloc_n_channels"] == 2


def test_tcyx_has_per_t_rows(proc, monkeypatch):
    monkeypatch.setenv("PIXEL_PATROL_STATS_TILE_SIZE", "8")
    data = da.from_array(np.ones((4, 2, 8, 8), dtype=np.float32), chunks=(1, 2, 8, 8))
    result = proc.run(record_from(data, {"dim_order": "TCYX"}))
    per_t = [r for r in result if "dim_t" in r and r.get("dim_y") is None and r["obs_level"] == 1]
    assert len(per_t) == 4
    assert {r["dim_t"] for r in per_t} == {0, 1, 2, 3}
