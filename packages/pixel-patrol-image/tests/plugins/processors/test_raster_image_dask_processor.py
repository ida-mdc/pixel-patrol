"""Tests for RasterImageDaskProcessor – full aggregation tree, obs_level scheme."""

import dask.array as da
import numpy as np
import pytest

from pixel_patrol_base.config import HISTOGRAM_BINS
from pixel_patrol_base.core.record import record_from
from pixel_patrol_image.plugins.processors.raster_image_dask_processor import RasterImageDaskProcessor


@pytest.fixture
def proc() -> RasterImageDaskProcessor:
    return RasterImageDaskProcessor()


# ---------------------------------------------------------------------------
# Result helpers
# ---------------------------------------------------------------------------

def _global(result):
    return next(r for r in result if r["obs_level"] == 0)


def _at_depth(result, depth):
    return [r for r in result if r["obs_level"] == depth]


def _tiles(result):
    """Deepest rows with both dim_x and dim_y set (all dims fixed)."""
    candidates = [r for r in result if "dim_x" in r and "dim_y" in r]
    if not candidates:
        return []
    max_depth = max(r["obs_level"] for r in candidates)
    return [r for r in candidates if r["obs_level"] == max_depth]


def _non_spatial(result):
    """Rows with no spatial dims set (global + non-spatial slices)."""
    return [r for r in result if "dim_x" not in r and "dim_y" not in r]


def _spatial(result):
    """Rows with at least one spatial dim (X or Y) set."""
    return [r for r in result if "dim_x" in r or "dim_y" in r]


def _reference_histogram_flat(values: np.ndarray) -> np.ndarray:
    """
    Histogram over all finite pixels using the same bin mapping as tile kernels:
    linear bins from global min to global max, ``idx = clip(floor((v-g_min)/bw), 0, BINS-1)``.
    """
    flat = np.asarray(values, dtype=np.float64).ravel()
    flat = flat[np.isfinite(flat)]
    n = flat.size
    counts = np.zeros(HISTOGRAM_BINS, dtype=np.float64)
    if n == 0:
        return counts
    g_min = float(np.min(flat))
    g_max = float(np.max(flat))
    if g_min >= g_max:
        counts[0] = n
        return counts
    bw = (g_max - g_min) / HISTOGRAM_BINS
    idx = np.clip(((flat - g_min) / bw).astype(int), 0, HISTOGRAM_BINS - 1)
    np.add.at(counts, idx, 1.0)
    return counts


def _truth_mean_in_tile(data_yx: np.ndarray, dim_y: int, dim_x: int, tile_size: int) -> float:
    """Mean of pixels covered by the tile footprint on the real image (partial tiles OK)."""
    h, w = data_yx.shape
    sl = data_yx[dim_y : min(dim_y + tile_size, h), dim_x : min(dim_x + tile_size, w)]
    return float(np.nanmean(sl))


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------

def test_returns_list(proc):
    data = da.from_array(np.ones((4, 4), dtype=np.float32), chunks=(2, 2))
    result = proc.run(record_from(data, {"dim_order": "YX"}))
    assert isinstance(result, list) and len(result) >= 1


def test_global_row_obs_level_zero(proc):
    data = da.from_array(np.ones((4, 4), dtype=np.float32), chunks=(2, 2))
    result = proc.run(record_from(data, {"dim_order": "YX"}))
    assert _global(result)["obs_level"] == 0


def test_global_row_has_required_keys(proc):
    data = da.from_array(np.arange(16, dtype=np.uint8).reshape(4, 4), chunks=(2, 2))
    row = _global(proc.run(record_from(data, {"dim_order": "YX"})))
    for k in ("mean_intensity", "std_intensity", "min_intensity", "max_intensity",
               "finite_pixel_count",
               "histogram_counts", "histogram_min", "histogram_max", "histogram_nan_count",
               "michelson_contrast", "mscn_variance", "local_std_ratio"):
        assert k in row, f"Missing key: {k}"
    # thumbnail is produced by the separate ThumbnailProcessor, not this processor
    assert "thumbnail" not in row


def test_lowercase_dim_order_still_runs_quality(proc):
    """BioIO may report dims in lowercase; spatial axes must still match."""
    rng = np.random.default_rng(0)
    data = da.from_array(rng.integers(10, 200, (8, 8), dtype=np.uint8).astype(np.float32), chunks=(4, 4))
    row = _global(proc.run(record_from(data, {"dim_order": "yx"})))
    assert np.isfinite(row["michelson_contrast"])
    assert np.isfinite(row["mscn_variance"])
    assert np.isfinite(row["local_std_ratio"])


# ---------------------------------------------------------------------------
# Full aggregation tree: row counts and obs_level values
# ---------------------------------------------------------------------------

def test_full_tree_yx_row_count(proc, monkeypatch):
    """YX image with 2×2 tiles: depth-0=1, depth-1=4 (2 X + 2 Y), depth-2=4 tiles."""
    monkeypatch.setenv("PIXEL_PATROL_STATS_TILE_SIZE", "4")
    data = da.from_array(np.ones((8, 8), dtype=np.float32), chunks=(4, 4))
    result = proc.run(record_from(data, {"dim_order": "YX"}))
    assert len(_at_depth(result, 0)) == 1
    assert len(_at_depth(result, 1)) == 4
    assert len(_at_depth(result, 2)) == 4


def test_full_tree_tyx_row_count(proc, monkeypatch):
    """TYX T=2, one XY tile: omit depth-1 dim_x/dim_y strips (single tile); omit (t,x)/(t,y)."""
    monkeypatch.setenv("PIXEL_PATROL_STATS_TILE_SIZE", "8")
    data = da.from_array(np.ones((2, 8, 8), dtype=np.float32), chunks=(1, 8, 8))
    result = proc.run(record_from(data, {"dim_order": "TYX"}))
    assert len(_at_depth(result, 0)) == 1
    assert len(_at_depth(result, 1)) == 2   # dim_t×2 only
    assert len(_at_depth(result, 2)) == 1   # (y,x)×1 only
    assert len(_at_depth(result, 3)) == 2


def test_full_tree_czyx_single_values(proc, monkeypatch):
    """CZYX C=1, Z=1, one XY tile: degenerate dim_x/dim_y-only rollups omitted → 8 rows."""
    monkeypatch.setenv("PIXEL_PATROL_STATS_TILE_SIZE", "8")
    data = da.from_array(np.ones((1, 1, 8, 8), dtype=np.float32), chunks=(1, 1, 8, 8))
    result = proc.run(record_from(data, {"dim_order": "CZYX"}))
    assert len(result) == 8


def test_full_tree_czyx_row_count(proc, monkeypatch):
    """CZYX C=2, Z=2, one XY tile: full power-set is halved once spatial strips are degenerate."""
    monkeypatch.setenv("PIXEL_PATROL_STATS_TILE_SIZE", "8")
    data = da.from_array(np.ones((2, 2, 8, 8), dtype=np.float32), chunks=(1, 1, 8, 8))
    result = proc.run(record_from(data, {"dim_order": "CZYX"}))
    assert len(result) == 18


def test_obs_level_yx_tile(proc, monkeypatch):
    """YX tile rows: obs_level=2 (X + Y fixed)."""
    monkeypatch.setenv("PIXEL_PATROL_STATS_TILE_SIZE", "4")
    data = da.from_array(np.ones((4, 4), dtype=np.float32), chunks=(4, 4))
    for row in _tiles(proc.run(record_from(data, {"dim_order": "YX"}))):
        assert row["obs_level"] == 2


def test_obs_level_tyx_tile(proc, monkeypatch):
    """TYX tile rows: obs_level=3 (T + X + Y fixed)."""
    monkeypatch.setenv("PIXEL_PATROL_STATS_TILE_SIZE", "4")
    data = da.from_array(np.ones((2, 4, 4), dtype=np.float32), chunks=(1, 4, 4))
    for row in _tiles(proc.run(record_from(data, {"dim_order": "TYX"}))):
        assert row["obs_level"] == 3


def test_obs_level_tczyx_tile(proc, monkeypatch):
    """TCZYX tile rows: obs_level=5 (T+C+Z+X+Y all fixed)."""
    monkeypatch.setenv("PIXEL_PATROL_STATS_TILE_SIZE", "4")
    data = da.from_array(np.ones((2, 2, 2, 4, 4), dtype=np.float32), chunks=(1, 1, 1, 4, 4))
    for row in _tiles(proc.run(record_from(data, {"dim_order": "TCZYX"}))):
        assert row["obs_level"] == 5


# ---------------------------------------------------------------------------
# Non-spatial rechunking: each element becomes its own block
# ---------------------------------------------------------------------------

def test_nonspatial_rechunked_to_one(proc, monkeypatch):
    """T=4 stored with chunk=2: rechunked to chunk=1 → 4 per-T rows at depth=1."""
    monkeypatch.setenv("PIXEL_PATROL_STATS_TILE_SIZE", "8")
    data = da.from_array(np.ones((4, 8, 8), dtype=np.float32), chunks=(2, 8, 8))
    result = proc.run(record_from(data, {"dim_order": "TYX"}))
    per_t = [r for r in result if r["obs_level"] == 1 and "dim_t" in r]
    assert len(per_t) == 4
    assert {r["dim_t"] for r in per_t} == {0, 1, 2, 3}


def test_tile_count_yx(proc, monkeypatch):
    """(8,8) with 4-pixel XY chunk: 4 tile rows."""
    monkeypatch.setenv("PIXEL_PATROL_STATS_TILE_SIZE", "4")
    data = da.from_array(np.ones((8, 8), dtype=np.float32), chunks=(4, 4))
    assert len(_tiles(proc.run(record_from(data, {"dim_order": "YX"})))) == 4


def test_tile_count_czyx(proc, monkeypatch):
    """CZYX C=2, Z=2, 2×2 XY tiles: 2×2×2×2=16 tile rows."""
    monkeypatch.setenv("PIXEL_PATROL_STATS_TILE_SIZE", "4")
    data = da.from_array(np.ones((2, 2, 8, 8), dtype=np.float32), chunks=(1, 1, 4, 4))
    assert len(_tiles(proc.run(record_from(data, {"dim_order": "CZYX"})))) == 16


# ---------------------------------------------------------------------------
# Dim coordinates
# ---------------------------------------------------------------------------

def test_dim_coordinates_yx(proc, monkeypatch):
    """(8,8) with (4,4) chunks: dim_y ∈ {0,4}, dim_x ∈ {0,4}."""
    monkeypatch.setenv("PIXEL_PATROL_STATS_TILE_SIZE", "4")
    data = da.from_array(np.ones((8, 8), dtype=np.float32), chunks=(4, 4))
    rows = _tiles(proc.run(record_from(data, {"dim_order": "YX"})))
    assert {r["dim_y"] for r in rows} == {0, 4}
    assert {r["dim_x"] for r in rows} == {0, 4}


def test_dim_coordinates_match_chunk_data(proc, monkeypatch):
    """Corner tile (dim_y=0, dim_x=0) mean matches the stored pixel slice."""
    monkeypatch.setenv("PIXEL_PATROL_STATS_TILE_SIZE", "4")
    data = np.arange(8 * 8, dtype=np.float32).reshape(8, 8)
    arr = da.from_array(data, chunks=(4, 4))
    result = proc.run(record_from(arr, {"dim_order": "YX"}))
    corner = next(r for r in _tiles(result) if r["dim_y"] == 0 and r["dim_x"] == 0)
    assert corner["mean_intensity"] == pytest.approx(float(data[:4, :4].mean()))


def test_dim_t_values_tyx(proc):
    """TYX T=4 stored chunk=1: per-T depth-1 rows have dim_t ∈ {0,1,2,3}."""
    data = da.from_array(np.ones((4, 4, 4), dtype=np.float32), chunks=(1, 4, 4))
    rows = [r for r in proc.run(record_from(data, {"dim_order": "TYX"}))
            if r["obs_level"] == 1 and "dim_t" in r]
    assert {r["dim_t"] for r in rows} == {0, 1, 2, 3}


def test_per_dim_c_rows_czyx(proc):
    """CZYX C=3: depth-1 per-C rows have dim_c ∈ {0,1,2}, no spatial dims."""
    data = da.from_array(np.ones((3, 2, 8, 8), dtype=np.float32), chunks=(1, 1, 8, 8))
    result = proc.run(record_from(data, {"dim_order": "CZYX"}))
    per_c = [r for r in result if r["obs_level"] == 1 and "dim_c" in r]
    assert len(per_c) == 3
    assert {r["dim_c"] for r in per_c} == {0, 1, 2}
    for r in per_c:
        assert "dim_x" not in r and "dim_y" not in r and "dim_z" not in r


def test_per_cz_slice_rows_czyx(proc):
    """CZYX C=2, Z=3: depth-2 per-CZ rows cover all 6 combinations, no spatial dims."""
    data = da.from_array(np.ones((2, 3, 8, 8), dtype=np.float32), chunks=(1, 1, 8, 8))
    result = proc.run(record_from(data, {"dim_order": "CZYX"}))
    per_cz = [r for r in result if r["obs_level"] == 2
              and "dim_c" in r and "dim_z" in r
              and "dim_x" not in r and "dim_y" not in r]
    assert len(per_cz) == 6
    assert {r["dim_c"] for r in per_cz} == {0, 1}
    assert {r["dim_z"] for r in per_cz} == {0, 1, 2}


def test_per_cx_rows_czyx(proc, monkeypatch):
    """CZYX C=2, 2 X tiles: depth-2 per-CX rows (dim_c + dim_x, no dim_y or dim_z)."""
    monkeypatch.setenv("PIXEL_PATROL_STATS_TILE_SIZE", "4")
    data = da.from_array(np.ones((2, 2, 4, 8), dtype=np.float32), chunks=(1, 1, 4, 4))
    result = proc.run(record_from(data, {"dim_order": "CZYX"}))
    per_cx = [r for r in result if r["obs_level"] == 2
              and "dim_c" in r and "dim_x" in r
              and "dim_z" not in r and "dim_y" not in r]
    assert len(per_cx) == 4
    assert {r["dim_c"] for r in per_cx} == {0, 1}
    assert {r["dim_x"] for r in per_cx} == {0, 4}


def test_czx_profile_rows(proc, monkeypatch):
    """When only one Y tile exists, per-(C,Z,X) rollups duplicate normalized leaf tiles → omitted."""
    monkeypatch.setenv("PIXEL_PATROL_STATS_TILE_SIZE", "4")
    data = da.from_array(np.ones((2, 2, 4, 8), dtype=np.float32), chunks=(1, 1, 4, 4))
    result = proc.run(record_from(data, {"dim_order": "CZYX"}))
    dup_rollups = [r for r in result if r["obs_level"] == 3
                   and "dim_c" in r and "dim_z" in r and "dim_x" in r and "dim_y" not in r]
    assert len(dup_rollups) == 0
    leaf_czx = [
        r for r in result
        if r["obs_level"] == 4
        and "dim_c" in r and "dim_z" in r and "dim_x" in r and "dim_y" in r
    ]
    assert len(leaf_czx) == 8  # 2 C × 2 Z × 2 X tiles


def test_spatial_profile_rows_depth1(proc, monkeypatch):
    """CZYX: global per-X rows at depth=1 cover all X tile positions."""
    monkeypatch.setenv("PIXEL_PATROL_STATS_TILE_SIZE", "4")
    data = da.from_array(np.ones((2, 2, 4, 8), dtype=np.float32), chunks=(1, 1, 4, 4))
    result = proc.run(record_from(data, {"dim_order": "CZYX"}))
    per_x = [r for r in result if r["obs_level"] == 1 and "dim_x" in r]
    assert len(per_x) == 2
    assert {r["dim_x"] for r in per_x} == {0, 4}


# ---------------------------------------------------------------------------
# Global stats correctness
# ---------------------------------------------------------------------------

def test_global_mean_std_min_max(proc):
    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
    row = _global(proc.run(record_from(da.from_array(data, chunks=(3, 3)), {"dim_order": "YX"})))
    assert row["mean_intensity"] == pytest.approx(5.0, rel=1e-5)
    assert row["min_intensity"]  == pytest.approx(1.0, rel=1e-5)
    assert row["max_intensity"]  == pytest.approx(9.0, rel=1e-5)
    assert row["std_intensity"]  == pytest.approx(2.5819888, rel=1e-5)


def test_constant_plane(proc):
    data = np.full((10, 10), 42.0, dtype=np.float32)
    row = _global(proc.run(record_from(da.from_array(data, chunks=(5, 5)), {"dim_order": "YX"})))
    assert row["mean_intensity"] == pytest.approx(42.0, rel=1e-5)
    assert row["std_intensity"]  == pytest.approx(0.0, rel=1e-5)


def test_nan_ignored_for_global_stats(proc):
    data = np.array([[0, 1, 2, 3, 4, np.nan]], dtype=np.float32)
    row = _global(proc.run(record_from(da.from_array(data, chunks=(1, 6)), {"dim_order": "YX"})))
    assert row["mean_intensity"] == pytest.approx(2.0, rel=1e-5)
    assert row["min_intensity"]  == pytest.approx(0.0, rel=1e-5)
    assert row["max_intensity"]  == pytest.approx(4.0, rel=1e-5)


def test_global_not_poisoned_by_nan_slice(proc):
    data = np.array(
        [[[1.0, 2.0], [3.0, 4.0]], [[np.nan, np.nan], [np.nan, np.nan]]],
        dtype=np.float32,
    )
    row = _global(proc.run(record_from(da.from_array(data, chunks=(1, 2, 2)), {"dim_order": "TYX"})))
    assert row["mean_intensity"] == pytest.approx(2.5, rel=1e-5)
    assert row["min_intensity"]  == pytest.approx(1.0, rel=1e-5)
    assert row["max_intensity"]  == pytest.approx(4.0, rel=1e-5)


# ---------------------------------------------------------------------------
# Aggregation correctness across tree levels
# ---------------------------------------------------------------------------

def test_per_c_stats_differ_for_non_uniform(proc):
    """Non-uniform channels: per-C means match channel values; global mean is midpoint."""
    c0 = np.zeros((8, 8), dtype=np.float32)
    c1 = np.ones((8, 8), dtype=np.float32) * 10.0
    arr = da.from_array(np.stack([c0, c1]), chunks=(1, 8, 8))
    result = proc.run(record_from(arr, {"dim_order": "CYX"}))
    per_c = sorted([r for r in result if r["obs_level"] == 1 and "dim_c" in r],
                   key=lambda r: r["dim_c"])
    assert per_c[0]["mean_intensity"] == pytest.approx(0.0, abs=1e-5)
    assert per_c[1]["mean_intensity"] == pytest.approx(10.0, rel=1e-5)
    assert _global(result)["mean_intensity"] == pytest.approx(5.0, rel=1e-5)


def test_per_c_mean_matches_global_for_uniform(proc):
    """Uniform image: every row in the tree has the same mean."""
    arr = da.from_array(np.full((2, 8, 8), 7.0, dtype=np.float32), chunks=(1, 8, 8))
    result = proc.run(record_from(arr, {"dim_order": "CYX"}))
    g_mean = float(_global(result)["mean_intensity"])
    for r in result:
        assert r["mean_intensity"] == pytest.approx(g_mean, rel=1e-4)


def test_per_tile_mean_matches_spatial_region(proc, monkeypatch):
    """Each tile's mean matches the pixel values in its spatial region."""
    monkeypatch.setenv("PIXEL_PATROL_STATS_TILE_SIZE", "4")
    data = np.zeros((8, 8), dtype=np.float32)
    data[:4, :4] = 1.0
    data[:4, 4:] = 2.0
    data[4:, :4] = 3.0
    data[4:, 4:] = 4.0
    arr = da.from_array(data, chunks=(4, 4))
    result = proc.run(record_from(arr, {"dim_order": "YX"}))
    tile_means = {(r["dim_y"], r["dim_x"]): float(r["mean_intensity"]) for r in _tiles(result)}
    assert tile_means[(0, 0)] == pytest.approx(1.0, rel=1e-5)
    assert tile_means[(0, 4)] == pytest.approx(2.0, rel=1e-5)
    assert tile_means[(4, 0)] == pytest.approx(3.0, rel=1e-5)
    assert tile_means[(4, 4)] == pytest.approx(4.0, rel=1e-5)


def test_all_levels_equal_for_uniform_data(proc, monkeypatch):
    """Uniform image: mean is identical at every depth level."""
    monkeypatch.setenv("PIXEL_PATROL_STATS_TILE_SIZE", "4")
    arr = da.from_array(np.full((2, 8, 8), 5.0, dtype=np.float32), chunks=(1, 4, 4))
    result = proc.run(record_from(arr, {"dim_order": "CYX"}))
    g_mean = float(_global(result)["mean_intensity"])
    for row in result:
        assert row["mean_intensity"] == pytest.approx(g_mean, rel=1e-4)


# ---------------------------------------------------------------------------
# Histogram
# ---------------------------------------------------------------------------

def test_histogram_uint8_bins(proc):
    data = np.array([[0, 128, 255], [50, 100, 200]], dtype=np.uint8)
    row = _global(proc.run(record_from(da.from_array(data, chunks=(2, 3)), {"dim_order": "YX"})))
    assert len(row["histogram_counts"]) == HISTOGRAM_BINS
    assert row["histogram_counts"][0]  == 1
    assert row["histogram_counts"][50] == 1
    assert int(row["histogram_counts"].sum()) == 6
    assert row["histogram_min"] == 0.0
    assert row["histogram_max"] == 255.0


def test_histogram_float_range(proc):
    data = np.array([[0.0, 0.5, 1.0], [0.25, 0.75, 0.9]], dtype=np.float32)
    row = _global(proc.run(record_from(da.from_array(data, chunks=(2, 3)), {"dim_order": "YX"})))
    assert int(row["histogram_counts"].sum()) == 6
    assert row["histogram_min"] == pytest.approx(0.0, rel=1e-5)
    assert row["histogram_max"] == pytest.approx(1.0, rel=1e-5)


def test_histogram_nan_count(proc):
    data = np.array([[0.0, 1.0, 2.0, 1.0], [np.nan, np.nan, np.nan, 255.0]], dtype=np.float32)
    assert _global(proc.run(
        record_from(da.from_array(data, chunks=(2, 4)), {"dim_order": "YX"})
    ))["histogram_nan_count"] == 3


def test_histogram_absent_in_all_spatial_rows(proc, monkeypatch):
    """Any row with dim_x or dim_y set must not carry histogram_counts."""
    monkeypatch.setenv("PIXEL_PATROL_STATS_TILE_SIZE", "4")
    data = np.random.randint(0, 256, size=(2, 2, 4, 4), dtype=np.uint8)
    result = proc.run(record_from(da.from_array(data, chunks=(1, 1, 4, 4)), {"dim_order": "CZYX"}))
    for row in _spatial(result):
        assert "histogram_counts" not in row, (
            f"Unexpected histogram at obs_level={row['obs_level']}"
        )


def test_histogram_present_in_all_non_spatial_rows(proc, monkeypatch):
    """Every non-spatial row (global, per-C, per-Z, per-CZ, …) carries histogram_counts."""
    monkeypatch.setenv("PIXEL_PATROL_STATS_TILE_SIZE", "8")
    data = np.random.randint(0, 256, size=(2, 2, 8, 8), dtype=np.uint8)
    result = proc.run(record_from(da.from_array(data, chunks=(1, 1, 8, 8)), {"dim_order": "CZYX"}))
    for row in _non_spatial(result):
        assert "histogram_counts" in row, (
            f"Missing histogram at obs_level={row['obs_level']} "
            f"dims={[k for k in row if k.startswith('dim_')]}"
        )


def test_per_c_histogram_pixel_count(proc):
    """Per-C histogram total == number of pixels in that channel."""
    data = np.random.randint(0, 256, size=(3, 8, 8), dtype=np.uint8)
    arr = da.from_array(data, chunks=(1, 8, 8))
    result = proc.run(record_from(arr, {"dim_order": "CYX"}))
    per_c = [r for r in result if r["obs_level"] == 1 and "dim_c" in r]
    for r in per_c:
        assert int(r["histogram_counts"].sum()) == 64


def test_global_histogram_pixel_count(proc):
    data = np.random.randint(0, 256, size=(3, 8, 8), dtype=np.uint8)
    arr = da.from_array(data, chunks=(1, 8, 8))
    assert int(_global(proc.run(record_from(arr, {"dim_order": "CYX"})))["histogram_counts"].sum()) == 3 * 64


def test_per_cz_histogram_pixel_count(proc):
    """Per-CZ histogram total == pixels in that CZ slice."""
    data = np.random.randint(0, 256, size=(2, 3, 8, 8), dtype=np.uint8)
    arr = da.from_array(data, chunks=(1, 1, 8, 8))
    result = proc.run(record_from(arr, {"dim_order": "CZYX"}))
    per_cz = [r for r in result if r["obs_level"] == 2
              and "dim_c" in r and "dim_z" in r and "dim_x" not in r]
    for r in per_cz:
        assert int(r["histogram_counts"].sum()) == 64


# ---------------------------------------------------------------------------
# Aggregated metrics vs pixel-level ground truth (user expectations)
# ---------------------------------------------------------------------------

def test_global_basic_stats_match_pixel_truth_with_partial_edge_tiles(proc, monkeypatch):
    """
    Global mean/min/max must equal statistics over **all pixels**, not any shortcut per tile.
    Uses a size that does not tile evenly so edge tiles cover fewer real pixels.
    """
    monkeypatch.setenv("PIXEL_PATROL_STATS_TILE_SIZE", "8")
    rng = np.random.default_rng(42)
    data = rng.standard_normal((10, 10)).astype(np.float32)
    result = proc.run(record_from(da.from_array(data, chunks=(10, 10)), {"dim_order": "YX"}))
    row = _global(result)

    assert row["mean_intensity"] == pytest.approx(float(np.nanmean(data)), rel=1e-4, abs=1e-5)
    assert row["min_intensity"] == pytest.approx(float(np.nanmin(data)), rel=1e-5)
    assert row["max_intensity"] == pytest.approx(float(np.nanmax(data)), rel=1e-5)
    assert int(row["finite_pixel_count"]) == int(np.isfinite(data).sum())

    tiles = _tiles(result)
    assert len(tiles) >= 1
    ts = 8
    for tr in tiles:
        dy, dx = int(tr["dim_y"]), int(tr["dim_x"])
        expected_tile_mean = _truth_mean_in_tile(data, dy, dx, ts)
        assert tr["mean_intensity"] == pytest.approx(expected_tile_mean, rel=1e-4, abs=1e-5)


def test_global_histogram_matches_all_pixels_after_multi_channel_merge(proc, monkeypatch):
    """
    Each slice histogram uses **that slice's** min/max; global merge rebins onto the union
    range via ``aggregate_histograms`` (bin-centre mapping). Total counts and grand range are
    exact; bin-wise shape is approximate vs a pixel-perfect histogram at global extrema.
    """
    monkeypatch.setenv("PIXEL_PATROL_STATS_TILE_SIZE", "16")
    rng = np.random.default_rng(7)
    low = rng.integers(0, 32, size=(28, 28), dtype=np.uint8)
    high = rng.integers(224, 256, size=(28, 28), dtype=np.uint8)
    stack = np.stack([low, high], axis=0)
    result = proc.run(record_from(da.from_array(stack, chunks=(1, 28, 28)), {"dim_order": "CYX"}))
    row = _global(result)

    ref = _reference_histogram_flat(stack)
    got = row["histogram_counts"].astype(np.float64)
    n = float(stack.size)
    assert got.sum() == pytest.approx(ref.sum()) == pytest.approx(n)
    assert row["histogram_min"] == pytest.approx(float(stack.min()))
    assert row["histogram_max"] == pytest.approx(float(stack.max()))
    # Bin-centre remap skews per-bin counts; normalized CDF stays close to pixel-perfect ref.
    cdf_got = np.cumsum(got) / got.sum()
    cdf_ref = np.cumsum(ref) / ref.sum()
    assert float(np.max(np.abs(cdf_got - cdf_ref))) <= 0.03


def test_global_histogram_integer_plane_matches_pixel_truth(proc, monkeypatch):
    """Single-plane uneven tiling: histogram totals and bins vs direct pixel binning."""
    monkeypatch.setenv("PIXEL_PATROL_STATS_TILE_SIZE", "8")
    rng = np.random.default_rng(99)
    data = rng.integers(0, 256, size=(11, 13), dtype=np.uint8)
    result = proc.run(record_from(da.from_array(data, chunks=(11, 13)), {"dim_order": "YX"}))
    row = _global(result)

    ref = _reference_histogram_flat(data)
    got = row["histogram_counts"].astype(np.int64)
    assert int(got.sum()) == data.size
    np.testing.assert_array_equal(got, np.round(ref).astype(np.int64))


# ---------------------------------------------------------------------------
# Tile metrics disabled
# ---------------------------------------------------------------------------

def test_tile_metrics_disabled_returns_only_global(proc, monkeypatch):
    monkeypatch.setenv("PIXEL_PATROL_RASTER_XY_TILE_METRICS", "0")
    data = np.ones((8, 8), dtype=np.float32)
    result = proc.run(record_from(da.from_array(data, chunks=(4, 4)), {"dim_order": "YX"}))
    assert len(result) == 1
    assert result[0]["obs_level"] == 0


# ---------------------------------------------------------------------------
# Compression artifact metrics
# ---------------------------------------------------------------------------

def test_compression_metrics_absent_when_disabled(proc, monkeypatch):
    monkeypatch.delenv("PIXEL_PATROL_METRICS_COMPRESSION", raising=False)
    data = np.linspace(0, 1, 64 * 64, dtype=np.float32).reshape(64, 64)
    row = _global(proc.run(record_from(da.from_array(data, chunks=(64, 64)), {"dim_order": "YX"})))
    assert "blocking_index" not in row
    assert "ringing_index" not in row


def test_compression_metrics_finite_when_enabled(proc, monkeypatch):
    monkeypatch.setenv("PIXEL_PATROL_METRICS_COMPRESSION", "1")
    data = np.linspace(0, 1, 64 * 64, dtype=np.float32).reshape(64, 64)
    row = _global(proc.run(record_from(da.from_array(data, chunks=(64, 64)), {"dim_order": "YX"})))
    assert np.isfinite(row["blocking_index"])
    assert np.isfinite(row["ringing_index"])


# ---------------------------------------------------------------------------
# Quality metrics
# ---------------------------------------------------------------------------

def test_quality_metrics_in_global_row(proc):
    data = np.linspace(0, 1, 40 * 40, dtype=np.float32).reshape(40, 40)
    row = _global(proc.run(record_from(da.from_array(data, chunks=(20, 20)), {"dim_order": "YX"})))
    assert np.isfinite(row["michelson_contrast"])
    assert np.isfinite(row["mscn_variance"])
    assert np.isfinite(row["local_std_ratio"])


def test_quality_metrics_present_in_all_rows(proc, monkeypatch):
    """Every row in the full tree carries finite quality metrics."""
    monkeypatch.setenv("PIXEL_PATROL_STATS_TILE_SIZE", "4")
    data = np.linspace(0, 1, 2 * 8 * 8, dtype=np.float32).reshape(2, 8, 8)
    result = proc.run(record_from(da.from_array(data, chunks=(1, 4, 4)), {"dim_order": "TYX"}))
    for row in result:
        assert np.isfinite(row["michelson_contrast"]), (
            f"michelson_contrast not finite at obs_level={row['obs_level']}"
        )
        assert np.isfinite(row["mscn_variance"]), (
            f"mscn_variance not finite at obs_level={row['obs_level']}"
        )
        assert np.isfinite(row["local_std_ratio"]), (
            f"local_std_ratio not finite at obs_level={row['obs_level']}"
        )
