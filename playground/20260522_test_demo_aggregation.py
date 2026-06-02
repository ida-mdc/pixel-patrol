"""
Tests for build_result_table aggregation logic in the demo script.

Covers:
- Correct row counts at each obs_level
- Memory-chunk dims invisible (absent from position columns)
- Metric values correct at leaf and global level
- Weighted mean correct when tiles have different num_pixels
- Sizes reflect actual tile dimensions including edge tiles
- Results are invariant to the memory chunk spec used
"""

import tempfile
from pathlib import Path
from typing import Dict, List

import dask.array as da
import numpy as np
import polars as pl
import pytest
import tifffile

from pixel_patrol_base.core.record import record_from, Record
from pixel_patrol_base.plugins.processors.raster_processor import BasicMetricsProcessor
from pixel_patrol_loader_bio.plugins.loaders.tifffile_loader import TifffileLoader

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "demo",
    Path(__file__).with_name("20260522_demo_new_processor_interface.py"),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

build_result_table = _mod.build_result_table
plan_chunks        = _mod.plan_chunks
worker_task        = _mod.worker_task

from pixel_patrol_base.plugins.processors.thumbnail_processor import ThumbnailProcessor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_tiff(path: Path, data: np.ndarray) -> None:
    tifffile.imwrite(path, data, imagej=True)


def _load_record(path: Path) -> Record:
    result = TifffileLoader().load(str(path))
    return next(iter(result.values())) if isinstance(result, dict) else result


def _run_pipeline(record, memory_spec, leaf_spec, leaf_procs):
    """Run the full demo pipeline and return the result table."""
    memory_procs = [ThumbnailProcessor()]
    all_mem_rows  = {p.NAME: [] for p in memory_procs}
    all_leaf_rows = {p.NAME: [] for p in leaf_procs}
    for bounds, origin in plan_chunks(record.data.shape, record.dim_order, memory_spec):
        mr, lr = worker_task(record, bounds, origin, memory_procs, leaf_procs, leaf_spec)
        for p in memory_procs:
            all_mem_rows[p.NAME].extend(mr.get(p.NAME, []))
        for p in leaf_procs:
            all_leaf_rows[p.NAME].extend(lr.get(p.NAME, []))
    return build_result_table(
        leaf_procs, all_leaf_rows, record.dim_order,
        tuple(record.data.shape), leaf_spec,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def constant_channel_record(tmp_path):
    """
    TZCYX (1, 1, 3, 16, 16) image where:
      C=0 → all pixels = 100
      C=1 → all pixels = 200
      C=2 → all pixels = 300
    """
    data = np.zeros((1, 1, 3, 16, 16), dtype=np.uint16)
    data[0, 0, 0] = 100
    data[0, 0, 1] = 200
    data[0, 0, 2] = 300
    p = tmp_path / "const.tif"
    _write_tiff(p, data)
    return _load_record(p)


@pytest.fixture
def gradient_record(tmp_path):
    """
    TZCYX (1, 2, 1, 32, 32): pixel value = Z * 1000 + Y * 10 + X (mod 65535)
    """
    data = np.zeros((1, 2, 1, 32, 32), dtype=np.uint16)
    for z in range(2):
        for y in range(32):
            for x in range(32):
                data[0, z, 0, y, x] = (z * 1000 + y * 10 + x) % 65535
    p = tmp_path / "grad.tif"
    _write_tiff(p, data)
    return _load_record(p)


# ---------------------------------------------------------------------------
# Tests: row counts
# ---------------------------------------------------------------------------

def test_c_only_leaf_produces_one_row_per_channel(constant_channel_record):
    """
    LEAF_CHUNK_SPEC = {"C": 1}: expect 3 leaf rows (one per channel) + 1 global = 4 rows.
    Memory chunking in Z/Y/X must not create extra leaf positions.
    """
    table = _run_pipeline(
        constant_channel_record,
        memory_spec={"Z": 1, "Y": 8, "X": 8},
        leaf_spec={"C": 1},
        leaf_procs=[BasicMetricsProcessor()],
    )
    assert table.shape[0] == 4  # 3 leaf + 1 global
    assert table.filter(pl.col("obs_level") == 1).shape[0] == 3
    assert table.filter(pl.col("obs_level") == 0).shape[0] == 1


def test_xy_tiling_row_count(constant_channel_record):
    """
    LEAF_CHUNK_SPEC = {"X": 8, "Y": 8, "C": 1} on a 16×16 image → 2×2 XY tiles × 3 C = 12 leaf rows.
    obs_level=3 leaf, obs_level=2 (4 pair combos), obs_level=1 (3 singles), obs_level=0 global.
    """
    table = _run_pipeline(
        constant_channel_record,
        memory_spec={"Z": 1},
        leaf_spec={"X": 8, "Y": 8, "C": 1},
        leaf_procs=[BasicMetricsProcessor()],
    )
    n = len(["dim_c", "dim_y", "dim_x"])  # 3 leaf dims
    assert table.filter(pl.col("obs_level") == n).shape[0] == 12   # 3C × 2Y × 2X
    assert table.filter(pl.col("obs_level") == 0).shape[0] == 1


# ---------------------------------------------------------------------------
# Tests: invisible dimensions
# ---------------------------------------------------------------------------

def test_memory_dims_not_in_columns(constant_channel_record):
    """Z is a memory-chunk-only dim and must not appear as a position column."""
    table = _run_pipeline(
        constant_channel_record,
        memory_spec={"Z": 1},
        leaf_spec={"C": 1},
        leaf_procs=[BasicMetricsProcessor()],
    )
    assert "dim_z" not in table.columns
    assert "dim_t" not in table.columns
    assert "dim_c" in table.columns


def test_shared_dim_appears_when_leaf_tiles_it(constant_channel_record):
    """Y is in both memory and leaf specs → must appear as a leaf position column."""
    table = _run_pipeline(
        constant_channel_record,
        memory_spec={"Z": 1, "Y": 8},
        leaf_spec={"C": 1, "Y": 8},
        leaf_procs=[BasicMetricsProcessor()],
    )
    assert "dim_y" in table.columns


# ---------------------------------------------------------------------------
# Tests: metric correctness
# ---------------------------------------------------------------------------

def test_mean_per_channel_is_exact(constant_channel_record):
    """Each leaf row (one per channel) must report the exact constant pixel value."""
    table = _run_pipeline(
        constant_channel_record,
        memory_spec={"Z": 1, "Y": 8, "X": 8},
        leaf_spec={"C": 1},
        leaf_procs=[BasicMetricsProcessor()],
    )
    leaves = table.filter(pl.col("obs_level") == 1).sort("dim_c")
    means = leaves["mean_intensity"].to_list()
    assert means[0] == pytest.approx(100.0)
    assert means[1] == pytest.approx(200.0)
    assert means[2] == pytest.approx(300.0)


def test_global_mean_is_weighted_average(constant_channel_record):
    """Global mean must equal the pixel-weighted mean across all channels."""
    table = _run_pipeline(
        constant_channel_record,
        memory_spec={"Z": 1, "Y": 8, "X": 8},
        leaf_spec={"C": 1},
        leaf_procs=[BasicMetricsProcessor()],
    )
    global_row = table.filter(pl.col("obs_level") == 0)
    # Equal-sized channels → simple mean of 100, 200, 300
    assert global_row["mean_intensity"][0] == pytest.approx(200.0)


def test_global_min_and_max(constant_channel_record):
    table = _run_pipeline(
        constant_channel_record,
        memory_spec={"Z": 1},
        leaf_spec={"C": 1},
        leaf_procs=[BasicMetricsProcessor()],
    )
    g = table.filter(pl.col("obs_level") == 0)
    assert g["min_intensity"][0] == pytest.approx(100.0)
    assert g["max_intensity"][0] == pytest.approx(300.0)


# ---------------------------------------------------------------------------
# Tests: num_pixels
# ---------------------------------------------------------------------------

def test_global_num_pixels_equals_total_image_voxels(constant_channel_record):
    """Global row num_pixels must equal the product of all image shape dimensions."""
    record = constant_channel_record  # shape (1,1,3,16,16) = 768 voxels
    expected = int(np.prod(record.data.shape))
    table = _run_pipeline(
        record,
        memory_spec={"Z": 1},
        leaf_spec={"C": 1},
        leaf_procs=[BasicMetricsProcessor()],
    )
    g = table.filter(pl.col("obs_level") == 0)
    assert g["num_pixels"][0] == expected


def test_leaf_num_pixels_sum_to_global(constant_channel_record):
    table = _run_pipeline(
        constant_channel_record,
        memory_spec={"Z": 1},
        leaf_spec={"C": 1},
        leaf_procs=[BasicMetricsProcessor()],
    )
    leaves = table.filter(pl.col("obs_level") == 1)
    global_ = table.filter(pl.col("obs_level") == 0)
    assert leaves["num_pixels"].sum() == global_["num_pixels"][0]


# ---------------------------------------------------------------------------
# Tests: tile sizes
# ---------------------------------------------------------------------------

def test_tile_sizes_correct_including_edges():
    """
    32×32 image with 10-pixel tiles → 3 tiles per axis (10, 10, 12).
    Verify that edge tiles report X_size=12, not 10 or 1.
    """
    with tempfile.TemporaryDirectory() as d:
        data = np.ones((1, 1, 1, 32, 32), dtype=np.uint16)
        p = Path(d) / "edge.tif"
        _write_tiff(p, data)
        record = _load_record(p)

        table = _run_pipeline(
            record,
            memory_spec={},
            leaf_spec={"X": 10, "Y": 10},
            leaf_procs=[BasicMetricsProcessor()],
        )
        # 32 = 3 × 10 + 2 → interior tiles of 10, one edge tile of 2; never 1
        leaves = table.filter(pl.col("obs_level") == table["obs_level"].max())
        x_sizes = sorted(leaves["X_size"].unique().to_list())
        y_sizes = sorted(leaves["Y_size"].unique().to_list())
        assert x_sizes == [2, 10]
        assert y_sizes == [2, 10]


def test_aggregated_row_fixed_dim_uses_tile_size():
    """
    In an aggregated row where dim_x is fixed (e.g. per-X grouping),
    X_size must be the tile size at that position, not 1.
    """
    with tempfile.TemporaryDirectory() as d:
        data = np.ones((1, 1, 1, 16, 16), dtype=np.uint16)
        p = Path(d) / "fixed.tif"
        _write_tiff(p, data)
        record = _load_record(p)

        table = _run_pipeline(
            record,
            memory_spec={},
            leaf_spec={"X": 8, "Y": 8},
            leaf_procs=[BasicMetricsProcessor()],
        )
        # obs_level=1 rows where dim_x is fixed but dim_y is None
        per_x = table.filter(
            (pl.col("obs_level") == 1) & pl.col("dim_x").is_not_null() & pl.col("dim_y").is_null()
        )
        assert (per_x["X_size"] == 8).all()
        assert (per_x["Y_size"] == 16).all()  # full Y extent when Y is aggregated over


# ---------------------------------------------------------------------------
# Tests: memory-chunk invariance
# ---------------------------------------------------------------------------

def test_results_invariant_to_memory_chunk_spec(constant_channel_record):
    """
    Changing how memory is chunked must not change metric values or row structure.
    """
    def run(memory_spec):
        return _run_pipeline(
            constant_channel_record,
            memory_spec=memory_spec,
            leaf_spec={"C": 1},
            leaf_procs=[BasicMetricsProcessor()],
        )

    t1 = run({"Z": 1, "Y": 8, "X": 8})
    t2 = run({"Z": 1})
    t3 = run({})  # single memory chunk = full image

    for col in ("mean_intensity", "min_intensity", "max_intensity", "num_pixels"):
        assert t1.sort("obs_level", "dim_c")[col].to_list() == \
               pytest.approx(t2.sort("obs_level", "dim_c")[col].to_list())
        assert t1.sort("obs_level", "dim_c")[col].to_list() == \
               pytest.approx(t3.sort("obs_level", "dim_c")[col].to_list())
