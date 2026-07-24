"""End-to-end integration tests for the processing pipeline on real (synthetic) images.

Unlike the unit tests elsewhere (test_rollup.py, test_run_record.py, etc.), these run the
full pipeline - build_records_df / Project.process_records - through a real loader
(TifffileLoader) and real processors, and check the actual output at every stage: the
in-memory df, the real part files written to disk, the final merged parquet, and its
footer metadata. Oracle values are computed independently with plain numpy, not by
importing the pipeline's own aggregation code, so these are genuine correctness checks
rather than "did it run" smoke tests.

Image dimensions are deliberately awkward (not evenly divisible by chunk counts) so
chunk/group boundaries are ragged - this is what actually exercises boundary bugs;
neatly-divisible sizes would hide them. Images are kept tiny (KB, not MB) - raggedness
is a property of the dimension arithmetic, not of the data volume, so a small array
triggers the same boundary bugs as a large one for a fraction of the runtime.

All tests share one module-scoped 2-worker Dask client (see `shared_client`) instead of
each spinning up its own cluster: real per-test clusters size `memory_limit` off
`mb_per_task` (`memory_limit = mb_per_task * 8`), so a tiny `mb_per_task` - needed to make
tiny images trigger chunking - starves the worker process below its own baseline RAM
footprint and gets it killed by Dask's memory monitor. Reusing one client with a real
`memory_limit` sidesteps that, and 2 workers means these tests also exercise cross-worker
behaviour that a max_workers=1 cluster never would.

Reusing a client across calls has its own known flakiness: build_records_df() scatters
the loader/processors fresh on every call and doesn't wait for the previous call's
scattered keys to finish being released, so back-to-back calls can occasionally race
("Key lost during replication" / "lost dependencies") and a task gets spuriously
cancelled. `_build` and `_process` below retry on that specific failure signature so the
suite isn't flaky; the race itself is a separate, real latent issue in the reused-client
path (tracked separately), not something these tests are responsible for fixing.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import polars as pl
import pytest
import tifffile
from distributed import Client, LocalCluster

from pixel_patrol_base.config import HISTOGRAM_BINS
from pixel_patrol_base.core.processing import build_records_df, save_parquet_from_parts
from pixel_patrol_base.core.processing_config import ProcessingConfig
from pixel_patrol_base.core.project import Project
from pixel_patrol_base.plugins.processors.raster_processor import BasicMetricsProcessor, HistogramProcessor
from pixel_patrol_loader_bio.plugins.loaders.tifffile_loader import TifffileLoader
from pixel_patrol_base.plugins.processors.thumbnail_processor import ThumbnailProcessor

PROCESSORS = [BasicMetricsProcessor(), HistogramProcessor(), ThumbnailProcessor()]

_RETRIES = 3


@pytest.fixture(scope="module")
def shared_client():
    """One real cluster for the whole module - see module docstring for why."""
    cluster = LocalCluster(n_workers=2, threads_per_worker=1, processes=True, memory_limit="512MB")
    client = Client(cluster)
    yield client
    client.close()
    cluster.close()


def _build(dirs, config, base_dir, **kwargs):
    """build_records_df, retrying past the reused-client scatter race (see module docstring)."""
    df, stats = None, {"n_images_failed": 1}
    for _ in range(_RETRIES):
        df, stats = build_records_df(dirs, loader=TifffileLoader(), processors=PROCESSORS,
                                      config=config, base_dir=base_dir, **kwargs)
        if stats.get("n_images_failed", 0) == 0:
            return df, stats
    return df, stats


def _process(proj: Project, config: ProcessingConfig) -> None:
    """proj.process_records, retrying past the reused-client scatter race (see module docstring)."""
    for attempt in range(_RETRIES):
        proj.process_records(config)
        if pl.read_parquet(proj.output_path).height > 0:
            return
    raise AssertionError("process_records produced an empty result after retries")


# ── Oracles: independent of pipeline aggregation code ───────────────────────

def _oracle_scalars(arr: np.ndarray) -> dict:
    a = arr.astype(np.float64)
    return {
        "mean_intensity": float(a.mean()),
        "std_intensity": float(a.std()),
        "min_intensity": float(a.min()),
        "max_intensity": float(a.max()),
        "finite_pixel_count": int(arr.size),
    }


def _oracle_histogram(arr: np.ndarray, s_min: float, s_max: float) -> np.ndarray:
    """Independently reproduces the floor-bin formula documented in
    raster_processor._histogram_counts, without importing it."""
    B = HISTOGRAM_BINS
    flat = arr.ravel().astype(np.float32)
    h = np.zeros(B, dtype=np.int64)
    bins = np.clip(np.floor((flat - s_min) / ((s_max - s_min) / B)).astype(np.int32), 0, B - 1)
    np.add.at(h, bins, 1)
    return h


def _assert_scalar_row_matches_oracle(row: dict, arr: np.ndarray) -> None:
    oracle = _oracle_scalars(arr)
    assert row["mean_intensity"] == pytest.approx(oracle["mean_intensity"], rel=1e-5)
    assert row["std_intensity"] == pytest.approx(oracle["std_intensity"], rel=1e-5)
    assert row["min_intensity"] == oracle["min_intensity"]
    assert row["max_intensity"] == oracle["max_intensity"]
    assert row["finite_pixel_count"] == oracle["finite_pixel_count"]
    oracle_hist = _oracle_histogram(arr, oracle["min_intensity"], oracle["max_intensity"])
    pipeline_hist = np.asarray(row["histogram_counts"])
    assert pipeline_hist.sum() == oracle["finite_pixel_count"]
    np.testing.assert_array_equal(pipeline_hist, oracle_hist)


# ── Test image builders ──────────────────────────────────────────────────────

def _write_small_zcyx(tmp_path: Path) -> Tuple[Path, np.ndarray]:
    """Z=3, C=3, Y=37, X=53 (~34.5KB) - small, fast, awkward Z (not divisible by 2)."""
    rng = np.random.default_rng(42)
    Z, C, Y, X = 3, 3, 37, 53
    arr = np.zeros((Z, C, Y, X), dtype=np.uint16)
    for c in range(C):
        arr[:, c] = (c + 1) * 50 + rng.integers(0, 10, size=(Z, Y, X))
    path = tmp_path / "small.tif"
    tifffile.imwrite(path, arr, metadata={"axes": "ZCYX"})
    return path, arr


def _write_medium_zcyx(tmp_path: Path, name: str = "medium.tif") -> Tuple[Path, np.ndarray]:
    """Z=7, C=3, Y=50, X=50 (~102.5KB) - at mb_per_task=0.05 still forces ragged
    Z-group memory chunking (7 -> 3+3+1); size is irrelevant to the raggedness,
    only the ratio of array size to budget is."""
    rng = np.random.default_rng(11)
    Z, C, Y, X = 7, 3, 50, 50
    arr = np.zeros((Z, C, Y, X), dtype=np.uint16)
    for c in range(C):
        arr[:, c] = (c + 1) * 50 + rng.integers(0, 10, size=(Z, Y, X))
    path = tmp_path / name
    tifffile.imwrite(path, arr, metadata={"axes": "ZCYX"})
    return path, arr


def _write_large_2d(tmp_path: Path) -> Tuple[Path, np.ndarray]:
    """Y=97, X=101 (~19.1KB), plain 2D - forces spatial (X/Y) memory-chunk splitting
    directly (nothing else to split first), with a ragged chunk boundary in X
    (101 -> 50+50+1) at mb_per_task=0.01.

    Value range is deliberately narrow (0-9): the histogram merge takes an exact
    fast path only when every chunk's local min/max exactly match the global
    min/max (raster_processor._aggregate_histograms), and the ragged remainder
    chunk here has only 97 samples - too few to reliably hit both extremes of a
    wide range (e.g. 0-255) by chance, which would fall into a lossy approximate
    remap path instead and break the byte-exact reconciliation this test checks.
    """
    rng = np.random.default_rng(7)
    Y, X = 97, 101
    arr = rng.integers(0, 10, size=(Y, X)).astype(np.uint16)
    path = tmp_path / "spatial.tif"
    tifffile.imwrite(path, arr, metadata={"axes": "YX"})
    return path, arr


# ── 1. Baseline: unchunked, default per-Z/per-C leaf granularity ────────────

def test_baseline_unchunked_matches_oracle_and_default_granularity(shared_client, tmp_path: Path):
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    _, arr = _write_small_zcyx(img_dir)

    config = ProcessingConfig(max_workers=1, mb_per_task=100)
    df, stats = _build([img_dir], config, img_dir)

    assert stats["n_images_processed"] == 1
    assert stats["task_types"] == {"BatchTask": 1}

    # Default leaf granularity (block size 1 for non-X/Y dims) produces per-Z, per-C,
    # and per-(Z,C) rows even with no chunking and no user slice_size at all.
    counts = dict(df.group_by("obs_level").len().iter_rows())
    assert counts == {0: 1, 1: 3 + 3, 2: 3 * 3}
    assert len(df) == 16

    global_row = df.filter(pl.col("obs_level") == 0).to_dicts()[0]
    assert (global_row["size_Z"], global_row["size_C"], global_row["size_Y"], global_row["size_X"]) == (3, 3, 37, 53)
    _assert_scalar_row_matches_oracle(global_row, arr)
    assert len(global_row["thumbnail"]) > 0

    # Per-Z rows (obs_level 1, dim_z set, dim_c null) reconcile against the oracle for that Z-slice.
    per_z = df.filter((pl.col("obs_level") == 1) & pl.col("dim_z").is_not_null())
    assert per_z.height == 3
    for row in per_z.to_dicts():
        z = row["dim_z"]
        _assert_scalar_row_matches_oracle(row, arr[z])

    # Per-C rows likewise.
    per_c = df.filter((pl.col("obs_level") == 1) & pl.col("dim_c").is_not_null())
    assert per_c.height == 3
    for row in per_c.to_dicts():
        c = row["dim_c"]
        _assert_scalar_row_matches_oracle(row, arr[:, c])

    # Per-(Z,C) rows (obs_level 2) reconcile against the oracle for that single leaf.
    per_zc = df.filter(pl.col("obs_level") == 2)
    assert per_zc.height == 9
    for row in per_zc.to_dicts():
        z, c = row["dim_z"], row["dim_c"]
        _assert_scalar_row_matches_oracle(row, arr[z, c])


# ── 2. Forced ragged memory-chunking along Z reconciles with the unchunked result ──

def test_ragged_z_memory_chunking_matches_unchunked(shared_client, tmp_path: Path):
    unchunked_dir = tmp_path / "unchunked"
    unchunked_dir.mkdir()
    _, arr = _write_medium_zcyx(unchunked_dir)
    chunked_dir = tmp_path / "chunked"
    chunked_dir.mkdir()
    tifffile.imwrite(chunked_dir / "medium.tif", arr, metadata={"axes": "ZCYX"})

    unchunked_cfg = ProcessingConfig(max_workers=1, mb_per_task=1)
    df_unchunked, stats_unchunked = _build([unchunked_dir], unchunked_cfg, unchunked_dir)
    assert stats_unchunked["task_types"] == {"BatchTask": 1}

    chunked_cfg = ProcessingConfig(max_workers=1, mb_per_task=0.05)
    df_chunked, stats_chunked = _build([chunked_dir], chunked_cfg, chunked_dir)
    # 3 ragged memory chunks: Z split into groups of 3 (7 -> 3+3+1), C untouched.
    assert stats_chunked["task_types"] == {"MemoryChunkTask": 3}

    g_unchunked = df_unchunked.filter(pl.col("obs_level") == 0).to_dicts()[0]
    g_chunked = df_chunked.filter(pl.col("obs_level") == 0).to_dicts()[0]

    _assert_scalar_row_matches_oracle(g_unchunked, arr)
    _assert_scalar_row_matches_oracle(g_chunked, arr)

    # Memory-chunk splitting is internal plumbing: same row count/shape either way.
    assert len(df_chunked) == len(df_unchunked) == 32
    # The assembled thumbnail must be byte-identical regardless of memory chunking -
    # it's a deterministic assembly, so exact equality (not "roughly") is the right bar.
    assert g_chunked["thumbnail"] == g_unchunked["thumbnail"]
    assert g_chunked["thumbnail_norm_min"] == g_unchunked["thumbnail_norm_min"]
    assert g_chunked["thumbnail_norm_max"] == g_unchunked["thumbnail_norm_max"]


# ── 3. Forced ragged spatial (X/Y) memory-chunking reconciles + thumbnail near-matches ──

def test_ragged_spatial_memory_chunking_matches_unchunked(shared_client, tmp_path: Path):
    unchunked_dir = tmp_path / "unchunked"
    unchunked_dir.mkdir()
    _, arr = _write_large_2d(unchunked_dir)
    chunked_dir = tmp_path / "chunked"
    chunked_dir.mkdir()
    tifffile.imwrite(chunked_dir / "spatial.tif", arr, metadata={"axes": "YX"})

    unchunked_cfg = ProcessingConfig(max_workers=1, mb_per_task=1)
    df_unchunked, stats_unchunked = _build([unchunked_dir], unchunked_cfg, unchunked_dir)
    assert stats_unchunked["task_types"] == {"BatchTask": 1}

    chunked_cfg = ProcessingConfig(max_workers=1, mb_per_task=0.01)
    df_chunked, stats_chunked = _build([chunked_dir], chunked_cfg, chunked_dir)
    # Ragged spatial split: X=101 -> 50+50+1, Y untouched.
    assert stats_chunked["task_types"] == {"MemoryChunkTask": 3}

    g_unchunked = df_unchunked.filter(pl.col("obs_level") == 0).to_dicts()[0]
    g_chunked = df_chunked.filter(pl.col("obs_level") == 0).to_dicts()[0]

    _assert_scalar_row_matches_oracle(g_unchunked, arr)
    _assert_scalar_row_matches_oracle(g_chunked, arr)
    # Near-match, not byte-exact: per-chunk nearest-neighbor downsampling picks slightly
    # different source pixels than a whole-image downsample at chunk boundaries.
    t_chunked = np.frombuffer(g_chunked["thumbnail"], dtype=np.uint8)
    t_unchunked = np.frombuffer(g_unchunked["thumbnail"], dtype=np.uint8)
    assert np.mean(t_chunked != t_unchunked) < 0.2


# ── 4. Custom slice_size overrides the default leaf granularity, still reconciles ──

def test_custom_slice_size_z_block_2_reconciles_with_global(shared_client, tmp_path: Path):
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    _, arr = _write_small_zcyx(img_dir)

    config = ProcessingConfig(max_workers=1, mb_per_task=100, slice_size={"Z": 2, "C": -1})
    df, stats = _build([img_dir], config, img_dir)

    # Z=3 in blocks of 2 -> 2 ragged groups (2,1); C pinned to -1 (never split).
    per_z = df.filter((pl.col("obs_level") == 1) & pl.col("dim_z").is_not_null()).sort("dim_z")
    assert per_z["dim_z"].to_list() == [0, 2]
    assert per_z.height == 2

    global_row = df.filter(pl.col("obs_level") == 0).to_dicts()[0]
    _assert_scalar_row_matches_oracle(global_row, arr)

    rows = per_z.to_dicts()
    z_bounds = [(0, 2), (2, 3)]  # last group ragged: only Z=2
    total_pixels = 0
    weighted_mean_num = 0.0
    for row, (z0, z1) in zip(rows, z_bounds):
        group_arr = arr[z0:z1]
        _assert_scalar_row_matches_oracle(row, group_arr)
        total_pixels += row["num_pixels"]
        weighted_mean_num += row["mean_intensity"] * row["num_pixels"]

    assert total_pixels == global_row["num_pixels"]
    assert (weighted_mean_num / total_pixels) == pytest.approx(global_row["mean_intensity"], rel=1e-5)


# ── 5. Mixed dataset: batch + container + memory-chunked in one run ─────────

def test_mixed_batch_container_and_chunked_dataset(shared_client, tmp_path: Path):
    img_dir = tmp_path / "images"
    img_dir.mkdir()

    # 2 tiny plain single-series files -> stay batched.
    rng = np.random.default_rng(1)
    for name in ("a.tif", "b.tif"):
        tifffile.imwrite(img_dir / name, rng.integers(0, 255, (16, 16), dtype=np.uint8))

    # 1 multi-series file with 3 sub-images -> container.
    with tifffile.TiffWriter(img_dir / "container.tif") as tw:
        for i in range(3):
            tw.write(rng.integers(0, 255, (16, 16), dtype=np.uint8), metadata={"axes": "YX"})

    # 1 medium file -> forced into memory chunks by the small mb_per_task below.
    _, medium_arr = _write_medium_zcyx(img_dir, name="medium.tif")

    config = ProcessingConfig(max_workers=1, mb_per_task=0.05)
    df, stats = _build([img_dir], config, img_dir)

    assert stats["n_images_processed"] == 2 + 3 + 1
    assert stats["task_types"] == {"BatchTask": 1, "ContainerTask": 1, "MemoryChunkTask": 3}

    global_rows = df.filter(pl.col("obs_level") == 0)
    assert global_rows.height == 6

    container_rows = global_rows.filter(pl.col("child_id").is_not_null())
    assert container_rows.height == 3
    assert set(container_rows["type"].to_list()) == {"sub_file"}

    non_container_rows = global_rows.filter(pl.col("child_id").is_null())
    assert non_container_rows.height == 3
    assert set(non_container_rows["type"].to_list()) == {"file"}

    medium_row = global_rows.filter(pl.col("path").str.ends_with("medium.tif")).to_dicts()[0]
    _assert_scalar_row_matches_oracle(medium_row, medium_arr)


# ── 6. processors_included/excluded wired end-to-end through Project ────────

def test_processor_exclusion_wired_end_to_end(shared_client, tmp_path: Path):
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    _, arr = _write_small_zcyx(img_dir)

    proj = Project("test", img_dir, loader="tifffile", output_path=tmp_path / "out.parquet")
    config = ProcessingConfig(
        max_workers=1, mb_per_task=100,
        processors_included={"raster-basic", "raster-histogram", "thumbnail"},
    )
    _process(proj, config)

    result = pl.read_parquet(proj.output_path)
    assert "mean_intensity" in result.columns
    assert "thumbnail" in result.columns

    config_excl = ProcessingConfig(
        max_workers=1, mb_per_task=100,
        processors_included={"raster-basic", "raster-histogram"},  # thumbnail dropped
    )
    proj2 = Project("test2", img_dir, loader="tifffile", output_path=tmp_path / "out2.parquet")
    _process(proj2, config_excl)

    result2 = pl.read_parquet(proj2.output_path)
    assert "mean_intensity" in result2.columns
    assert "thumbnail" not in result2.columns


# ── 7. Real parts on disk, real merge, real footer metadata ─────────────────

def test_real_parts_written_and_merged_with_correct_metadata(shared_client, tmp_path: Path):
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    rng = np.random.default_rng(3)
    arrays = {}
    for i, name in enumerate(("img0.tif", "img1.tif", "img2.tif")):
        arr = rng.integers(0, 255, (8, 8)).astype(np.uint8) + i * 10  # distinguishable per image
        tifffile.imwrite(img_dir / name, arr)
        arrays[name] = arr

    parts_dir = tmp_path / "_parts"
    config = ProcessingConfig(max_workers=1, mb_per_task=100, rows_per_part=1)
    proj = Project("test", img_dir, loader="tifffile", output_path=tmp_path / "out.parquet")
    proj.metadata = ProcessingConfig().metadata.populate_from_project(proj)

    df, stats = _build([img_dir], config, img_dir, parts_dir=parts_dir)

    # rows_per_part=1 with one row per (unchunked, single-series) image -> one part per image.
    part_files = sorted(parts_dir.glob("part_*.parquet"))
    assert len(part_files) == 3

    seen_paths = set()
    for part_path in part_files:
        part_df = pl.read_parquet(part_path)
        assert len(part_df) == 1
        row = part_df.to_dicts()[0]
        name = Path(row["path"]).name
        seen_paths.add(name)
        _assert_scalar_row_matches_oracle(row, arrays[name])
    assert seen_paths == set(arrays)

    dest = tmp_path / "merged.parquet"
    n_rows = save_parquet_from_parts(part_files, dest, proj.metadata)
    assert n_rows == 3

    merged = pl.read_parquet(dest)
    assert len(merged) == 3
    for row in merged.to_dicts():
        name = Path(row["path"]).name
        _assert_scalar_row_matches_oracle(row, arrays[name])

    # Footer metadata: loader name round-trips, and columns carry descriptions.
    import pyarrow.parquet as pq
    schema = pq.read_schema(dest)
    kv_meta = {k.decode(): v.decode() for k, v in (schema.metadata or {}).items()}
    assert kv_meta.get("pp_loader") == "tifffile"

    mean_field = schema.field("mean_intensity")
    field_meta = mean_field.metadata or {}
    assert b"description" in field_meta
