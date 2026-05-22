#!/usr/bin/env python3
"""
Demo: new chunk-based processor interface.

Standalone — does not use pp's processing pipeline.
Intended for a collaborator to understand the new interface before updating the pipeline.

Two-level chunking
──────────────────
MEMORY_CHUNK_SPEC  coarse tiling; each chunk is one unit of distributed work
                   (a dask task / worker in production).
LEAF_CHUNK_SPEC    fine tiling applied inside each worker on the materialised
                   memory chunk; leaf processors run once per leaf.

ChunkSpec  {"dim_letter": size, ...}  —  -1 or omitted means full extent.
"""

from __future__ import annotations

import tempfile
from itertools import combinations, product as iter_product
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import polars as pl
import tifffile

from pixel_patrol_base.core.record import record_from, Record
from pixel_patrol_base.plugins.processors.thumbnail_processor import ThumbnailProcessor
from pixel_patrol_image.plugins.processors.raster_image_processor import (
    CompressionMetricsProcessor,
    QualityMetricsProcessor,
)
from pixel_patrol_base.plugins.processors.raster_processor import (
    BasicMetricsProcessor,
    HistogramProcessor as RasterHistogramProcessor,
)
from pixel_patrol_loader_bio.plugins.loaders.tifffile_loader import TifffileLoader

ChunkSpec = Dict[str, int]


def _size_col(dim_key: str) -> str:
    """'dim_y' → 'Y_size'"""
    return f"{dim_key.removeprefix('dim_').upper()}_size"


# ──────────────────────────────────────────────────────────────────────────────
# TIFF creation and loading
# ──────────────────────────────────────────────────────────────────────────────

def create_synthetic_tiff(path: Path) -> None:
    rng = np.random.default_rng(0)
    data = rng.integers(0, 65535, size=(2, 4, 3, 64, 64), dtype=np.uint16)
    tifffile.imwrite(path, data, imagej=True)
    print(f"created  {path}  shape={data.shape}  dtype={data.dtype}")


def load_record(path: Path) -> Record:
    result = TifffileLoader().load(str(path))
    record = next(iter(result.values())) if isinstance(result, dict) else result
    print(f"loaded   dim_order={record.dim_order}  shape={tuple(record.data.shape)}"
          f"  capabilities={record.capabilities}")
    return record


# ──────────────────────────────────────────────────────────────────────────────
# Chunking
# ──────────────────────────────────────────────────────────────────────────────

def plan_chunks(
    shape: tuple,
    dim_order: str,
    chunk_spec: ChunkSpec,
) -> List[Tuple[tuple, List[int]]]:
    """
    Return (bounds, origin) pairs that cover *shape* according to *chunk_spec*.

    bounds  — tuple of slice objects, used to index the array
    origin  — starting coordinate of this chunk in the parent array's space
    """
    dims = [
        (int(shape[i]), chunk_spec.get(dim, int(shape[i]))
         if chunk_spec.get(dim, -1) >= 0 else int(shape[i]))
        for i, dim in enumerate(dim_order)
    ]
    chunks = []
    for starts in iter_product(*[range(0, ext, tile) for ext, tile in dims]):
        bounds = tuple(
            slice(start, min(start + tile, ext))
            for start, (ext, tile) in zip(starts, dims)
        )
        chunks.append((bounds, list(starts)))
    return chunks


# ──────────────────────────────────────────────────────────────────────────────
# Row annotation  (pipeline concern, not processor concern)
# ──────────────────────────────────────────────────────────────────────────────

def _annotate(row: Dict, record: Record) -> Dict:
    """Add position (dim_*) and tile-size (*_size, num_pixels) to a metric row."""
    arr = record.data
    dim_order = list(record.dim_order)
    row.update({f"dim_{d.lower()}": record.meta.get(f"dim_{d.lower()}", 0) for d in dim_order})
    row.update({
        "num_pixels": int(np.prod(arr.shape)),
        **{f"{d}_size": arr.shape[i] for i, d in enumerate(dim_order)},
    })
    return row


# ──────────────────────────────────────────────────────────────────────────────
# Worker task
# ──────────────────────────────────────────────────────────────────────────────

def worker_task(
    record: Record,
    mem_bounds: tuple,
    mem_origin: List[int],
    memory_procs: List,
    leaf_procs: List,
    leaf_chunk_spec: ChunkSpec,
) -> Tuple[Dict[str, List[Dict]], Dict[str, List[Dict]]]:
    """
    Process one memory chunk. In production this runs on a distributed worker.

    1. Materialise the memory chunk and create its Record.
    2. Run MEMORY processors on the whole chunk.
    3. Subdivide into leaf chunks; run LEAF processors on each.

    Origins are global (relative to the full record) so rows from different
    workers can be aggregated together.
    """
    mem_arr: np.ndarray = record.data[mem_bounds].compute()
    mem_record = record_from(
        mem_arr,
        {**record.meta, "shape": list(mem_arr.shape), "ndim": mem_arr.ndim,
         "dim_order": record.dim_order,
         **{f"dim_{d.lower()}": mem_origin[i] for i, d in enumerate(record.dim_order)}},
        kind=record.kind,
    )

    mem_rows: Dict[str, List[Dict]] = {
        proc.NAME: [_annotate(proc.run_chunk(mem_record), mem_record)]
        for proc in memory_procs
    }

    leaf_chunks = plan_chunks(mem_arr.shape, record.dim_order, leaf_chunk_spec)
    leaf_rows: Dict[str, List[Dict]] = {p.NAME: [] for p in leaf_procs}

    for bounds, local_origin in leaf_chunks:
        leaf_arr: np.ndarray = mem_arr[bounds]
        global_origin = [mo + lo for mo, lo in zip(mem_origin, local_origin)]

        leaf_record = record_from(
            leaf_arr,
            {**mem_record.meta, "shape": list(leaf_arr.shape), "ndim": leaf_arr.ndim,
             **{f"dim_{d.lower()}": global_origin[i] for i, d in enumerate(record.dim_order)}},
            kind=record.kind,
        )

        for proc in leaf_procs:
            try:
                row = proc.run_chunk(leaf_record)
                if row:
                    leaf_rows[proc.NAME].append(_annotate(row, leaf_record))
            except Exception as exc:
                print(f"    [{proc.NAME}] origin={global_origin}: {exc}")

    return mem_rows, leaf_rows


# ──────────────────────────────────────────────────────────────────────────────
# Result table  (leaf resolution; memory chunks invisible)
# ──────────────────────────────────────────────────────────────────────────────

def _merge_by_leaf_position(
    leaf_procs: List,
    all_leaf_rows: Dict[str, List[Dict]],
    leaf_pos_keys: List[str],
) -> List[Dict]:
    """
    Merge per-processor rows into one row per leaf position.

    Rows at the same leaf position from different memory chunks (e.g. Z slices
    that are not in the leaf spec) are aggregated here so the output has exactly
    one merged row per unique (dim_c, dim_y, …) leaf coordinate.
    """
    by_pos: Dict[tuple, Dict[str, List[Dict]]] = {}
    for proc in leaf_procs:
        for row in all_leaf_rows.get(proc.NAME, []):
            pos = tuple(row.get(k, 0) for k in leaf_pos_keys)
            by_pos.setdefault(pos, {}).setdefault(proc.NAME, []).append(row)

    leaf_rows: List[Dict] = []
    for pos in sorted(by_pos):
        merged: Dict[str, Any] = {}
        for proc in leaf_procs:
            rows = by_pos[pos].get(proc.NAME, [])
            if not rows:
                continue
            if len(rows) == 1:
                merged.update(rows[0])
            else:
                merged.update({k: v for k, v in rows[0].items()
                                if k.startswith("dim_") or k.endswith("_size")})
                merged["num_pixels"] = sum(r.get("num_pixels", 0) for r in rows)
                for col in proc.OUTPUT_SCHEMA:
                    if (fn := proc.get_aggregation(col)) is not None:
                        merged[col] = fn(rows, [])
        leaf_rows.append(merged)
    return leaf_rows


def _build_aggregation_tree(
    leaf_rows: List[Dict],
    leaf_procs: List,
    dim_names: List[str],
    tile_sizes: Dict[str, Dict[int, int]],
    full_size: Dict[str, int],
    memory_procs: Optional[List],
    all_mem_rows: Optional[Dict[str, List[Dict]]],
    enable_leaf_rows: bool,
) -> List[Dict]:
    """
    Produce every useful grouping of the leaf dimensions.

      obs_level = n  →  one row per leaf tile (all dimensions fixed)
      obs_level = k  →  aggregated over (n - k) dimensions
      obs_level = 0  →  one global row; memory-processor results added here
    """
    n = len(dim_names)
    metric_cols = {col for proc in leaf_procs for col in proc.OUTPUT_SCHEMA}
    size_cols   = {_size_col(dk) for dk in dim_names}

    def _agg(rows: List[Dict], g_dims: List[str]) -> Dict[str, Any]:
        result = {}
        for proc in leaf_procs:
            for col in proc.OUTPUT_SCHEMA:
                if (fn := proc.get_aggregation(col)) is not None:
                    if (val := fn(rows, g_dims)) is not None:
                        result[col] = val
        return result

    def _tile_size(dk: str, val) -> int:
        return full_size[dk] if val is None else tile_sizes.get(dk, {}).get(val, 1)

    all_rows: List[Dict] = []

    if enable_leaf_rows:
        for row in leaf_rows:
            out = {k: v for k, v in row.items()
                   if k in metric_cols or k in set(dim_names)
                   or k in size_cols or k == "num_pixels"}
            out["obs_level"] = n
            all_rows.append(out)

    for depth in reversed(range(n)):
        for g_dims in combinations(dim_names, depth):
            groups: Dict[tuple, List[Dict]] = {}
            for row in leaf_rows:
                groups.setdefault(tuple(row.get(d) for d in g_dims), []).append(row)

            for key, group_rows in groups.items():
                out: Dict[str, Any] = {"obs_level": depth}
                for i, dk in enumerate(g_dims):
                    out[dk] = key[i]
                for dk in dim_names:
                    out[dk] = out.get(dk)  # None if not in g_dims
                for dk in dim_names:
                    out[_size_col(dk)] = _tile_size(dk, out[dk])
                out["num_pixels"] = sum(r.get("num_pixels", 0) for r in group_rows)
                out.update(_agg(group_rows, list(g_dims)))

                if depth == 0 and memory_procs and all_mem_rows:
                    for proc in memory_procs:
                        rows = all_mem_rows.get(proc.NAME, [])
                        if rows:
                            for col in proc.OUTPUT_SCHEMA:
                                if (fn := proc.get_aggregation(col)) is not None:
                                    out[col] = fn(rows, [])

                all_rows.append(out)

    return all_rows


def build_result_table(
    leaf_procs: List,
    all_leaf_rows: Dict[str, List[Dict]],
    dim_order: str,
    full_shape: tuple,
    leaf_chunk_spec: ChunkSpec,
    memory_procs: Optional[List] = None,
    all_mem_rows: Optional[Dict[str, List[Dict]]] = None,
    enable_leaf_rows: bool = True,
) -> pl.DataFrame:
    """
    Build the full result table.

      obs_level = n  →  one row per leaf tile
      obs_level = k  →  aggregated over (n - k) leaf dimensions
      obs_level = 0  →  one global row (includes memory-processor results)

    Memory-chunk dimensions are invisible: always aggregated over, never appear
    as columns. Sizes for fixed dimensions reflect actual tile sizes.
    """
    leaf_spec_dims = {f"dim_{d.lower()}" for d in leaf_chunk_spec}
    leaf_pos_keys  = [f"dim_{d.lower()}" for d in dim_order
                      if f"dim_{d.lower()}" in leaf_spec_dims]

    leaf_rows = _merge_by_leaf_position(leaf_procs, all_leaf_rows, leaf_pos_keys)

    dim_names = [
        f"dim_{d.lower()}" for d in dim_order
        if f"dim_{d.lower()}" in leaf_spec_dims
        and len({r.get(f"dim_{d.lower()}", 0) for r in leaf_rows}) > 1
    ]
    tile_sizes: Dict[str, Dict[int, int]] = {}
    for row in leaf_rows:
        for dk in dim_names:
            if (pos := row.get(dk)) is not None:
                if (sz := row.get(_size_col(dk))) is not None:
                    tile_sizes.setdefault(dk, {})[pos] = sz
    full_size = {
        f"dim_{d.lower()}": full_shape[list(dim_order).index(d)]
        for d in dim_order if f"dim_{d.lower()}" in set(dim_names)
    }

    all_rows = _build_aggregation_tree(
        leaf_rows, leaf_procs, dim_names, tile_sizes, full_size,
        memory_procs, all_mem_rows, enable_leaf_rows,
    )

    if not all_rows:
        return pl.DataFrame()
    all_keys = list(dict.fromkeys(k for r in all_rows for k in r))
    cols: Dict[str, List] = {}
    for key in all_keys:
        vals = [r.get(key) for r in all_rows]
        sample = next((v for v in vals if v is not None), None)
        cols[key] = [v.tolist() if isinstance(v, np.ndarray) else v for v in vals] \
                    if isinstance(sample, np.ndarray) else vals
    return pl.DataFrame({k: pl.Series(k, v, strict=False) for k, v in cols.items()})


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    MEMORY_CHUNK_SPEC: ChunkSpec = {"X": 50, "Y": 50, "Z": 1}
    LEAF_CHUNK_SPEC:   ChunkSpec = {"X": 25, "Y": 25, "C": 1}
    # LEAF_CHUNK_SPEC:   ChunkSpec = {"C": 1}

    memory_procs = [ThumbnailProcessor()]
    leaf_procs   = [
        BasicMetricsProcessor(),
        RasterHistogramProcessor(),
        QualityMetricsProcessor(),
        CompressionMetricsProcessor(),
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        tiff_path = Path(tmpdir) / "synthetic_5d.tif"

        print("\n── create TIFF ───────────────────────────────────────────────────")
        create_synthetic_tiff(tiff_path)

        print("\n── load → Record ─────────────────────────────────────────────────")
        record = load_record(tiff_path)

        print("\n── iterate memory chunks ─────────────────────────────────────────")
        mem_chunks = plan_chunks(record.data.shape, record.dim_order, MEMORY_CHUNK_SPEC)
        print(f"  {len(mem_chunks)} memory chunks  spec={MEMORY_CHUNK_SPEC}")

        all_mem_rows:  Dict[str, List[Dict]] = {p.NAME: [] for p in memory_procs}
        all_leaf_rows: Dict[str, List[Dict]] = {p.NAME: [] for p in leaf_procs}

        for i, (bounds, origin) in enumerate(mem_chunks):
            print(f"\n  worker task {i}  origin={origin}")
            mem_rows, leaf_rows = worker_task(
                record, bounds, origin,
                memory_procs, leaf_procs, LEAF_CHUNK_SPEC,
            )
            print(f"    {sum(len(v) for v in leaf_rows.values())} leaf rows")
            for proc in memory_procs:
                all_mem_rows[proc.NAME].extend(mem_rows.get(proc.NAME, []))
            for proc in leaf_procs:
                all_leaf_rows[proc.NAME].extend(leaf_rows.get(proc.NAME, []))

        print("\n── result table  (full aggregation tree, memory chunks invisible) ─")
        table = build_result_table(
            leaf_procs, all_leaf_rows, record.dim_order,
            tuple(record.data.shape), LEAF_CHUNK_SPEC,
            memory_procs=memory_procs, all_mem_rows=all_mem_rows,
        )
        print(table)
        print(f"\n  {table.shape[0]} rows × {table.shape[1]} columns")

        out_path = Path(__file__).with_suffix(".parquet")
        table.write_parquet(out_path)
        print(f"\n  written to {out_path}")


if __name__ == "__main__":
    main()
