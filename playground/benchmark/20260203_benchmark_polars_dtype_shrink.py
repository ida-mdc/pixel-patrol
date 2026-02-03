#!/usr/bin/env python3
"""
STANDALONE Benchmark: Shrink dtypes per-batch vs at-the-end

Usage:
    python dtype_shrink_benchmark.py --demo   # Quick demo
    python dtype_shrink_benchmark.py --quick  # Quick benchmark
    python dtype_shrink_benchmark.py          # Full benchmark

Tests whether it's better to:
1. Shrink dtypes on each batch before merging
2. Shrink dtypes once on the final merged DataFrame
"""

import gc
import time
import statistics
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable

import polars as pl
import numpy as np


# =============================================================================
# Validate concat dtype behavior FIRST
# =============================================================================

def validate_concat_dtype_behavior() -> bool:
    """
    Verify that pl.concat correctly upcasts to safe supertypes.

    Critical: Int8 + Int16 → Int16 (not Int8, not Int64)
              Float32 + Float64 → Float64 (safe upcast)

    Returns True if behavior is correct for per-batch shrinking.
    """
    print("=" * 60)
    print("VALIDATING: concat dtype preservation")
    print("=" * 60)

    all_passed = True
    tests = [
        # (name, batches, expected_dtype)
        ("Int8 + Int16 → Int16",
         [pl.DataFrame({"a": pl.Series([1, 2], dtype=pl.Int8)}),
          pl.DataFrame({"a": pl.Series([1000, 2000], dtype=pl.Int16)})],
         pl.Int16),

        ("Int8 + Int32 → Int32",
         [pl.DataFrame({"a": pl.Series([1], dtype=pl.Int8)}),
          pl.DataFrame({"a": pl.Series([100000], dtype=pl.Int32)})],
         pl.Int32),

        ("UInt8 + UInt16 → UInt16",
         [pl.DataFrame({"a": pl.Series([1], dtype=pl.UInt8)}),
          pl.DataFrame({"a": pl.Series([1000], dtype=pl.UInt16)})],
         pl.UInt16),

        ("Float32 + Float32 → Float32",
         [pl.DataFrame({"a": pl.Series([1.0], dtype=pl.Float32)}),
          pl.DataFrame({"a": pl.Series([2.0], dtype=pl.Float32)})],
         pl.Float32),

        ("Float32 + Float64 → Float64",
         [pl.DataFrame({"a": pl.Series([1.0], dtype=pl.Float32)}),
          pl.DataFrame({"a": pl.Series([2.0], dtype=pl.Float64)})],
         pl.Float64),

        ("Int8 + Int8 + Int16 + Int8 → Int16",
         [pl.DataFrame({"a": pl.Series([1], dtype=pl.Int8)}),
          pl.DataFrame({"a": pl.Series([2], dtype=pl.Int8)}),
          pl.DataFrame({"a": pl.Series([1000], dtype=pl.Int16)}),
          pl.DataFrame({"a": pl.Series([3], dtype=pl.Int8)})],
         pl.Int16),
    ]

    for name, batches, expected in tests:
        result = pl.concat(batches, how="vertical_relaxed")
        actual = result["a"].dtype
        passed = actual == expected
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name} → got {actual}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("✓ ALL PASSED: concat correctly upcasts - shrink_per_batch is SAFE")
    else:
        print("✗ FAILED: concat behavior unexpected - use shrink_final only")
    print("=" * 60 + "\n")

    return all_passed


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class BenchmarkConfig:
    name: str
    num_rows: int
    num_int_cols: int
    num_float_cols: int
    num_string_cols: int
    num_list_cols: int
    batch_size: int
    null_percentage: float = 0.0
    num_iterations: int = 3

    @property
    def total_cols(self) -> int:
        return self.num_int_cols + self.num_float_cols + self.num_string_cols + self.num_list_cols

    @property
    def num_batches(self) -> int:
        return max(1, (self.num_rows + self.batch_size - 1) // self.batch_size)


@dataclass
class BenchmarkResult:
    config_name: str
    strategy: str
    mean_time: float
    std_time: float
    memory_before: int
    memory_after: int
    memory_saved_pct: float
    rows_per_second: float
    sample_dtypes: Dict[str, str] = field(default_factory=dict)


# =============================================================================
# Shrinking Function
# =============================================================================

FLOAT32_MAX = 3.4e38


def shrink_dataframe(df: pl.DataFrame, shrink_floats: bool = True) -> pl.DataFrame:
    """
    Shrink numeric dtypes to minimum safe size.
    Single pass, single with_columns call.
    """
    series_updates = []

    for col_name in df.columns:
        s = df[col_name]
        dtype = s.dtype

        # Integers: use Series.shrink_dtype()
        if dtype in (pl.Int64, pl.Int32, pl.Int16, pl.Int8,
                     pl.UInt64, pl.UInt32, pl.UInt16, pl.UInt8):
            shrunk = s.shrink_dtype()
            if shrunk.dtype != dtype:
                series_updates.append(shrunk.alias(col_name))

        # Floats: safe downcast to Float32
        elif dtype == pl.Float64 and shrink_floats:
            clean = s.drop_nulls().drop_nans()
            if clean.len() == 0 or (abs(clean.min()) < FLOAT32_MAX and abs(clean.max()) < FLOAT32_MAX):
                series_updates.append(s.cast(pl.Float32).alias(col_name))

    return df.with_columns(series_updates) if series_updates else df


# =============================================================================
# Data Generation
# =============================================================================

def generate_batch_dataframe(
    num_rows: int,
    num_int_cols: int,
    num_float_cols: int,
    num_string_cols: int,
    num_list_cols: int,
    null_percentage: float,
    seed: Optional[int] = None,
) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    data = {}

    # Int columns with varying ranges
    for i in range(num_int_cols):
        col_name = f"int_col_{i}"
        if i % 4 == 0:
            values = rng.integers(0, 100, size=num_rows)
        elif i % 4 == 1:
            values = rng.integers(0, 10000, size=num_rows)
        elif i % 4 == 2:
            values = rng.integers(0, 10_000_000, size=num_rows)
        else:
            values = rng.integers(0, 2**50, size=num_rows)

        series = pl.Series(col_name, values, dtype=pl.Int64)
        if null_percentage > 0:
            null_indices = np.where(rng.random(num_rows) < null_percentage)[0].tolist()
            if null_indices:
                series = series.scatter(null_indices, [None] * len(null_indices))
        data[col_name] = series

    # Float columns
    for i in range(num_float_cols):
        col_name = f"float_col_{i}"
        if i % 3 == 0:
            values = rng.random(num_rows)
        elif i % 3 == 1:
            values = rng.random(num_rows) * 255
        else:
            values = rng.random(num_rows) * 1e6

        series = pl.Series(col_name, values, dtype=pl.Float64)
        if null_percentage > 0:
            null_indices = np.where(rng.random(num_rows) < null_percentage)[0].tolist()
            if null_indices:
                series = series.scatter(null_indices, [None] * len(null_indices))
        data[col_name] = series

    # String columns
    pools = [[".png", ".jpg", ".tiff", ".bmp"], ["image", "video", "document"]]
    for i in range(num_string_cols):
        col_name = f"str_col_{i}"
        pool = pools[i % len(pools)]
        values = [pool[rng.integers(0, len(pool))] for _ in range(num_rows)]
        if null_percentage > 0:
            values = [None if rng.random() < null_percentage else v for v in values]
        data[col_name] = pl.Series(col_name, values, dtype=pl.String)

    # List columns
    for i in range(num_list_cols):
        col_name = f"list_col_{i}"
        list_len = 256 if i % 2 == 0 else 64
        values = [list(rng.integers(0, 10000, size=list_len)) for _ in range(num_rows)]
        if null_percentage > 0:
            values = [None if rng.random() < null_percentage else v for v in values]
        data[col_name] = pl.Series(col_name, values)

    return pl.DataFrame(data)


def generate_batches(config: BenchmarkConfig, seed: int = 42) -> List[pl.DataFrame]:
    batches = []
    rows_remaining = config.num_rows
    batch_seed = seed

    while rows_remaining > 0:
        batch_rows = min(config.batch_size, rows_remaining)
        batch = generate_batch_dataframe(
            num_rows=batch_rows,
            num_int_cols=config.num_int_cols,
            num_float_cols=config.num_float_cols,
            num_string_cols=config.num_string_cols,
            num_list_cols=config.num_list_cols,
            null_percentage=config.null_percentage,
            seed=batch_seed,
        )
        batches.append(batch)
        rows_remaining -= batch_rows
        batch_seed += 1

    return batches


# =============================================================================
# Strategies (only the two that matter)
# =============================================================================

def strategy_no_shrink(batches: List[pl.DataFrame]) -> pl.DataFrame:
    if len(batches) == 1:
        return batches[0]
    return pl.concat(batches, how="vertical_relaxed", rechunk=True)


def strategy_shrink_final(batches: List[pl.DataFrame]) -> pl.DataFrame:
    merged = pl.concat(batches, how="vertical_relaxed", rechunk=True) if len(batches) > 1 else batches[0]
    return shrink_dataframe(merged)


def strategy_shrink_per_batch(batches: List[pl.DataFrame]) -> pl.DataFrame:
    shrunk = [shrink_dataframe(b) for b in batches]
    if len(shrunk) == 1:
        return shrunk[0]
    return pl.concat(shrunk, how="vertical_relaxed", rechunk=True)


STRATEGIES = {
    "no_shrink": strategy_no_shrink,
    "shrink_final": strategy_shrink_final,
    "shrink_per_batch": strategy_shrink_per_batch,
}


# =============================================================================
# Benchmarking
# =============================================================================

def run_single_benchmark(
    config: BenchmarkConfig,
    strategy_name: str,
    strategy_fn: Callable,
    batches: List[pl.DataFrame],
) -> BenchmarkResult:
    times = []
    memory_before = sum(b.estimated_size() for b in batches)
    final_df = None

    for _ in range(config.num_iterations):
        batches_copy = [b.clone() for b in batches]
        gc.collect()

        start = time.perf_counter()
        result_df = strategy_fn(batches_copy)
        elapsed = time.perf_counter() - start

        times.append(elapsed)
        final_df = result_df

    memory_after = final_df.estimated_size()

    return BenchmarkResult(
        config_name=config.name,
        strategy=strategy_name,
        mean_time=statistics.mean(times),
        std_time=statistics.stdev(times) if len(times) > 1 else 0.0,
        memory_before=memory_before,
        memory_after=memory_after,
        memory_saved_pct=100.0 * (memory_before - memory_after) / memory_before,
        rows_per_second=config.num_rows / statistics.mean(times),
    )


def run_benchmark_suite(configs: List[BenchmarkConfig], verbose: bool = True) -> List[BenchmarkResult]:
    results = []

    for config in configs:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Config: {config.name}")
            print(f"  Rows: {config.num_rows:,}, Cols: {config.total_cols}")
            print(f"  Batches: {config.num_batches} x {config.batch_size:,} rows")
            print(f"{'='*60}")

        batches = generate_batches(config)

        for strategy_name, strategy_fn in STRATEGIES.items():
            if verbose:
                print(f"\n  {strategy_name}:")

            result = run_single_benchmark(config, strategy_name, strategy_fn, batches)
            results.append(result)

            if verbose:
                print(f"    Time: {result.mean_time:.4f}s ± {result.std_time:.4f}s")
                print(f"    Memory: {result.memory_before/1e6:.2f}MB → {result.memory_after/1e6:.2f}MB ({result.memory_saved_pct:.1f}% saved)")
                print(f"    Throughput: {result.rows_per_second:,.0f} rows/s")

        del batches
        gc.collect()

    return results


# =============================================================================
# Configurations
# =============================================================================

def get_quick_configs() -> List[BenchmarkConfig]:
    return [
        BenchmarkConfig("quick_10k", 10_000, 15, 20, 8, 2, 2_000),
        BenchmarkConfig("quick_100k", 100_000, 15, 20, 8, 2, 20_000),
    ]


def get_default_configs() -> List[BenchmarkConfig]:
    configs = []

    # Size scaling (fixed 50k batch)
    for name, rows in [("10k", 10_000), ("100k", 100_000), ("500k", 500_000),
                       ("1m", 1_000_000), ("5m", 5_000_000)]:
        configs.append(BenchmarkConfig(
            name=f"size_{name}",
            num_rows=rows,
            num_int_cols=20,
            num_float_cols=30,
            num_string_cols=10,
            num_list_cols=2,
            batch_size=50_000,
        ))

    # Batch size at 1M rows - just two extremes
    for batch_size in [10_000, 500_000]:
        num_batches = 1_000_000 // batch_size
        configs.append(BenchmarkConfig(
            name=f"1m_batch_{batch_size//1000}k_x{num_batches}",
            num_rows=1_000_000,
            num_int_cols=20,
            num_float_cols=30,
            num_string_cols=10,
            num_list_cols=2,
            batch_size=batch_size,
        ))

    # Column distribution at 1M
    configs.append(BenchmarkConfig("1m_int_heavy", 1_000_000, 50, 10, 5, 2, 50_000))
    configs.append(BenchmarkConfig("1m_float_heavy", 1_000_000, 10, 50, 5, 2, 50_000))

    # Your use case
    configs.append(BenchmarkConfig("pixel_patrol", 71_000, 25, 35, 15, 4, 10_000))

    return configs


# =============================================================================
# Results
# =============================================================================

def print_summary(results: List[BenchmarkResult]) -> None:
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    configs = list(dict.fromkeys(r.config_name for r in results))

    print(f"\n{'Config':<30} {'Strategy':<18} {'Time (s)':<12} {'Saved':<10} {'Winner'}")
    print("-" * 85)

    for config in configs:
        cr = [r for r in results if r.config_name == config]
        shrink_only = [r for r in cr if r.strategy != "no_shrink"]
        best = min(shrink_only, key=lambda r: r.mean_time)

        for r in cr:
            winner = "← BEST" if r == best else ("(baseline)" if r.strategy == "no_shrink" else "")
            print(f"{r.config_name:<30} {r.strategy:<18} {r.mean_time:<12.4f} {r.memory_saved_pct:<9.1f}% {winner}")
        print()


def print_recommendation(results: List[BenchmarkResult]) -> None:
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)

    # Per-config winner
    print("\n--- Per-config winners ---")
    configs = list(dict.fromkeys(r.config_name for r in results))
    final_wins = 0
    per_batch_wins = 0

    for config in configs:
        cr = [r for r in results if r.config_name == config]
        final = next(r for r in cr if r.strategy == "shrink_final")
        per_batch = next(r for r in cr if r.strategy == "shrink_per_batch")

        if final.mean_time < per_batch.mean_time:
            winner = "shrink_final"
            final_wins += 1
        else:
            winner = "shrink_per_batch"
            per_batch_wins += 1

        diff = abs(final.mean_time - per_batch.mean_time) / max(final.mean_time, per_batch.mean_time) * 100
        print(f"  {config:<30}: {winner} ({diff:.1f}% faster)")

    print(f"\n--- Overall ---")
    print(f"  shrink_final wins: {final_wins}")
    print(f"  shrink_per_batch wins: {per_batch_wins}")

    if final_wins > per_batch_wins:
        print(f"\n✓ RECOMMENDATION: shrink_final (wins {final_wins}/{len(configs)} configs)")
    else:
        print(f"\n✓ RECOMMENDATION: shrink_per_batch (wins {per_batch_wins}/{len(configs)} configs)")


# =============================================================================
# Demo
# =============================================================================

def demo():
    print("=" * 60)
    print("DEMO: Dtype Shrinking")
    print("=" * 60)

    rng = np.random.default_rng(42)
    n = 10_000

    df = pl.DataFrame({
        "row_index": np.arange(n, dtype=np.int64),
        "depth": rng.integers(0, 10, n).astype(np.int64),
        "width": rng.integers(100, 4096, n).astype(np.int64),
        "size_bytes": rng.integers(0, 2**40, n).astype(np.int64),
        "mean_intensity": rng.random(n) * 255,
        "variance": rng.random(n) * 1e6,
    })

    print(f"\nBEFORE: {df.estimated_size()/1e6:.2f} MB")
    for col in df.columns:
        print(f"  {col}: {df[col].dtype}")

    shrunk = shrink_dataframe(df)

    print(f"\nAFTER: {shrunk.estimated_size()/1e6:.2f} MB")
    for col in shrunk.columns:
        old, new = df[col].dtype, shrunk[col].dtype
        changed = " ← SHRUNK" if old != new else ""
        print(f"  {col}: {old} → {new}{changed}")

    saved = df.estimated_size() - shrunk.estimated_size()
    print(f"\nSaved: {saved/1e6:.2f} MB ({100*saved/df.estimated_size():.1f}%)")


# =============================================================================
# Main
# =============================================================================

def main(quick: bool = False):
    print("=" * 80)
    print("POLARS DTYPE SHRINKING BENCHMARK")
    print("=" * 80)

    # Validate concat behavior first
    validate_concat_dtype_behavior()

    print("Comparing: shrink_final vs shrink_per_batch")

    configs = get_quick_configs() if quick else get_default_configs()
    print(f"\n[{'QUICK' if quick else 'FULL'} benchmark: {len(configs)} configs]")

    results = run_benchmark_suite(configs)
    print_summary(results)
    print_recommendation(results)

    return results


if __name__ == "__main__":
    import sys

    if "-h" in sys.argv or "--help" in sys.argv:
        print("Usage: python dtype_shrink_benchmark.py [--demo] [--quick]")
        sys.exit(0)

    if "--demo" in sys.argv or "-d" in sys.argv:
        demo()
    else:
        main(quick="--quick" in sys.argv or "-q" in sys.argv)


################################# OUTPUT E's MACHINE #############################################
#
# == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == ==
# POLARS
# DTYPE
# SHRINKING
# BENCHMARK
# == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == ==
# == == == == == == == == == == == == == == == == == == == == == == == == == == == == == ==
# VALIDATING: concat
# dtype
# preservation
# == == == == == == == == == == == == == == == == == == == == == == == == == == == == == ==
# ✓ PASS: Int8 + Int16 → Int16 → got
# Int16
# ✓ PASS: Int8 + Int32 → Int32 → got
# Int32
# ✓ PASS: UInt8 + UInt16 → UInt16 → got
# UInt16
# ✓ PASS: Float32 + Float32 → Float32 → got
# Float32
# ✓ PASS: Float32 + Float64 → Float64 → got
# Float64
# ✓ PASS: Int8 + Int8 + Int16 + Int8 → Int16 → got
# Int16
#
# ✓ ALL
# PASSED: concat
# correctly
# upcasts - shrink_per_batch is SAFE
# == == == == == == == == == == == == == == == == == == == == == == == == == == == == == ==
#
# Comparing: shrink_final
# vs
# shrink_per_batch
#
# [FULL benchmark: 10
# configs]
#
# == == == == == == == == == == == == == == == == == == == == == == == == == == == == == ==
# Config: size_10k
# Rows: 10, 000, Cols: 62
# Batches: 1
# x
# 50, 000
# rows
# == == == == == == == == == == == == == == == == == == == == == == == == == == == == == ==
#
# no_shrink:
# Time: 0.0000
# s ± 0.0000
# s
# Memory: 30.27
# MB → 30.27
# MB(0.0 % saved)
# Throughput: 4, 564, 125, 925
# rows / s
#
# shrink_final:
# Time: 0.0037
# s ± 0.0010
# s
# Memory: 30.27
# MB → 28.22
# MB(6.8 % saved)
# Throughput: 2, 728, 534
# rows / s
#
# shrink_per_batch:
# Time: 0.0033
# s ± 0.0004
# s
# Memory: 30.27
# MB → 28.22
# MB(6.8 % saved)
# Throughput: 3, 031, 042
# rows / s
#
# == == == == == == == == == == == == == == == == == == == == == == == == == == == == == ==
# Config: size_100k
# Rows: 100, 000, Cols: 62
# Batches: 2
# x
# 50, 000
# rows
# == == == == == == == == == == == == == == == == == == == == == == == == == == == == == ==
#
# no_shrink:
# Time: 0.0581
# s ± 0.0363
# s
# Memory: 302.72
# MB → 302.72
# MB(0.0 % saved)
# Throughput: 1, 720, 430
# rows / s
#
# shrink_final:
# Time: 0.0344
# s ± 0.0075
# s
# Memory: 302.72
# MB → 282.22
# MB(6.8 % saved)
# Throughput: 2, 910, 919
# rows / s
#
# shrink_per_batch:
# Time: 0.0362
# s ± 0.0119
# s
# Memory: 302.72
# MB → 282.22
# MB(6.8 % saved)
# Throughput: 2, 762, 845
# rows / s
#
# == == == == == == == == == == == == == == == == == == == == == == == == == == == == == ==
# Config: size_500k
# Rows: 500, 000, Cols: 62
# Batches: 10
# x
# 50, 000
# rows
# == == == == == == == == == == == == == == == == == == == == == == == == == == == == == ==
#
# no_shrink:
# Time: 0.2911
# s ± 0.1896
# s
# Memory: 1513.62
# MB → 1513.62
# MB(0.0 % saved)
# Throughput: 1, 717, 605
# rows / s
#
# shrink_final:
# Time: 0.2570
# s ± 0.2068
# s
# Memory: 1513.62
# MB → 1411.12
# MB(6.8 % saved)
# Throughput: 1, 945, 325
# rows / s
#
# shrink_per_batch:
# Time: 0.1624
# s ± 0.0350
# s
# Memory: 1513.62
# MB → 1411.12
# MB(6.8 % saved)
# Throughput: 3, 07
# 8, 449
# rows / s
#
# == == == == == == == == == == == == == == == == == == == == == == == == == == == == == ==
# Config: size_1m
# Rows: 1, 000, 000, Cols: 62
# Batches: 20
# x
# 50, 000
# rows
# == == == == == == == == == == == == == == == == == == == == == == == == == == == == == ==
#
# no_shrink:
# Time: 0.5779
# s ± 0.3755
# s
# Memory: 3027.25
# MB → 3027.25
# MB(0.0 % saved)
# Throughput: 1, 730, 277
# rows / s
#
# shrink_final:
# Time: 0.3007
# s ± 0.0362
# s
# Memory: 3027.25
# MB → 2822.25
# MB(6.8 % saved)
# Throughput: 3, 325, 920
# rows / s
#
# shrink_per_batch:
# Time: 0.5495
# s ± 0.3335
# s
# Memory: 3027.25
# MB → 2822.25
# MB(6.8 % saved)
# Throughput: 1, 819, 972
# rows / s
#
# == == == == == == == == == == == == == == == == == == == == == == == == == == == == == ==
# Config: size_5m
# Rows: 5, 000, 000, Cols: 62
# Batches: 100
# x
# 50, 000
# rows
# == == == == == == == == == == == == == == == == == == == == == == == == == == == == == ==
#
# no_shrink:
# Time: 2.9213
# s ± 1.9089
# s
# Memory: 15136.25
# MB → 15136.25
# MB(0.0 % saved)
# Throughput: 1, 711, 590
# rows / s
#
# shrink_final:
# Time: 1.5957
# s ± 0.3550
# s
# Memory: 15136.25
# MB → 14111.25
# MB(6.8 % saved)
# Throughput: 3, 133, 437
# rows / s
#
# shrink_per_batch:
# Time: 2.2611
# s ± 0.2926
# s
# Memory: 15136.25
# MB → 14111.25
# MB(6.8 % saved)
# Throughput: 2, 211, 344
# rows / s
#
# == == == == == == == == == == == == == == == == == == == == == == == == == == == == == ==
# Config: 1
# m_batch_10k_x100
# Rows: 1, 000, 000, Cols: 62
# Batches: 100
# x
# 10, 000
# rows
# == == == == == == == == == == == == == == == == == == == == == == == == == == == == == ==
#
# no_shrink:
# Time: 0.2318
# s ± 0.0710
# s
# Memory: 3027.25
# MB → 3027.25
# MB(0.0 % saved)
# Throughput: 4, 314, 07
# 9
# rows / s
#
# shrink_final:
# Time: 0.3299
# s ± 0.0896
# s
# Memory: 3027.25
# MB → 2822.25
# MB(6.8 % saved)
# Throughput: 3, 030, 921
# rows / s
#
# shrink_per_batch:
# Time: 0.5559
# s ± 0.0060
# s
# Memory: 3027.25
# MB → 2822.25
# MB(6.8 % saved)
# Throughput: 1, 798, 741
# rows / s
#
# == == == == == == == == == == == == == == == == == == == == == == == == == == == == == ==
# Config: 1
# m_batch_500k_x2
# Rows: 1, 000, 000, Cols: 62
# Batches: 2
# x
# 500, 000
# rows
# == == == == == == == == == == == == == == == == == == == == == == == == == == == == == ==
#
# no_shrink:
# Time: 0.5637
# s ± 0.3733
# s
# Memory: 3027.25
# MB → 3027.25
# MB(0.0 % saved)
# Throughput: 1, 774, 112
# rows / s
#
# shrink_final:
# Time: 0.3794
# s ± 0.1058
# s
# Memory: 3027.25
# MB → 2822.25
# MB(6.8 % saved)
# Throughput: 2, 635, 655
# rows / s
#
# shrink_per_batch:
# Time: 0.3246
# s ± 0.0717
# s
# Memory: 3027.25
# MB → 2822.25
# MB(6.8 % saved)
# Throughput: 3, 0
# 80, 953
# rows / s
#
# == == == == == == == == == == == == == == == == == == == == == == == == == == == == == ==
# Config: 1
# m_int_heavy
# Rows: 1, 000, 000, Cols: 67
# Batches: 20
# x
# 50, 000
# rows
# == == == == == == == == == == == == == == == == == == == == == == == == == == == == == ==
#
# no_shrink:
# Time: 0.2310
# s ± 0.0786
# s
# Memory: 3080.75
# MB → 3080.75
# MB(0.0 % saved)
# Throughput: 4, 328, 438
# rows / s
#
# shrink_final:
# Time: 0.4580
# s ± 0.3392
# s
# Memory: 3080.75
# MB → 2823.75
# MB(8.3 % saved)
# Throughput: 2, 183, 255
# rows / s
#
# shrink_per_batch:
# Time: 0.3844
# s ± 0.0767
# s
# Memory: 3080.75
# MB → 2823.75
# MB(8.3 % saved)
# Throughput: 2, 601, 361
# rows / s
#
# == == == == == == == == == == == == == == == == == == == == == == == == == == == == == ==
# Config: 1
# m_float_heavy
# Rows: 1, 000, 000, Cols: 67
# Batches: 20
# x
# 50, 000
# rows
# == == == == == == == == == == == == == == == == == == == == == == == == == == == == == ==
#
# no_shrink:
# Time: 0.2228
# s ± 0.0703
# s
# Memory: 3080.75
# MB → 3080.75
# MB(0.0 % saved)
# Throughput: 4, 488, 558
# rows / s
#
# shrink_final:
# Time: 0.3277
# s ± 0.0699
# s
# Memory: 3080.75
# MB → 2833.75
# MB(8.0 % saved)
# Throughput: 3, 051, 969
# rows / s
#
# shrink_per_batch:
# Time: 0.4066
# s ± 0.0697
# s
# Memory: 3080.75
# MB → 2833.75
# MB(8.0 % saved)
# Throughput: 2, 459, 541
# rows / s
#
# == == == == == == == == == == == == == == == == == == == == == == == == == == == == == ==
# Config: pixel_patrol
# Rows: 71, 000, Cols: 79
# Batches: 8
# x
# 10, 000
# rows
# == == == == == == == == == == == == == == == == == == == == == == == == == == == == == ==
#
# no_shrink:
# Time: 0.0240
# s ± 0.0065
# s
# Memory: 405.27
# MB → 405.27
# MB(0.0 % saved)
# Throughput: 2, 961, 208
# rows / s
#
# shrink_final:
# Time: 0.0282
# s ± 0.0004
# s
# Memory: 405.27
# MB → 387.59
# MB(4.4 % saved)
# Throughput: 2, 521, 189
# rows / s
#
# shrink_per_batch:
# Time: 0.0454
# s ± 0.0076
# s
# Memory: 405.27
# MB → 387.59
# MB(4.4 % saved)
# Throughput: 1, 562, 419
# rows / s
#
# == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == ==
# SUMMARY
# == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == ==
#
# Config
# Strategy
# Time(s)
# Saved
# Winner
# -------------------------------------------------------------------------------------
# size_10k
# no_shrink
# 0.0000
# 0.0 % (baseline)
# size_10k
# shrink_final
# 0.0037
# 6.8 %
# size_10k
# shrink_per_batch
# 0.0033
# 6.8 % ← BEST
#
# size_100k
# no_shrink
# 0.0581
# 0.0 % (baseline)
# size_100k
# shrink_final
# 0.0344
# 6.8 % ← BEST
# size_100k
# shrink_per_batch
# 0.0362
# 6.8 %
#
# size_500k
# no_shrink
# 0.2911
# 0.0 % (baseline)
# size_500k
# shrink_final
# 0.2570
# 6.8 %
# size_500k
# shrink_per_batch
# 0.1624
# 6.8 % ← BEST
#
# size_1m
# no_shrink
# 0.5779
# 0.0 % (baseline)
# size_1m
# shrink_final
# 0.3007
# 6.8 % ← BEST
# size_1m
# shrink_per_batch
# 0.5495
# 6.8 %
#
# size_5m
# no_shrink
# 2.9213
# 0.0 % (baseline)
# size_5m
# shrink_final
# 1.5957
# 6.8 % ← BEST
# size_5m
# shrink_per_batch
# 2.2611
# 6.8 %
#
# 1
# m_batch_10k_x100
# no_shrink
# 0.2318
# 0.0 % (baseline)
# 1
# m_batch_10k_x100
# shrink_final
# 0.3299
# 6.8 % ← BEST
# 1
# m_batch_10k_x100
# shrink_per_batch
# 0.5559
# 6.8 %
#
# 1
# m_batch_500k_x2
# no_shrink
# 0.5637
# 0.0 % (baseline)
# 1
# m_batch_500k_x2
# shrink_final
# 0.3794
# 6.8 %
# 1
# m_batch_500k_x2
# shrink_per_batch
# 0.3246
# 6.8 % ← BEST
#
# 1
# m_int_heavy
# no_shrink
# 0.2310
# 0.0 % (baseline)
# 1
# m_int_heavy
# shrink_final
# 0.4580
# 8.3 %
# 1
# m_int_heavy
# shrink_per_batch
# 0.3844
# 8.3 % ← BEST
#
# 1
# m_float_heavy
# no_shrink
# 0.2228
# 0.0 % (baseline)
# 1
# m_float_heavy
# shrink_final
# 0.3277
# 8.0 % ← BEST
# 1
# m_float_heavy
# shrink_per_batch
# 0.4066
# 8.0 %
#
# pixel_patrol
# no_shrink
# 0.0240
# 0.0 % (baseline)
# pixel_patrol
# shrink_final
# 0.0282
# 4.4 % ← BEST
# pixel_patrol
# shrink_per_batch
# 0.0454
# 4.4 %
#
# == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == ==
# ANALYSIS
# == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == ==
#
# --- Per - config
# winners - --
# size_10k: shrink_per_batch(10.0 % faster)
# size_100k: shrink_final(5.1 % faster)
# size_500k: shrink_per_batch(36.8 % faster)
# size_1m: shrink_final(45.3 % faster)
# size_5m: shrink_final(29.4 % faster)
# 1
# m_batch_10k_x100: shrink_final(40.7 % faster)
# 1
# m_batch_500k_x2: shrink_per_batch(14.5 % faster)
# 1
# m_int_heavy: shrink_per_batch(16.1 % faster)
# 1
# m_float_heavy: shrink_final(19.4 % faster)
# pixel_patrol: shrink_final(38.0 % faster)
#
# --- Overall - --
# shrink_final
# wins: 6
# shrink_per_batch
# wins: 4
#
# ✓ RECOMMENDATION: shrink_final(wins
# 6 / 10
# configs)
