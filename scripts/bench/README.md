# Benchmark processors

Focus: to highlight the histogram processor, pass `--focus-processor HistogramProcessor` or `--focus-processor histogram` (the latter matches plugin NAME).


This folder contains a small benchmarking helper `benchmark_processors.py` to profile and compare processor performance across branches.

Quick start

- Single-tree local run (recommended for per-processor timings):

```bash
python scripts/bench/benchmark_processors.py --run-local --base-path examples/datasets/demo_data_2d --processing-max-workers 1 --out bench_out/local.json
```

- Compare branches (automates `git worktree` creation and runs local benchmark in each):

```bash
python scripts/bench/benchmark_processors.py --branches main my-feature-branch --out-dir bench_out
```

- Example: compare three branches, write cProfile files per branch, and automatically generate SnakeViz HTML reports and open them in your browser (no changes to your current worktree). Install SnakeViz with `pip install snakeviz` if you don't have it.

```bash
# from repository root
python scripts/bench/benchmark_processors.py --branches main feature/a feature/b \
  --out-dir bench_out --processing-max-workers 1 --snakeviz --open-snakeviz \
  --focus-processor HistogramProcessor
```

Note: if some branches do not contain the benchmark script or the dataset used for benchmarking, the runner will by default use the host repository's script and dataset for the run so results are comparable. You can override this behaviour with `--no-use-host-script` or `--no-use-host-data` to allow branch-local scripts/data when available.

Postprocessing only:
If you already have profiler files in `--out-dir` (for example if a run aborted), you can start SnakeViz on existing profiles without re-running benchmarks:

```bash
python scripts/bench/benchmark_processors.py --postprocess-only --out-dir bench_out --snakeviz --open-snakeviz
```
This writes `bench_out/postprocess_summary.json` with discovered `.prof` files and any SnakeViz PIDs started.
Or, if you prefer to use the project's `uv` runner from a subdir (your current workflow), two equivalent ways:

```bash
# 1) run from repository root via uv
uv run scripts/bench/benchmark_processors.py --branches main feature/a feature/b \
  --out-dir bench_out --processing-max-workers 1 --snakeviz --open-snakeviz \
  --focus-processor HistogramProcessor

# 2) run from the examples directory as you already do
cd examples
uv run ../scripts/bench/benchmark_processors.py --branches main feature/a feature/b \
  --out-dir ../bench_out --processing-max-workers 1 --snakeviz --open-snakeviz \
  --focus-processor HistogramProcessor
```

This will create profiler files `bench_out/bench_profile_{branch}.prof` and HTML files `bench_out/bench_profile_{branch}_snakeviz.html` and open them in your browser.
Notes

- Per-processor timings are accurate when `--processing-max-workers 1` (single process), because worker processes do not report timing back to the main process.
- The script will attempt to add temporary `git worktree`s for branch comparisons; you must have `git` available and permissions to create/remove worktrees. The script uses detached worktrees and does not modify your current checked-out worktree or its index.
- If you want deeper profiling, pass `--cprofile path/to/profile.prof` to have cProfile data written during the run.
- The benchmark output is written as JSON and includes `wall_time_seconds`, `files_processed`, `memory_rss_bytes` and a `per_processor` breakdown (with `calls` and `total_time`).

Dependencies

- `psutil` is optional but recommended for memory reporting.
- `snakeviz` is optional for generating HTML reports from cProfile files.