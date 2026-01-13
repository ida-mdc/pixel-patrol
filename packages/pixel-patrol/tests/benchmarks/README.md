# Benchmark processors
This folder contains a small benchmarking helper `benchmark_processors.py` and a small dataset to profile and compare processor performance across branches.

You can focus a plguin processor in the summary report:  
To highlight e.g. the histogram processor, pass `--focus-processor HistogramProcessor` (class name) or `--focus-processor histogram` (the latter matches plugin name).

### Quick start

#### Run local state (recommended for per-processor timings):

```bash
python packages/pixel-patrol/tests/benchmarks/benchmark_processors.py --run-local --base-path packages/pixel-patrol/tests/benchmarks/benchmark_dataset --processing-max-workers 1 --out packages/pixel-patrol/tests/benchmarks/output/local.json
```

#### Compare branches (automates `git worktree` creation and runs local benchmark in each):

```bash
python packages/pixel-patrol/tests/benchmarks/benchmark_processors.py --branches origin/main my-feature-branch --out-dir packages/pixel-patrol/tests/benchmarks/output
```

Example: compare three branches, write cProfile files per branch, and automatically generate SnakeViz HTML reports and open them in your browser (no changes to your current worktree). Install SnakeViz with `pip install snakeviz` if you don't have it.

```bash
# from repository root
python packages/pixel-patrol/tests/benchmarks/benchmark_processors.py --branches origin/main feature/a feature/b \
  --out-dir packages/pixel-patrol/tests/benchmarks/output --processing-max-workers 1 --snakeviz --open-snakeviz \
  --focus-processor HistogramProcessor
```
Note: if some branches do not contain the benchmark script or the dataset used for benchmarking, the runner will by default use the host repository's script and dataset for the run so results are comparable. You can override this behaviour with `--no-use-host-script` or `--no-use-host-data` to allow branch-local scripts/data when available.

#### Postprocessing only:
If you already have profiler files in `--out-dir` (for example if a run aborted), you can start SnakeViz on existing profiles without re-running benchmarks:

```bash
python packages/pixel-patrol/tests/benchmarks/benchmark_processors.py --postprocess-only --out-dir packages/pixel-patrol/tests/benchmarks/output --snakeviz --open-snakeviz
```
This writes `packages/pixel-patrol/tests/benchmarks/output/postprocess_summary.json` with discovered `.prof` files and any SnakeViz PIDs started.
Or, if you prefer to use the project's `uv` runner from a subdir (your current workflow), two equivalent ways:

```bash
# 1) run from repository root via uv
uv run packages/pixel-patrol/tests/benchmarks/benchmark_processors.py --branches main feature/a feature/b \
  --out-dir bench_out --processing-max-workers 1 --snakeviz --open-snakeviz \
  --focus-processor HistogramProcessor

# 2) run from the examples directory as you already do
cd examples
uv run ../packages/pixel-patrol/tests/benchmarks/benchmark_processors.py --branches main feature/a feature/b \
  --out-dir ../bench_out --processing-max-workers 1 --snakeviz --open-snakeviz \
  --focus-processor HistogramProcessor
```

This will create profiler files `packages/pixel-patrol/tests/benchmarks/output/bench_profile_{branch}.prof` and start SnakeViz servers for those profiles; use the `--postprocess-only` option to operate on existing profiles.

### Notes

- Per-processor timings are accurate when `--processing-max-workers 1` (single process), because worker processes do not report timing back to the main process.
- The script will attempt to add temporary `git worktree`s for branch comparisons; you must have `git` available and permissions to create/remove worktrees. The script uses detached worktrees and does not modify your current checked-out worktree or its index.
- If you want deeper profiling, pass `--cprofile path/to/profile.prof` to have cProfile data written during the run.
- The benchmark output is written as JSON and includes `wall_time_seconds`, `files_processed`, `memory_rss_bytes` and a `per_processor` breakdown (with `calls` and `total_time`).

#### Dependencies

- `psutil` is optional but recommended for memory reporting.
- `snakeviz` is optional for generating HTML reports from cProfile files.
