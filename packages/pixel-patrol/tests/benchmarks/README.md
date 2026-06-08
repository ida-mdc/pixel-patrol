# Benchmarks

Four standalone scripts, each measuring a different layer of the stack.
Run from this directory (`cd packages/pixel-patrol/tests/benchmarks`).

---

## benchmark_pipeline.py - pipeline throughput & scaling

Measures Dask pipeline performance using a NullLoader (no disk I/O) and NumpyLoader
(real I/O). Covers four scenarios: tiny batches, many small files, spatially-chunked
files, and container files with many sub-images. Reports wall time, stall time, worker
RSS, throughput, and speedup vs 1 worker.

```bash
python benchmark_pipeline.py                           # all scenarios, 1/2/4 workers
python benchmark_pipeline.py --scenarios container_subimages many_small
python benchmark_pipeline.py --workers 1 8
python benchmark_pipeline.py --scheduler tcp://host:8786   # remote Dask cluster
python benchmark_pipeline.py --skip-file               # memory mode only
```

Results saved to `results/pipeline_<timestamp>.json`.

---

## benchmark_e2e.py - full pipeline + viewer

Measures end-to-end time: process → viewer widget load. Uses synthetic TIFF files
and the bioio loader. Results saved to `results/e2e_results.csv`.

```bash
python benchmark_e2e.py --mode quick
python benchmark_e2e.py --mode full
python benchmark_e2e.py --branch main        # compare against another branch
python benchmark_e2e.py --dataset-dir /path  # reuse existing data
```

Generate a Markdown report from the CSV:

```bash
python generate_benchmark_report.py --csv results/e2e_results.csv --figures
```

---

## benchmark_processors.py - per-processor timing in pipeline context

Measures individual processor wall time within a real pipeline run. Injects
instrumented wrappers via the plugin registry to record CPU seconds per processor.

```bash
python benchmark_processors.py --mode quick
python benchmark_processors.py --mode full --branch main
```

---

## benchmark_metrics.py - isolated processor cost

Measures the cost of each processor's `run_chunk()` call on a fixed in-memory
plane, with warmup + multiple samples. No Dask, no I/O.

```bash
python benchmark_metrics.py
```

---

## Common args (e2e + processors)

| Arg | Description |
|---|---|
| `--mode quick\|full` | Dataset size (default: quick) |
| `--branch BRANCH` | Compare against another git branch |
| `--dataset-dir PATH` | Reuse pre-generated datasets |
