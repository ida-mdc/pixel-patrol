# Benchmarks (pixel-patrol)

## Run E2E benchmark

Measures full pipeline: process → export → import → widget

```
python benchmark_e2e.py --mode quick
```

Compare against another branch:

```
python benchmark_e2e.py --mode quick --branch main
```

Use existing data from dir (instead of data generation):

```
python benchmark_e2e.py --mode quick --dataset-dir /path/to/datasets
```

### Args (also for run processor benchmark)

`--mode` {`quick`,`full`} (default: `quick`)  
`--branch` <git-branch> (optional comparison)  
`--dataset-dir` <path> (optional)  


## Run processor benchmark

Measures per-processor execution time.

```
python benchmark_processors.py --mode quick
```

(or with other args)

### Generate report (from CSV)

Creates `REPORT.md` (and optional plots).

```
python generate_benchmark_report.py --csv results/e2e_results.csv
```

### Args:

--`csv` <path> (required)  
--`figures` (off by default)  
--`ref-branch` <branch> (delta = branch − ref)  
--`processor` <NAME> (run for one processor only)  
