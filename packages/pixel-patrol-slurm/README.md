### PixelPatrol SLURM Launcher (pixel-patrol-slurm)

Launches a [Dask SLURMCluster](https://jobqueue.dask.org/en/latest/generated/dask_jobqueue.SLURMCluster.html), waits for workers to come online, then runs `pixel-patrol process` against the cluster.

#### Installation

```bash
pip install pixel-patrol-slurm
```

#### Usage

```
pixel-patrol-slurm [SLURM options] -- BASE_DIR [pixel-patrol process options]
```

Arguments **before** `--` control the Slurm cluster.
Arguments **after** `--` are passed verbatim to `pixel-patrol process` (`--scheduler` is injected automatically).

#### Examples

```bash
# Minimal — use defaults (4 jobs, 4 cores, 16 GB, 2 h walltime)
pixel-patrol-slurm -- /data/images --output results.parquet

# Custom cluster
pixel-patrol-slurm --jobs 16 --cores 4 --memory 32GB --partition gpu \
  -- /data/images --output results.parquet --flavor myproject

# Short walltime, fewer jobs
pixel-patrol-slurm --walltime 04:00:00 --jobs 8 \
  -- /data/images -o out.parquet --processors-exclude histogram
```

#### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--jobs` | 4 | Number of Slurm jobs (workers) to submit |
| `--cores` | 4 | CPU cores per job |
| `--memory` | 16GB | Memory per job |
| `--partition` | *(none)* | Slurm partition / queue name |
| `--walltime` | 02:00:00 | Job walltime HH:MM:SS |
| `--processes` | 1 | Dask worker processes per job |
| `--wait` | 60 | Seconds to wait for workers to start |
| `--min-workers` | 1 | Minimum workers before proceeding |
| `--env-extra` | *(none)* | Extra environment string, e.g. `'module load cuda'` |
