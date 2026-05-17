#!/usr/bin/env python
"""
Run pixel-patrol processing distributed across SLURM worker jobs.

Submit this script as a lightweight coordinator job:

    sbatch -J pp-coord -t 48:00:00 --mem=8G -c 2 \\
        --wrap="python scripts/run_slurm_cluster.py \\
            /fast/project/data \\
            /fast/output/report.parquet \\
            --name my_project \\
            --workers 20 \\
            --mem-per-worker 60GB \\
            --walltime 47:30:00"

Each --workers job is submitted as a separate SLURM step by dask-jobqueue.
Adapt --partition / --account below to match your cluster.
"""

import argparse
import os
from pathlib import Path

from dask_jobqueue import SLURMCluster
from dask.distributed import Client

from pixel_patrol_base.api import create_project, add_paths, process_files


def main():
    p = argparse.ArgumentParser()
    p.add_argument("data_dir", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--name", default=None)
    p.add_argument("--loader", default="tifffile")
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--mem-per-worker", default="60GB")
    p.add_argument("--walltime", default="47:30:00")
    p.add_argument("--tile-size", type=int, default=1024)
    p.add_argument("--chunk-mb", type=int, default=1024)
    p.add_argument("--partition", default=None)   # set to your cluster partition
    p.add_argument("--account", default=None)     # set to your cluster account
    args = p.parse_args()

    os.environ["PIXEL_PATROL_STATS_TILE_SIZE"] = str(args.tile_size)
    os.environ["PIXEL_PATROL_MAX_BLOCK_MB"] = str(args.chunk_mb)

    extra = []
    if args.partition:
        extra.append(f"--partition={args.partition}")
    if args.account:
        extra.append(f"--account={args.account}")

    cluster = SLURMCluster(
        cores=1,
        processes=1,
        memory=args.mem_per_worker,
        walltime=args.walltime,
        job_extra_directives=extra,
        # pass env vars into worker jobs
        job_script_prologue=[
            f"export PIXEL_PATROL_STATS_TILE_SIZE={args.tile_size}",
            f"export PIXEL_PATROL_MAX_BLOCK_MB={args.chunk_mb}",
            "export DASK_DISTRIBUTED__LOGGING__DISTRIBUTED=error",
        ],
    )
    cluster.scale(jobs=args.workers)

    with Client(cluster) as client:
        print(f"Scheduler:  {client.scheduler.address}")
        print(f"Dashboard:  {client.dashboard_link}")
        print(f"Waiting for workers (need ≥1 of {args.workers})...")
        client.wait_for_workers(n_workers=1, timeout=600)
        print(f"Workers ready: {len(client.scheduler_info()['workers'])}")

        name = args.name or args.data_dir.name
        project = create_project(name, str(args.data_dir),
                                  loader=args.loader,
                                  output_path=args.output)
        add_paths(project, args.data_dir)
        process_files(project)


if __name__ == "__main__":
    main()
