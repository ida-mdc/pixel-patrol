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

Outputs (written next to the parquet file):
  report.parquet              — main data
  report.summary.txt          — compact timing summary
  report.cluster_report.html  — dask dashboard snapshot (task stream, CPU, RAM, bandwidth)
  report.cluster_stats.json   — per-worker task counts, final memory/CPU readings
"""

import argparse
import json
import os
import time
from pathlib import Path

from dask_jobqueue import SLURMCluster
from dask.distributed import Client, performance_report

from pixel_patrol_base.api import create_project, add_paths, process_files


def _collect_worker_stats(client: Client) -> dict:
    """Collect per-worker task counts and final resource state before shutdown."""
    stats = {"workers": {}, "collected_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    try:
        info = client.scheduler_info().get("workers", {})
        for addr, w in info.items():
            hostname = addr.split("//")[-1].split(":")[0]
            stats["workers"][addr] = {
                "host":          hostname,
                "nthreads":      w.get("nthreads", 0),
                "memory_limit":  w.get("memory_limit", 0),
                "metrics":       w.get("metrics", {}),
            }
    except Exception as exc:
        stats["error"] = str(exc)

    # Run psutil on every worker to get final CPU / RAM snapshot.
    try:
        def _worker_snapshot():
            import psutil, os
            vm = psutil.virtual_memory()
            cpu = psutil.cpu_percent(interval=0.5)
            return {
                "pid":          os.getpid(),
                "cpu_pct":      cpu,
                "ram_used_gb":  vm.used  / 1024**3,
                "ram_total_gb": vm.total / 1024**3,
                "ram_pct":      vm.percent,
            }
        snapshots = client.run(_worker_snapshot)
        for addr, snap in snapshots.items():
            if addr in stats["workers"]:
                stats["workers"][addr]["snapshot"] = snap
    except Exception:
        pass

    return stats


def _print_cluster_summary(stats: dict) -> None:
    workers = stats.get("workers", {})
    if not workers:
        return
    print(f"\n  Cluster stats  ({len(workers)} workers)")
    print(f"  {'Host':<20}  {'CPU%':>5}  {'RAM used':>10}  {'RAM%':>5}")
    print(f"  {'─'*20}  {'─'*5}  {'─'*10}  {'─'*5}")
    for w in sorted(workers.values(), key=lambda x: x["host"]):
        snap = w.get("snapshot", {})
        cpu   = snap.get("cpu_pct", float("nan"))
        ram_u = snap.get("ram_used_gb", float("nan"))
        ram_p = snap.get("ram_pct", float("nan"))
        print(f"  {w['host']:<20}  {cpu:>4.0f}%  {ram_u:>8.1f} GB  {ram_p:>4.0f}%")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("data_dir", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--name", default=None)
    p.add_argument("--loader", default="tifffile")
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--min-workers", type=int, default=None,
                   help="Wait for at least this many workers before starting (default: all --workers).")
    p.add_argument("--mem-per-worker", default="60GB")
    p.add_argument("--walltime", default="47:30:00")
    p.add_argument("--tile-size", type=int, default=1024)
    p.add_argument("--chunk-mb", type=int, default=1024)
    p.add_argument("--partition", default=None)
    p.add_argument("--account", default=None)
    p.add_argument("--processors-exclude", nargs="*", default=None,
                   help="Processor names to skip, e.g. channel-colocalization")
    args = p.parse_args()

    os.environ["PIXEL_PATROL_STATS_TILE_SIZE"] = str(args.tile_size)
    os.environ["PIXEL_PATROL_MAX_BLOCK_MB"] = str(args.chunk_mb)

    output = Path(args.output).resolve()
    perf_report_path = output.with_suffix(".cluster_report.html")
    stats_path        = output.with_suffix(".cluster_stats.json")

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
        job_script_prologue=[
            f"export PIXEL_PATROL_STATS_TILE_SIZE={args.tile_size}",
            f"export PIXEL_PATROL_MAX_BLOCK_MB={args.chunk_mb}",
            "export DASK_DISTRIBUTED__LOGGING__DISTRIBUTED=error",
        ],
    )
    cluster.scale(jobs=args.workers)

    min_workers = args.min_workers if args.min_workers is not None else args.workers

    with Client(cluster) as client:
        print(f"Scheduler:  {client.scheduler.address}")
        print(f"Dashboard:  {client.dashboard_link}")
        print(f"Waiting for workers (need ≥{min_workers} of {args.workers})...")
        client.wait_for_workers(n_workers=min_workers, timeout=600)
        n_ready = len(client.scheduler_info()["workers"])
        print(f"Workers ready: {n_ready}")

        name = args.name or args.data_dir.name
        project = create_project(name, str(args.data_dir),
                                  loader=args.loader,
                                  output_path=output)
        add_paths(project, args.data_dir)
        excluded = set(args.processors_exclude) if args.processors_exclude else None

        # performance_report captures the full Bokeh dashboard:
        # task stream, CPU/RAM/bandwidth per worker over the entire run.
        with performance_report(filename=str(perf_report_path)):
            process_files(project, processors_excluded=excluded)

        print(f"Cluster report: {perf_report_path}")

        # Collect final per-worker snapshot before workers disconnect.
        stats = _collect_worker_stats(client)
        stats_path.write_text(json.dumps(stats, indent=2))
        print(f"Cluster stats:  {stats_path}")
        _print_cluster_summary(stats)


if __name__ == "__main__":
    main()
