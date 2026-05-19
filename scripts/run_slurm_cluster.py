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
import os
import re
import time
from pathlib import Path

from dask_jobqueue import SLURMCluster
from dask.distributed import Client, performance_report

from pixel_patrol_base.api import create_project, add_paths, process_files

_W = 62   # box width — matches processing_summary.py


def _parse_cluster_report(html_path: Path) -> dict:
    """Extract CPU, memory and worker stats from a dask performance_report HTML file."""
    try:
        html = html_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return {}

    stats: dict = {}

    # CPU min / max / mean  (e.g.  'min: 0.0%'  'max: 213.9%'  'mean: 19.1%')
    cpu = {}
    for label, val in re.findall(r'"(min|max|mean): ([\d.]+)%"', html):
        cpu[label] = float(val)
    if cpu:
        stats["cpu"] = cpu

    # Memory  (e.g.  'min: 299.70 MiB'  'max: 3.45 GiB')
    mem = {}
    for label, val, unit in re.findall(r'"(min|max|mean): ([\d.]+) (MiB|GiB)"', html):
        gb = float(val) / 1024 if unit == "MiB" else float(val)
        mem[label] = gb
    if mem:
        stats["memory_gb"] = mem

    # Unique worker addresses embedded in the task-stream data
    addrs = set(re.findall(r"tcp://[\d.]+:\d+", html))
    if addrs:
        stats["worker_addrs"] = sorted(addrs)
        stats["n_workers_connected"] = len(addrs)
        stats["worker_nodes"] = sorted(set(
            a.split("//")[1].split(":")[0] for a in addrs
        ))

    return stats


def _append_cluster_section(summary_path: Path, cluster_stats: dict,
                             workers_requested: int) -> None:
    """Append a compact cluster-utilisation block to the .summary.txt file."""
    if not cluster_stats:
        return

    def _ln(content: str) -> str:
        return f"║{content:<{_W}}║"

    lines = ["", f"╔{'═' * _W}╗"]
    lines.append(_ln("  Cluster utilisation  (source: dask performance_report)".center(_W)))
    lines.append(f"╠{'═' * _W}╣")

    n_conn = cluster_stats.get("n_workers_connected", 0)
    nodes  = cluster_stats.get("worker_nodes", [])
    lines.append(_ln(f"  Workers  {n_conn} connected / {workers_requested} requested"
                     f"  ·  {len(nodes)} node{'s' if len(nodes) != 1 else ''}"))

    cpu = cluster_stats.get("cpu", {})
    if cpu:
        lines.append(_ln(
            f"  CPU      min {cpu.get('min', 0):.0f}%"
            f"  ·  mean {cpu.get('mean', 0):.0f}%"
            f"  ·  max {cpu.get('max', 0):.0f}%"
        ))

    mem = cluster_stats.get("memory_gb", {})
    if mem:
        def _fmt_gb(v: float) -> str:
            return f"{v:.2f} GB" if v < 1 else f"{v:.1f} GB"
        lines.append(_ln(
            f"  RAM      min {_fmt_gb(mem.get('min', 0))}"
            f"  ·  mean {_fmt_gb(mem.get('mean', 0))}"
            f"  ·  max {_fmt_gb(mem.get('max', 0))}"
        ))

    if nodes:
        prefix = "  Nodes    "
        shown, rest = [], list(nodes)
        avail = _W - len(prefix)
        while rest:
            trailer = f"  +{len(rest)} more" if rest else ""
            c = rest[0]
            if len("  ".join(shown + [c])) + len(trailer) <= avail:
                shown.append(rest.pop(0))
            else:
                break
        trailer = f"  +{len(rest)} more" if rest else ""
        lines.append(_ln(prefix + "  ".join(shown) + trailer))

    lines.append(f"╚{'═' * _W}╝")

    with open(summary_path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


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
    p.add_argument("--tile-size", type=int, default=None,
                   help="XY tile size in pixels for spatial metrics (default: 256 with --tile-metrics, "
                        "65536 without — effectively one tile per plane).")
    p.add_argument("--tile-metrics", action="store_true", default=False,
                   help="Store per-tile spatial metrics in output (off by default; "
                        "increases output size and coordinator RAM usage significantly).")
    p.add_argument("--chunk-mb", type=int, default=1024)
    p.add_argument("--partition", default=None)
    p.add_argument("--account", default=None)
    p.add_argument("--processors-exclude", nargs="*", default=None,
                   help="Processor names to skip, e.g. channel-colocalization")
    args = p.parse_args()

    tile_size = args.tile_size if args.tile_size is not None else (256 if args.tile_metrics else 65536)
    os.environ["PIXEL_PATROL_STATS_TILE_SIZE"] = str(tile_size)
    os.environ["PIXEL_PATROL_MAX_BLOCK_MB"] = str(args.chunk_mb)
    if args.tile_metrics:
        os.environ["PIXEL_PATROL_RASTER_XY_TILE_METRICS"] = "1"

    output = Path(args.output).resolve()
    perf_report_path = output.with_suffix(".cluster_report.html")

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
            f"export PIXEL_PATROL_STATS_TILE_SIZE={tile_size}",
            f"export PIXEL_PATROL_MAX_BLOCK_MB={args.chunk_mb}",
            f"export PIXEL_PATROL_RASTER_XY_TILE_METRICS={'1' if args.tile_metrics else '0'}",
            "export DASK_DISTRIBUTED__LOGGING__DISTRIBUTED=error",
        ],
    )
    cluster.scale(jobs=args.workers)

    min_workers = args.min_workers if args.min_workers is not None else args.workers

    with Client(cluster) as client:
        print(f"Scheduler:  {client.scheduler.address}")
        print(f"Dashboard:  {client.dashboard_link}")
        print(f"Waiting for workers (need ≥{min_workers} of {args.workers})...")
        client.wait_for_workers(n_workers=min_workers)
        n_ready = len(client.scheduler_info()["workers"])
        print(f"Workers ready: {n_ready}/{args.workers}")

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

        # Parse the HTML and append a cluster section to the summary file.
        cluster_stats = _parse_cluster_report(perf_report_path)
        summary_path  = output.with_suffix(".summary.txt")
        _append_cluster_section(summary_path, cluster_stats, args.workers)

        # Print the full summary (includes both processing + cluster sections).
        if summary_path.exists():
            print()
            print(summary_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
