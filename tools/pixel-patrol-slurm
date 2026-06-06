#!/usr/bin/env python3
"""
Launch a Dask SLURMCluster, wait for workers, then run `pixel-patrol process`.

Usage:
  pixel-patrol-slurm [SLURM options] -- BASE_DIR [pixel-patrol process options]

Examples:
  pixel-patrol-slurm -- /data/images --output results.parquet
  pixel-patrol-slurm --jobs 16 --cores 4 --memory 32GB --partition gpu -- /data/images --output results.parquet --flavor myproject
  pixel-patrol-slurm --walltime 04:00:00 --jobs 8 -- /data/images -o out.parquet --processors-exclude histogram

All arguments after -- are passed verbatim to `pixel-patrol process` (with --scheduler injected automatically).
Arguments before -- control the Slurm cluster.
"""

import argparse
import subprocess
import sys
import time


def parse_args():
    p = argparse.ArgumentParser(
        description="Run pixel-patrol process on a Dask SLURMCluster.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
        add_help=True,
    )
    p.add_argument("--jobs",       type=int,   default=4,        help="Number of Slurm jobs (workers) to submit (default: 4)")
    p.add_argument("--cores",      type=int,   default=4,        help="CPU cores per job (default: 4)")
    p.add_argument("--memory",     type=str,   default="16GB",   help="Memory per job, e.g. 16GB (default: 16GB)")
    p.add_argument("--partition",  type=str,   default=None,     help="Slurm partition / queue name")
    p.add_argument("--walltime",   type=str,   default="02:00:00", help="Job walltime HH:MM:SS (default: 02:00:00)")
    p.add_argument("--processes",  type=int,   default=1,        help="Dask worker processes per job (default: 1)")
    p.add_argument("--wait",       type=int,   default=60,       help="Seconds to wait for workers to start (default: 60)")
    p.add_argument("--min-workers",type=int,   default=1,        help="Minimum workers before proceeding (default: 1)")
    p.add_argument("--env-extra",  type=str,   default=None,     help="Extra environment string for Slurm job, e.g. 'module load cuda'")

    # Capture everything after -- as pixel-patrol process args
    argv = sys.argv[1:]
    if "--" in argv:
        split = argv.index("--")
        slurm_argv = argv[:split]
        process_argv = argv[split + 1:]
    else:
        slurm_argv = []
        process_argv = argv

    args = p.parse_args(slurm_argv)
    return args, process_argv


def main():
    args, process_argv = parse_args()

    if not process_argv:
        print("Error: no arguments for `pixel-patrol process` provided after --", file=sys.stderr)
        print("Usage: pixel-patrol-slurm [slurm options] -- BASE_DIR [pixel-patrol process options]", file=sys.stderr)
        sys.exit(1)

    try:
        from dask_jobqueue import SLURMCluster
        from dask.distributed import Client
    except ImportError:
        print("dask-jobqueue is not installed. Run: pip install dask-jobqueue", file=sys.stderr)
        sys.exit(1)

    cluster_kwargs = dict(
        cores=args.cores,
        memory=args.memory,
        processes=args.processes,
        walltime=args.walltime,
    )
    if args.partition:
        cluster_kwargs["queue"] = args.partition
    if args.env_extra:
        cluster_kwargs["env_extra"] = [args.env_extra]

    print(f"Starting SLURMCluster: {args.jobs} jobs × {args.cores} cores × {args.memory} …")
    cluster = SLURMCluster(**cluster_kwargs)
    cluster.scale(jobs=args.jobs)

    print(f"Scheduler address: {cluster.scheduler_address}")
    print(f"Waiting up to {args.wait}s for at least {args.min_workers} worker(s) …")

    client = Client(cluster)
    deadline = time.monotonic() + args.wait
    while time.monotonic() < deadline:
        n = len(client.scheduler_info().get("workers", {}))
        if n >= args.min_workers:
            print(f"  {n} worker(s) online.")
            break
        print(f"  {n} worker(s) online, waiting …")
        time.sleep(5)
    else:
        n = len(client.scheduler_info().get("workers", {}))
        if n == 0:
            print("No workers came online in time. Check `squeue` for job status.", file=sys.stderr)
            client.close()
            cluster.close()
            sys.exit(1)
        print(f"Proceeding with {n} worker(s).")

    cmd = ["pixel-patrol", "process", "--scheduler", cluster.scheduler_address] + process_argv
    print(f"\nRunning: {' '.join(cmd)}\n")

    try:
        result = subprocess.run(cmd)
    finally:
        print("\nShutting down cluster …")
        client.close()
        cluster.close()

    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
