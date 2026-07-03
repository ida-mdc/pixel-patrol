"""Process the whole cpg0036-EU-OS-bioactives dataset from S3, on a SLURM cluster.

Detects every load_data.csv manifest across all four imaging sites
(FMP, IMTM, MEDINA, USC) and their batches/plates on the public AWS Cell Painting
Gallery, adds them all as inputs, and processes the channel TIFFs directly from
S3 (streamed, never bulk-downloaded). The imaging site and batch are derived from
each manifest's S3 path and land in the report as Metadata_ImagingSite /
Metadata_BatchDir (the manifest also keeps its own Metadata_Site = field number).

Scale: the full dataset is ~32 batches x many plates x thousands of fields x 4
channels - hundreds of thousands of images. Use the knobs below to sample while
trying things out, then remove the caps for a full run.

Cluster: uses a Dask SLURMCluster (dask-jobqueue) when USE_SLURM is on, otherwise a
local cluster so the script also runs on a workstation. Because loading is
I/O-bound, Pixel Patrol plans one image per task and spreads the concurrent S3
reads across all workers automatically.

Requires: pixel-patrol-loader-bio, s3fs, and (for SLURM) dask-jobqueue - installed
in an environment available on the compute nodes.
"""
from pathlib import Path
import logging
import sys

import s3fs
from dask.distributed import Client, LocalCluster

from pixel_patrol_base import api
from pixel_patrol_base.plugins.sources.manifest_source import ManifestSource

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cytodata_slurm")

# ── What to process ──────────────────────────────────────────────────────────
DATASET_ROOT = "cellpainting-gallery/cpg0036-EU-OS-bioactives"   # S3 key (no scheme)
SITES = ["FMP", "IMTM", "MEDINA", "USC"]
MAX_MANIFESTS = None          # cap number of manifests (plates); None = every plate
SAMPLE_ROWS_PER_MANIFEST = 2  # rows (fields) per manifest; None = all rows
# Combine the per-channel files of each field into one multi-channel (C,Y,X) image
# so the report has one row per field with per-channel breakdown, instead of a
# separate image per channel. False = one image per channel file.
COMBINE_CHANNELS = True
# fetch_sizes adds file size_bytes to the report via one S3 HEAD per image.
# ManifestSource fetches them concurrently (see SIZE_WORKERS), so it is no longer a
# throughput bottleneck; set False only if you want to skip the HEADs entirely.
FETCH_SIZES = True
SIZE_WORKERS = 64          # concurrent HEADs for fetch_sizes (also sets the S3 connection pool)

# ── Cluster ──────────────────────────────────────────────────────────────────
# Loading is I/O-bound (waiting on S3), so oversubscribe cores: more workers than
# CPUs keeps reads overlapping. Raise until img/s stops improving (network/S3 cap).
N_WORKERS = 16             # local-cluster worker count (ignored when USE_SLURM)
USE_SLURM = False          # True: submit Dask workers as SLURM jobs (dask-jobqueue)
SLURM_JOBS = 4             # number of SLURM jobs; total workers = SLURM_JOBS x SLURM_PROCESSES
SLURM_QUEUE = "cpu"        # partition/queue name - set to your cluster's
SLURM_CORES = 8            # cores per SLURM job
SLURM_PROCESSES = 1        # dask worker PROCESSES per job. dask-jobqueue defaults this to cores,
                           #   which is why N jobs gave 4x workers; 1 = one worker per job (SLURM_CORES threads).
SLURM_MEMORY = "32GB"      # memory per SLURM job
SLURM_WALLTIME = "02:00:00"
SLURM_ACCOUNT = None       # --account, if your cluster requires one (sbatch rejects the job otherwise)
SLURM_INTERFACE = None     # network interface for scheduler+workers, e.g. "ib0"; set if workers start but never connect
SLURM_LOG_DIR = "dask-worker-logs"  # worker stdout/err land here - READ THESE when no workers appear
MIN_WORKERS = 1            # wait for >= this many SLURM workers before starting (jobs start from the queue)
WORKER_WAIT_SECONDS = 300  # ... up to this long, then proceed; more workers join as their jobs start and get used
# Extra shell setup on the node. Usually unneeded: workers are launched with this
# process's own interpreter (python=sys.executable below), so the env is inherited
# without `activate` - which typically does nothing in a non-interactive batch job.
SLURM_PROLOGUE = []

OUTPUT = Path("out/cytodata_all_batches.parquet")


def discover_manifests(fs: s3fs.S3FileSystem) -> list[str]:
    """Return s3:// URLs of every load_data.csv under each site's load_data_csv/."""
    manifests: list[str] = []
    for site in SITES:
        pattern = f"{DATASET_ROOT}/{site}/workspace/load_data_csv/**/load_data.csv"
        keys = fs.glob(pattern)
        logger.info("Site %s: found %d manifests", site, len(keys))
        manifests.extend(f"s3://{key}" for key in keys)
    return sorted(manifests)


def site_from_path(manifest_url: str) -> dict:
    """Derive the imaging site (FMP/IMTM/...) and batch from a manifest's S3 path.

    .../cpg0036-EU-OS-bioactives/<SITE>/workspace/load_data_csv/<BATCH>/<PLATE>/load_data.csv

    Named Metadata_ImagingSite, NOT Metadata_Site: the manifest already has a
    Metadata_Site column (Cell Painting's field-of-view number, 1-9), and a
    manifest's own columns take precedence over path-derived ones.
    """
    parts = manifest_url.split("/")
    try:
        i = parts.index("cpg0036-EU-OS-bioactives")
        return {"Metadata_ImagingSite": parts[i + 1], "Metadata_BatchDir": parts[i + 4]}
    except (ValueError, IndexError):
        return {}


def make_cluster():
    if not USE_SLURM:
        logger.info("Local cluster: %d workers (set USE_SLURM=True to submit to SLURM)", N_WORKERS)
        return LocalCluster(n_workers=N_WORKERS, threads_per_worker=1)

    from dask_jobqueue import SLURMCluster
    kwargs = dict(
        queue=SLURM_QUEUE,
        cores=SLURM_CORES,
        processes=SLURM_PROCESSES,
        memory=SLURM_MEMORY,
        walltime=SLURM_WALLTIME,
        # Launch workers with THIS interpreter, so the env is inherited without
        # shell activation on the node (assumes the env is on a shared filesystem).
        python=sys.executable,
        log_directory=SLURM_LOG_DIR,
        job_script_prologue=SLURM_PROLOGUE,
    )
    if SLURM_ACCOUNT:
        kwargs["account"] = SLURM_ACCOUNT
    if SLURM_INTERFACE:
        kwargs["interface"] = SLURM_INTERFACE

    cluster = SLURMCluster(**kwargs)
    # Print the exact sbatch script - the fastest way to see why workers don't start.
    logger.info("Submitting %d SLURM job(s). Worker job script:\n%s", SLURM_JOBS, cluster.job_script())
    cluster.scale(jobs=SLURM_JOBS)
    return cluster


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fs = s3fs.S3FileSystem(anon=True)

    manifests = discover_manifests(fs)
    if MAX_MANIFESTS is not None:
        manifests = manifests[:MAX_MANIFESTS]
    logger.info("Processing %d manifests", len(manifests))

    source = ManifestSource(
        max_rows=SAMPLE_ROWS_PER_MANIFEST,
        path_metadata=site_from_path,   # site/batch from the S3 path -> Metadata_*
        fetch_sizes=FETCH_SIZES,
        size_workers=SIZE_WORKERS,      # concurrent HEADs so fetch_sizes is not a bottleneck
        combine_channels=COMBINE_CHANNELS,  # one multi-channel image per field
    )

    # Creating a Client makes Pixel Patrol reuse this cluster (SLURM or local)
    # instead of spawning its own process pool.
    cluster = make_cluster()
    with Client(cluster) as client:
        logger.info("Dask dashboard: %s", client.dashboard_link)
        if USE_SLURM:
            # SLURM jobs start from the queue, so wait before processing - starting
            # with 0 workers would submit nothing. Late-joining workers are still
            # picked up (the pipeline grows its in-flight window as they connect).
            logger.info("Waiting up to %ds for >= %d worker(s)...", WORKER_WAIT_SECONDS, MIN_WORKERS)
            try:
                client.wait_for_workers(MIN_WORKERS, timeout=WORKER_WAIT_SECONDS)
            except TimeoutError:
                pass
            logger.info("%d worker(s) online; more may join as jobs start.",
                        len(client.scheduler_info()["workers"]))
        project = api.create_project(
            "cytodata-all-batches",
            base_dir=OUTPUT.parent,
            loader="bioio",
            output_path=OUTPUT,
            source=source,
        )
        api.add_paths(project, manifests)
        api.process_files(
            project,
            description="cpg0036-EU-OS-bioactives, all sites/batches, streamed from the AWS Cell Painting Gallery.",
        )

    logger.info("Done -> %s", OUTPUT)
    logger.info("View with: pixel-patrol view %s", OUTPUT)


if __name__ == "__main__":
    main()
