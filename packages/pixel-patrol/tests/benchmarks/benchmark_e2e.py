"""
End-to-end benchmark — full pipeline: process → viewer widget load.

Usage:
    python benchmark_e2e.py                  # quick mode
    python benchmark_e2e.py --mode full
    python benchmark_e2e.py --branch main    # compare against another branch
    python benchmark_e2e.py --dataset-dir /path/to/datasets

Results saved to results/e2e_<timestamp>.csv.
"""
import argparse
import gc
import logging
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

from utils import (
    DatasetConfig,
    E2EResultsWriter,
    WARMUP_ITERATIONS,
    TEST_ITERATIONS,
    DEFAULT_LOADER,
    get_dataset_configs,
    generate_dataset,
    get_repo_root,
    run_branch_comparison,
    get_current_branch,
    add_common_arguments,
    print_header,
    print_scenario_header,
    print_footer,
    validate_import_source,
)

logger = logging.getLogger("benchmark.e2e")


def run_e2e_cycle(
    config:            DatasetConfig,
    images_dir:        Path,
    iteration:         int,
    writer:            E2EResultsWriter,
    branch:            Optional[str],
    save:              bool,
    expected_worktree: Optional[Path] = None,
) -> None:
    """Run one complete E2E cycle: process → widget load."""
    from pixel_patrol_base import api
    from pixel_patrol_base.core.processing_config import ProcessingConfig

    if iteration == 1 and expected_worktree is not None:
        if not validate_import_source("pixel_patrol_base", expected_worktree):
            raise RuntimeError(
                f"Import validation failed: pixel_patrol_base not loaded from {expected_worktree}"
            )

    gc.collect()

    project = api.create_project(
        f"bench_{config.name}_{iteration}",
        base_dir=images_dir,
        loader=DEFAULT_LOADER,
    )

    t0 = time.perf_counter()
    project.process_records(ProcessingConfig(selected_file_extensions={"tif"}))
    t_process = time.perf_counter() - t0

    n_records = 0
    n_columns = 0
    if project.records_df is not None and not project.records_df.is_empty():
        n_records = project.records_df.height
        n_columns = len(project.records_df.columns)
    elif project.output_path and project.output_path.exists():
        import polars as pl
        df = pl.scan_parquet(project.output_path).head(0).collect()
        n_columns = len(df.columns)
        n_records = pl.scan_parquet(project.output_path).select(pl.len()).collect().item()

    if n_records == 0:
        logger.error("No records processed for %s iteration %d", config.name, iteration)
        if save:
            writer.write_result(config=config, branch=branch, iteration=iteration,
                                processing_time=t_process, widget_time=float("inf"),
                                n_records=0, n_columns=0)
        return

    t_widget = _measure_widget_load(project)

    if save:
        print(f"    Run {iteration}: process={t_process:.3f}s  widget={t_widget:.3f}s  "
              f"records={n_records}")
        writer.write_result(config=config, branch=branch, iteration=iteration,
                            processing_time=t_process, widget_time=t_widget,
                            n_records=n_records, n_columns=n_columns)


def _measure_widget_load(project) -> float:
    from pixel_patrol_base.report.dashboard_app import create_app
    try:
        t0 = time.perf_counter()
        app = create_app(project)
        _ = app.layout
        elapsed = time.perf_counter() - t0
        del app
        return elapsed
    except Exception as exc:
        logger.warning("Widget load failed: %s", exc)
        return float("inf")


def run_scenario(
    config:            DatasetConfig,
    writer:            E2EResultsWriter,
    branch:            Optional[str],
    dataset_dir:       Optional[Path],
    tmp_dir:           Path,
    expected_worktree: Optional[Path] = None,
) -> None:
    print_scenario_header(config, branch)

    images_dir = (
        dataset_dir / config.name
        if dataset_dir and (dataset_dir / config.name).exists()
        else generate_dataset(tmp_dir, config)
    )

    for i in range(WARMUP_ITERATIONS):
        run_e2e_cycle(config, images_dir, i + 1, writer, branch, save=False,
                      expected_worktree=expected_worktree)

    for i in range(TEST_ITERATIONS):
        run_e2e_cycle(config, images_dir, i + 1, writer, branch, save=True,
                      expected_worktree=expected_worktree)


def run_benchmark(mode: str, branch: Optional[str], dataset_dir: Optional[Path]) -> None:
    repo_root      = get_repo_root()
    current_branch = get_current_branch(repo_root)
    configs        = get_dataset_configs(mode)
    writer         = E2EResultsWriter()

    print_header("E2E", mode, branch)

    with tempfile.TemporaryDirectory(prefix="bench_e2e_") as tmp:
        tmp_dir = Path(tmp)
        for config in configs:
            run_scenario(config, writer, current_branch, dataset_dir, tmp_dir)
        if branch:
            run_branch_comparison(
                repo_root=repo_root, branch=branch, configs=configs,
                run_scenario_fn=run_scenario, writer=writer,
                dataset_dir=dataset_dir, tmp_dir=tmp_dir,
            )

    print_footer()


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_arguments(parser)
    args = parser.parse_args()
    run_benchmark(mode=args.mode, branch=args.branch, dataset_dir=args.dataset_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
