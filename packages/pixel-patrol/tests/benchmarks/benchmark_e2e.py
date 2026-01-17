"""
End-to-end benchmark for pixel-patrol.
Tests full pipeline: process -> export -> import -> widget.
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


# ============================================================================
# E2E Cycle
# ============================================================================

def run_e2e_cycle(
    config: DatasetConfig,
    images_dir: Path,
    iteration: int,
    writer: E2EResultsWriter,
    branch: Optional[str],
    save: bool,
    expected_worktree: Optional[Path] = None,
) -> None:
    """Run one complete E2E cycle: process -> export -> import -> widget."""
    from pixel_patrol_base import api
    from pixel_patrol_base.core.project_settings import Settings

    # Validate imports are from expected location (only on first iteration to avoid spam)
    if iteration == 1 and expected_worktree is not None:
        print(f"  [Validate] Checking imports are from worktree...")
        if not validate_import_source("pixel_patrol_base", expected_worktree):
            raise RuntimeError(
                f"Import validation failed: pixel_patrol_base not loaded from "
                f"expected worktree {expected_worktree}. Branch comparison would be invalid."
            )
        print(f"  [Validate] ✓ Imports verified from correct branch")

    gc.collect()

    # --- Process ---
    project = api.create_project(
        f"bench_{config.name}_{iteration}",
        base_dir=images_dir,
        loader=DEFAULT_LOADER,
    )
    project.set_settings(Settings(selected_file_extensions={"tif"}))

    t0 = time.perf_counter()
    project.process_records()
    t_process = time.perf_counter() - t0

    if project.records_df is None or project.records_df.is_empty():
        logger.error(f"No records processed for {config.name} iteration {iteration}")
        if save:
            print(f"    Run {iteration}: FAILED - no records processed")
            writer.write_result(
                config=config,
                branch=branch,
                iteration=iteration,
                processing_time=t_process,
                import_time=float('inf'),
                widget_time=float('inf'),
                n_records=0,
                n_columns=0,
            )
        return

    n_records = project.records_df.height
    n_columns = len(project.records_df.columns)

    # --- Widget ---
    t_widget = _measure_widget_load(project)

    # --- Export/Import ---
    # Note: _measure_import needs to export first, so we pass the project
    # but it should NOT delete the project as we may need it for reporting
    t_import = _measure_import(project)

    # --- Record results ---
    if save:
        print(f"    Run {iteration}: Process={t_process:.3f}s | Import={t_import:.3f}s | Widget={t_widget:.3f}s")
        writer.write_result(
            config=config,
            branch=branch,
            iteration=iteration,
            processing_time=t_process,
            import_time=t_import,
            widget_time=t_widget,
            n_records=n_records,
            n_columns=n_columns,
        )


def _measure_widget_load(project) -> float:
    """Measure widget load time."""
    from pixel_patrol_base.report.dashboard_app import create_app

    try:
        t0 = time.perf_counter()
        app = create_app(project)
        _ = app.layout
        elapsed = time.perf_counter() - t0
        del app
        return elapsed
    except (ImportError, AttributeError, RuntimeError, ValueError) as e:
        logger.warning(f"Widget load failed: {e}")
        return float('inf')


def _measure_import(project) -> float:
    """
    Measure export/import cycle time (only import is timed).

    Note: This function exports the project to a temp file, then times
    the import. The original project object is NOT modified or deleted.
    """
    from pixel_patrol_base import api

    zip_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
            zip_path = Path(tmp.name)

        # Export the project (don't delete the original)
        api.export_project(project, zip_path)

        # Force garbage collection before timing import
        gc.collect()

        # Time only the import
        t0 = time.perf_counter()
        _ = api.import_project(zip_path)
        elapsed = time.perf_counter() - t0

        gc.collect()

        return elapsed

    except (OSError, IOError, RuntimeError, ValueError) as e:
        logger.warning(f"Import test failed: {e}")
        return float('inf')

    finally:
        if zip_path is not None:
            try:
                if zip_path.exists():
                    zip_path.unlink()
            except OSError as e:
                logger.warning(f"Failed to delete temp zip {zip_path}: {e}")


# ============================================================================
# Scenario Runner
# ============================================================================

def run_scenario(
    config: DatasetConfig,
    writer: E2EResultsWriter,
    branch: Optional[str],
    dataset_dir: Optional[Path],
    tmp_dir: Path,
    expected_worktree: Optional[Path] = None,
) -> None:
    """Run a complete benchmark scenario for one configuration."""
    print_scenario_header(config, branch)

    # Setup dataset
    if dataset_dir and (dataset_dir / config.name).exists():
        images_dir = dataset_dir / config.name
        print(f"  [Setup] Using existing dataset: {images_dir}")
    else:
        print(f"  [Setup] Generating dataset...")
        images_dir = generate_dataset(tmp_dir, config)

    # Warmup
    print(f"  [Warmup] Running {WARMUP_ITERATIONS} warmup cycle(s)...")
    for i in range(WARMUP_ITERATIONS):
        run_e2e_cycle(config, images_dir, i+1, writer, branch, save=False,
                      expected_worktree=expected_worktree)

    # Measure
    print(f"  [Measure] Running {TEST_ITERATIONS} test iteration(s)...")
    for i in range(TEST_ITERATIONS):
        run_e2e_cycle(config, images_dir, i+1, writer, branch, save=True,
                      expected_worktree=expected_worktree)


# ============================================================================
# Main Benchmark
# ============================================================================

def run_benchmark(mode: str, branch: Optional[str], dataset_dir: Optional[Path]) -> None:
    """Run the E2E benchmark."""
    repo_root = get_repo_root()
    current_branch = get_current_branch(repo_root)

    configs = get_dataset_configs(mode)
    writer = E2EResultsWriter()

    print_header("E2E", mode, branch)

    with tempfile.TemporaryDirectory(prefix="bench_e2e_") as tmp:
        tmp_dir = Path(tmp)

        # Current branch
        for config in configs:
            run_scenario(config, writer, current_branch, dataset_dir, tmp_dir)

        # Comparison branch (if specified)
        if branch:
            run_branch_comparison(
                repo_root=repo_root,
                branch=branch,
                configs=configs,
                run_scenario_fn=run_scenario,
                writer=writer,
                dataset_dir=dataset_dir,
                tmp_dir=tmp_dir,
            )

    print_footer()


# ============================================================================
# CLI
# ============================================================================

def main() -> int:
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="End-to-end benchmark for pixel-patrol")
    add_common_arguments(parser)
    args = parser.parse_args()

    run_benchmark(
        mode=args.mode,
        branch=args.branch,
        dataset_dir=args.dataset_dir,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())