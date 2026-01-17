"""
Processor-level benchmark for pixel-patrol.
Tests individual processor performance with timing instrumentation.
"""
import argparse
import functools
import gc
import logging
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    psutil = None
    PSUTIL_AVAILABLE = False


from utils import (
    DatasetConfig,
    ProcessorResultsWriter,
    DEFAULT_LOADER,
    WARMUP_ITERATIONS,
    TEST_ITERATIONS,
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

logger = logging.getLogger("benchmark.processors")


# ============================================================================
# Processor Stats Tracking
# ============================================================================

@dataclass
class ProcessorStats:
    """Timing stats for a single processor."""
    calls: int = 0
    total_time: float = 0.0


@dataclass
class ProcessorStatsRegistry:
    """Registry to track processor timing stats."""
    stats: Dict[str, ProcessorStats] = field(default_factory=dict)

    def clear(self) -> None:
        self.stats.clear()

    def record(self, name: str, elapsed: float) -> None:
        if name not in self.stats:
            self.stats[name] = ProcessorStats()
        self.stats[name].calls += 1
        self.stats[name].total_time += elapsed

    def get_times(self) -> Dict[str, float]:
        return {
            name.replace("Timed_", ""): s.total_time
            for name, s in self.stats.items()
        }


# Module-level registry instance
stats_registry = ProcessorStatsRegistry()


# ============================================================================
# Processor Instrumentation
# ============================================================================

def discover_all_processor_classes() -> List:
    """Discover processor plugin classes from ALL packages via entry points."""

    from pixel_patrol_base import plugin_registry
    from pixel_patrol_base.plugin_registry import discover_plugins_from_entrypoints

    processor_classes = discover_plugins_from_entrypoints("pixel_patrol.processor_plugins")

    if processor_classes:
        logger.info(f"Discovered {len(processor_classes)} processor classes from entry points")
    else:
        logger.warning("No processor entry points found, falling back to base registry")
        processor_classes = plugin_registry.register_processor_plugins()

    return processor_classes


class TimedRunMethod:
    """Descriptor that wraps a processor's run method with timing."""

    def __init__(self, processor_name: str, original_run):
        self.processor_name = processor_name
        self.original_run = original_run

    def __call__(self, instance, *args, **kwargs):
        t0 = time.perf_counter()
        try:
            return self.original_run(instance, *args, **kwargs)
        finally:
            elapsed = time.perf_counter() - t0
            stats_registry.record(self.processor_name, elapsed)

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return functools.partial(self, obj)


def create_wrapped_processor_class(cls):
    """Create a wrapped processor class with timing instrumentation."""
    cls_name = cls.__name__
    original_run = cls.run

    attrs = {"run": TimedRunMethod(cls_name, original_run)}
    wrapped_cls = type(f"Timed_{cls_name}", (cls,), attrs)

    if hasattr(cls, "NAME"):
        wrapped_cls.NAME = cls.NAME

    return wrapped_cls


class InstrumentedRegisterFunction:
    """Callable that returns instrumented processor classes."""

    def __init__(self, processor_classes: List):
        self.processor_classes = processor_classes

    def __call__(self):
        return [create_wrapped_processor_class(cls) for cls in self.processor_classes]


# ============================================================================
# Benchmark Runner
# ============================================================================

def run_processor_cycle(
    config: DatasetConfig,
    images_dir: Path,
    iteration: int,
    writer: ProcessorResultsWriter,
    branch: Optional[str],
    save: bool,
    expected_worktree: Optional[Path] = None,
) -> None:
    """Run processor benchmark for one iteration."""

    from pixel_patrol_base import api, plugin_registry
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

    stats_registry.clear()

    # Discover and instrument processors
    all_processor_classes = discover_all_processor_classes()
    original_register = plugin_registry.register_processor_plugins
    plugin_registry.register_processor_plugins = InstrumentedRegisterFunction(all_processor_classes)

    try:
        gc.collect()

        if PSUTIL_AVAILABLE:
            mem_before = psutil.Process().memory_info().rss
        else:
            mem_before = 0

        # Create and run project
        project = api.create_project(
            f"bench_{config.name}_{iteration}",
            base_dir=images_dir,
            loader=DEFAULT_LOADER,
        )
        project.set_settings(Settings(
            selected_file_extensions={"tif"},
        ))

        t0 = time.perf_counter()
        project.process_records()
        wall_time = time.perf_counter() - t0

        if PSUTIL_AVAILABLE:
            mem_after = psutil.Process().memory_info().rss
        else:
            mem_after = 0
        memory_delta = mem_after - mem_before

        # Get results
        n_records = 0
        n_columns = 0
        if project.records_df is not None and not project.records_df.is_empty():
            n_records = project.records_df.height
            n_columns = len(project.records_df.columns)

        # Get processor times
        processor_times = stats_registry.get_times()

        # Print and write results only if save=True
        if save:
            mem_str = f"{memory_delta / 1024 / 1024:.1f}MB" if PSUTIL_AVAILABLE else "N/A"
            print(f"    Run {iteration}: Wall time: {wall_time:.3f}s | Memory: {mem_str} | Records: {n_records}")
            if processor_times:
                print(f"    [Processors]")
                for name in sorted(processor_times.keys()):
                    print(f"      {name}: {processor_times[name]:.3f}s")

            writer.write_result(
                config=config,
                branch=branch,
                iteration=iteration,
                wall_time=wall_time,
                memory_bytes=memory_delta,
                n_records=n_records,
                n_columns=n_columns,
                processor_times=processor_times,
            )

    finally:
        # Restore original registration
        plugin_registry.register_processor_plugins = original_register


def run_scenario(
    config: DatasetConfig,
    writer: ProcessorResultsWriter,
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

    # Warmup iterations
    print(f"  [Warmup] Running {WARMUP_ITERATIONS} warmup cycle(s)...")
    for i in range(WARMUP_ITERATIONS):
        run_processor_cycle(
            config, images_dir, i + 1, writer, branch,
            save=False,
            expected_worktree=expected_worktree,
        )

    # Test iterations
    print(f"  [Measure] Running {TEST_ITERATIONS} test iteration(s)...")
    for i in range(TEST_ITERATIONS):
        run_processor_cycle(
            config, images_dir, i + 1, writer, branch,
            save=True,
            expected_worktree=expected_worktree,
        )


# ============================================================================
# Main Benchmark
# ============================================================================

def run_benchmark(mode: str, branch: Optional[str], dataset_dir: Optional[Path]) -> None:
    """Run the processor benchmark."""
    # Warn if psutil is not available
    if not PSUTIL_AVAILABLE:
        logger.warning(
            "psutil is not installed - memory measurements will not be available. "
            "Install with: pip install psutil"
        )
        print("WARNING: psutil not available, memory measurements disabled")

    repo_root = get_repo_root()
    current_branch = get_current_branch(repo_root)

    configs = get_dataset_configs(mode)
    writer = ProcessorResultsWriter()

    print_header("Processor", mode, branch)

    with tempfile.TemporaryDirectory(prefix="bench_proc_") as tmp:
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

    parser = argparse.ArgumentParser(description="Processor-level benchmark for pixel-patrol")
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