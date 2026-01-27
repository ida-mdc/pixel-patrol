"""
Shared utilities for pixel-patrol benchmarks.
Contains dataset generation, git helpers, and results writing.
"""
import csv
import logging
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Callable, Iterable
import sys
from contextlib import contextmanager
import os

import numpy as np
import tifffile

logger = logging.getLogger("benchmark")


# ============================================================================
# Constants
# ============================================================================

WARMUP_ITERATIONS = 1
TEST_ITERATIONS = 3
DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent / "results"
DEFAULT_LOADER = "bioio"


# ============================================================================
# Dataset Configuration
# ============================================================================

@dataclass
class DatasetConfig:
    """Configuration for a benchmark dataset."""
    name: str
    num_files: int
    t: int
    c: int
    z: int
    y: int
    x: int


def _build_quick_configs() -> List[DatasetConfig]:
    """Build quick mode configurations."""
    return [
        DatasetConfig("quick_smallest", 1, 1, 1, 1, 5, 5),
        DatasetConfig("quick_xy_small", 1, 1, 3, 1, 10, 10),
        DatasetConfig("quick_xy_medium", 1, 1, 3, 1, 20, 20),
        DatasetConfig("quick_xy_large", 1, 1, 3, 1, 30, 30),
        DatasetConfig("quick_tz_medium", 1, 3, 3, 3, 5, 5),
        DatasetConfig("quick_tz_large", 1, 5, 3, 5, 5, 5),
    ]


def _build_full_configs() -> List[DatasetConfig]:
    """Build full mode configurations."""
    configs = []

    # Scaling by number of files
    for num_files in [10, 20, 30, 40, 50]:
        configs.append(
            DatasetConfig(f"scaling_files_{num_files}", num_files, 1, 1, 1, 5, 5)
        )

    # Scaling by XY dimensions
    for xy_size in [10, 20, 30, 40, 50]:
        configs.append(
            DatasetConfig(f"scaling_xy_{xy_size}", 10, 1, 3, 1, xy_size, xy_size)
        )

    # Scaling by TZ dimensions
    for tz_size in [2, 3, 4, 5, 6]:
        configs.append(
            DatasetConfig(f"scaling_tz_{tz_size}", 10, tz_size, 3, tz_size, 5, 5)
        )

    return configs


QUICK_CONFIGS = _build_quick_configs()
FULL_CONFIGS = _build_full_configs()


def get_dataset_configs(mode: str) -> List[DatasetConfig]:
    """Get dataset configurations for the specified mode."""
    if mode == "quick":
        return QUICK_CONFIGS
    elif mode == "full":
        return FULL_CONFIGS
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'quick' or 'full'.")


# ============================================================================
# Dataset Generation
# ============================================================================

def create_synthetic_tiff(
    file_path: Path,
    t: int,
    c: int,
    z: int,
    y: int,
    x: int,
    seed: int,
    dtype=np.uint8,
) -> None:
    """Create a synthetic TIFF file with deterministic random data."""
    shape = (t, c, z, y, x)
    rng = np.random.RandomState(seed)
    data = rng.randint(0, 256, size=shape, dtype=dtype)
    tifffile.imwrite(str(file_path), data, photometric='minisblack')


def generate_dataset(output_dir: Path, config: DatasetConfig, base_seed: int = 42) -> Path:
    """
    Generate a dataset for a single configuration.

    Returns the path to the generated dataset directory.

    Raises:
        OSError: If dataset generation fails.
    """
    dataset_dir = output_dir / config.name

    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)

    dataset_dir.mkdir(parents=True, exist_ok=True)

    generated_files = []
    try:
        for i in range(config.num_files):
            file_path = dataset_dir / f"img_{i:04d}.tif"
            create_synthetic_tiff(
                file_path,
                config.t,
                config.c,
                config.z,
                config.y,
                config.x,
                seed=base_seed + i,
            )
            generated_files.append(file_path)
    except Exception as e:
        # Clean up partial dataset on failure
        logger.error(f"Dataset generation failed after {len(generated_files)} files: {e}")
        if dataset_dir.exists():
            shutil.rmtree(dataset_dir)
        raise OSError(f"Failed to generate dataset '{config.name}': {e}") from e

    logger.info(f"Generated {len(generated_files)} files in {dataset_dir}")
    return dataset_dir


# ============================================================================
# Git Helpers
# ============================================================================

def get_current_branch(repo_root: Path) -> Optional[str]:
    """Get the current git branch name."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def create_worktree(repo_root: Path, branch: str) -> Path:
    """
    Create a git worktree for the specified branch.

    Returns repo_root if already on that branch, otherwise creates a temp worktree.
    """
    current = get_current_branch(repo_root)
    if current == branch:
        logger.info(f"Already on branch {branch}, using current directory")
        return repo_root

    tmpdir = Path(tempfile.mkdtemp(prefix=f"bench_{branch.replace('/', '_')}_"))

    result = subprocess.run(
        ["git", "worktree", "add", "--detach", str(tmpdir), branch],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to create worktree for branch {branch}: {result.stderr}"
        )

    logger.info(f"Created worktree for branch {branch} at {tmpdir}")
    return tmpdir


# Default package prefixes to clear when switching branches
# Note: This is a fallback - prefer dynamic clearing based on source paths
DEFAULT_PACKAGE_PREFIXES = (
    "pixel_patrol",
)


def _get_modules_from_paths(src_paths: List[str]) -> List[str]:
    """
    Find all loaded modules that originate from the given source paths.

    This dynamically determines which modules to clear based on their
    actual file locations, rather than hardcoding package names.
    """
    modules_to_clear = []
    src_paths_resolved = [Path(p).resolve() for p in src_paths]

    for name, mod in list(sys.modules.items()):
        if mod is None:
            continue
        if not hasattr(mod, '__file__') or mod.__file__ is None:
            continue

        try:
            mod_path = Path(mod.__file__).resolve()
            for src_path in src_paths_resolved:
                try:
                    mod_path.relative_to(src_path)
                    modules_to_clear.append(name)
                    break
                except ValueError:
                    continue
        except (OSError, TypeError):
            continue

    return modules_to_clear


def _clear_modules_for_worktree(worktree: Path, package_prefixes: Iterable[str]) -> List[str]:
    """
    Clear cached modules that should be reloaded from the worktree.

    Uses two strategies:
    1. Clear modules matching package prefixes (fallback)
    2. Clear modules whose __file__ is under the OLD source paths (before worktree switch)

    Returns list of cleared module names for logging.
    """
    cleared = []
    prefix_tuple = tuple(package_prefixes)

    # Strategy 1: Clear by prefix
    for name in list(sys.modules):
        if any(name == p or name.startswith(p + ".") for p in prefix_tuple):
            del sys.modules[name]
            cleared.append(name)

    # Strategy 2: Clear any remaining modules from src paths NOT in the worktree
    # This catches transitive dependencies we might have missed
    worktree_resolved = worktree.resolve()
    for name, mod in list(sys.modules.items()):
        if name in cleared:
            continue
        if mod is None or not hasattr(mod, '__file__') or mod.__file__ is None:
            continue

        try:
            mod_path = Path(mod.__file__).resolve()
            # Check if module is from a "src" directory but NOT from this worktree
            if '/src/' in str(mod_path) or '\\src\\' in str(mod_path):
                try:
                    mod_path.relative_to(worktree_resolved)
                    # Module IS in worktree, keep it
                except ValueError:
                    # Module is NOT in worktree but is from some src/ path
                    # This might be from the other branch - clear it
                    del sys.modules[name]
                    cleared.append(name)
        except (OSError, TypeError):
            continue

    return cleared


@contextmanager
def use_worktree_python_paths(
    worktree: Path,
    package_prefixes: Iterable[str] = DEFAULT_PACKAGE_PREFIXES,
):
    """
    Context manager to temporarily use Python paths from a worktree.

    Clears cached imports for specified packages to ensure fresh imports
    from the worktree source paths.

    Args:
        worktree: Path to the git worktree
        package_prefixes: Package prefixes to clear from sys.modules (fallback)
    """
    old_sys_path = list(sys.path)
    old_pp = os.environ.get("PYTHONPATH")

    src_paths = find_python_src_paths(worktree)

    if not src_paths:
        logger.warning(f"No Python source paths found in worktree: {worktree}")

    try:
        if src_paths:
            sys.path[:0] = src_paths

            prefix = os.pathsep.join(src_paths)
            os.environ["PYTHONPATH"] = prefix + (os.pathsep + old_pp if old_pp else "")

            # Clear cached imports - uses both prefix matching and path-based detection
            cleared_modules = _clear_modules_for_worktree(worktree, package_prefixes)

            if cleared_modules:
                logger.info(f"Cleared {len(cleared_modules)} cached modules for worktree switch")
                logger.debug(f"Cleared modules: {cleared_modules}")

        yield

    finally:
        sys.path[:] = old_sys_path
        if old_pp is None:
            os.environ.pop("PYTHONPATH", None)
        else:
            os.environ["PYTHONPATH"] = old_pp


def validate_import_source(module_name: str, expected_worktree: Optional[Path] = None) -> bool:
    """
    Validate that a module is loaded from the expected location.

    Args:
        module_name: Name of the module to check
        expected_worktree: If provided, verify module comes from this worktree

    Returns:
        True if validation passes, False otherwise

    Logs warnings if validation fails.
    """
    if module_name not in sys.modules:
        logger.warning(f"Module '{module_name}' is not loaded - cannot validate")
        return False

    mod = sys.modules[module_name]

    if not hasattr(mod, '__file__') or mod.__file__ is None:
        logger.warning(f"Module '{module_name}' has no __file__ attribute")
        return False

    mod_path = Path(mod.__file__).resolve()
    logger.info(f"Module '{module_name}' loaded from: {mod_path}")

    if expected_worktree is not None:
        worktree_resolved = expected_worktree.resolve()
        try:
            mod_path.relative_to(worktree_resolved)
            logger.info(f"  ✓ Confirmed: module is from worktree {worktree_resolved}")
            return True
        except ValueError:
            logger.error(
                f"  ✗ MISMATCH: Module '{module_name}' loaded from {mod_path}, "
                f"but expected worktree {worktree_resolved}. "
                f"Branch comparison may be invalid!"
            )
            return False

    return True


def remove_worktree(repo_root: Path, worktree_path: Path) -> None:
    """
    Remove a git worktree (no-op if worktree_path is repo_root).

    Properly removes the worktree from git and cleans up the directory.
    """
    if worktree_path.resolve() == repo_root.resolve():
        return

    # First try to remove via git
    try:
        result = subprocess.run(
            ["git", "worktree", "remove", "-f", str(worktree_path)],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            logger.warning(f"git worktree remove failed: {result.stderr}")
            # Try to prune stale worktrees
            subprocess.run(
                ["git", "worktree", "prune"],
                cwd=str(repo_root),
                capture_output=True,
            )
    except (OSError, subprocess.SubprocessError) as e:
        logger.warning(f"Failed to remove worktree via git: {e}")

    # Ensure directory is removed even if git command failed
    if worktree_path.exists():
        try:
            shutil.rmtree(worktree_path)
            logger.debug(f"Removed worktree directory: {worktree_path}")
        except OSError as e:
            logger.error(f"Failed to remove worktree directory {worktree_path}: {e}")


def run_branch_comparison(
    repo_root: Path,
    branch: str,
    configs: list,
    run_scenario_fn: Callable,
    writer,
    dataset_dir: Optional[Path],
    tmp_dir: Path,
) -> None:
    """Run benchmark scenarios on a comparison branch via a git worktree."""
    worktree = create_worktree(repo_root, branch)
    try:
        with use_worktree_python_paths(worktree):
            for config in configs:
                run_scenario_fn(config, writer, branch, dataset_dir, tmp_dir, worktree)
    finally:
        remove_worktree(repo_root, worktree)


def find_python_src_paths(repo_root: Path) -> List[str]:
    """Find Python source paths in the repository."""
    src_paths = []

    # Look for packages/*/src
    for p in repo_root.glob("packages/*/src"):
        if p.is_dir():
            src_paths.append(str(p))

    # Look for top-level src/
    top_src = repo_root / "src"
    if top_src.exists() and top_src.is_dir():
        src_paths.append(str(top_src))

    return src_paths


def get_repo_root() -> Path:
    """
    Get the repository root directory by looking for .git directory.

    Walks up from the current file's directory until a .git directory is found.

    Raises:
        RuntimeError: If no .git directory is found.
    """
    current = Path(__file__).resolve().parent

    # Walk up the directory tree looking for .git
    for parent in [current] + list(current.parents):
        if (parent / ".git").exists():
            return parent

    # Fallback: assume fixed structure (4 levels up)
    # This maintains backwards compatibility but logs a warning
    fallback = Path(__file__).resolve().parents[4]
    logger.warning(
        f"Could not find .git directory, using fallback repo root: {fallback}. "
        f"This may be incorrect if the file structure has changed."
    )
    return fallback


# ============================================================================
# Results Writers
# ============================================================================

class ProcessorResultsWriter:
    """CSV writer for processor benchmark results."""

    def __init__(self, output_dir: Path = DEFAULT_RESULTS_DIR):
        self.csv_path = output_dir / "processor_results.csv"
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.processor_names: set = set()
        self.initialized = False

        # Clear previous results
        if self.csv_path.exists():
            self.csv_path.unlink()

    def _get_fieldnames(self) -> List[str]:
        """Get all field names including dynamic processor columns."""
        base = [
            "test_name", "branch", "iteration",
            "num_files", "t", "c", "z", "y", "x",
            "wall_time_sec", "memory_mb", "n_records", "n_columns",
        ]
        proc_cols = [f"proc_{name}_sec" for name in sorted(self.processor_names)]
        return base + proc_cols

    def _rewrite_header(self) -> None:
        """Rewrite CSV with updated header when new processors are discovered."""
        rows = []
        if self.csv_path.exists():
            with open(self.csv_path, 'r', newline='') as f:
                reader = csv.DictReader(f)
                rows = list(reader)

        fieldnames = self._get_fieldnames()
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                # Fill missing columns
                for field in fieldnames:
                    if field not in row:
                        row[field] = ""
                writer.writerow(row)

    def write_result(
        self,
        config: DatasetConfig,
        branch: Optional[str],
        iteration: int,
        wall_time: float,
        memory_bytes: int,
        n_records: int,
        n_columns: int,
        processor_times: Dict[str, float],
    ) -> None:
        """Write a processor benchmark result row."""
        # Check for new processors
        new_procs = set(processor_times.keys()) - self.processor_names
        if new_procs:
            self.processor_names.update(new_procs)
            self._rewrite_header()
            self.initialized = True
        elif not self.initialized:
            self._rewrite_header()
            self.initialized = True

        row = {
            "test_name": config.name,
            "branch": branch or "",
            "iteration": iteration,
            "num_files": config.num_files,
            "t": config.t,
            "c": config.c,
            "z": config.z,
            "y": config.y,
            "x": config.x,
            "wall_time_sec": f"{wall_time:.4f}",
            "memory_mb": f"{memory_bytes / 1024 / 1024:.2f}",
            "n_records": n_records,
            "n_columns": n_columns,
        }

        for proc_name in self.processor_names:
            col = f"proc_{proc_name}_sec"
            if proc_name in processor_times:
                row[col] = f"{processor_times[proc_name]:.4f}"
            else:
                row[col] = ""

        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self._get_fieldnames())
            writer.writerow(row)


def _get_fieldnames() -> List[str]:
    return [
        "test_name", "branch", "iteration",
        "num_files", "t", "c", "z", "y", "x",
        "processing_time_sec", "import_time_sec", "widget_time_sec",
        "n_records", "n_columns",
    ]


class E2EResultsWriter:
    """CSV writer for end-to-end benchmark results."""

    def __init__(self, output_dir: Path = DEFAULT_RESULTS_DIR):
        self.csv_path = output_dir / "e2e_results.csv"
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)

        # Clear previous results and write header
        self._write_header()

    def _write_header(self) -> None:
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=_get_fieldnames())
            writer.writeheader()

    def write_result(
        self,
        config: DatasetConfig,
        branch: Optional[str],
        iteration: int,
        processing_time: float,
        import_time: float,
        widget_time: float,
        n_records: int,
        n_columns: int,
    ) -> None:
        """Write an e2e benchmark result row."""
        row = {
            "test_name": config.name,
            "branch": branch or "",
            "iteration": iteration,
            "num_files": config.num_files,
            "t": config.t,
            "c": config.c,
            "z": config.z,
            "y": config.y,
            "x": config.x,
            "processing_time_sec": f"{processing_time:.4f}",
            "import_time_sec": f"{import_time:.4f}",
            "widget_time_sec": f"{widget_time:.4f}",
            "n_records": n_records,
            "n_columns": n_columns,
        }

        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=_get_fieldnames())
            writer.writerow(row)


# ============================================================================
# CLI Helpers
# ============================================================================

def add_common_arguments(parser) -> None:
    """Add common arguments to an argument parser."""
    parser.add_argument(
        "--mode",
        choices=["quick", "full"],
        default="quick",
        help="Benchmark mode: quick (6 tests) or full (15+ tests)",
    )
    parser.add_argument(
        "--branch",
        type=str,
        help="Compare against this branch (creates a git worktree)",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        help="Use existing dataset directory instead of generating",
    )


def print_header(benchmark_type: str, mode: str, branch: Optional[str] = None) -> None:
    """Print a standard benchmark header."""
    print(f"\n{'=' * 70}")
    print(f"{benchmark_type.upper()} BENCHMARK - {mode.upper()} MODE")
    print(f"Results: {DEFAULT_RESULTS_DIR}/")
    if branch:
        print(f"Comparing branch: {branch}")
    print(f"{'=' * 70}")


def print_scenario_header(config: DatasetConfig, branch: Optional[str] = None) -> None:
    """Print a scenario header."""
    print(f"\n{'=' * 60}")
    print(f"SCENARIO: {config.name}")
    print(f"  Files: {config.num_files}, Dims: T={config.t}, C={config.c}, Z={config.z}, Y={config.y}, X={config.x}")
    if branch:
        print(f"  Branch: {branch}")
    print(f"{'=' * 60}")


def print_footer(results_dir: Path = DEFAULT_RESULTS_DIR) -> None:
    """Print a standard benchmark footer."""
    print(f"\n{'=' * 70}")
    print(f"BENCHMARK COMPLETE")
    print(f"Results saved to: {results_dir}/")
    print(f"{'=' * 70}")