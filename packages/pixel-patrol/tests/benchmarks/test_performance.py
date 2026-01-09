import csv
import faulthandler
import gc
import os
import shutil
import time
from pathlib import Path
from typing import List

import numpy as np
import pytest
import tifffile

# Enable crash diagnostics
faulthandler.enable()

from pixel_patrol_base import api
from pixel_patrol_base.core.project_settings import Settings
from pixel_patrol_base.report.dashboard_app import create_app

# ============================================================================
# Benchmark Configuration
# ============================================================================
WARMUP_CYCLES = 1      # Runs to discard (primes cache)
TEST_ITERATIONS = 3    # Runs to record
FILE_COUNTS = [10, 20, 30, 40, 50]
DIMENSION_SIZES_TZ = [2,3,4,5,6]
DIMENSION_SIZES_XY = [10, 20, 30, 40, 50]
DEFAULT_X, DEFAULT_Y = 5, 5

# ============================================================================
# Utilities & Helpers
# ============================================================================

def create_synthetic_tiff(
    file_path: Path, t: int, c: int, z: int, y: int, x: int, dtype=np.uint8
) -> None:
    shape = (t, c, z, y, x)
    data = np.random.randint(0, 256, size=shape, dtype=dtype)
    tifffile.imwrite(str(file_path), data, photometric='minisblack')

def generate_image_dataset(
    base_dir: Path, num_files: int, t: int, c: int, z: int, y: int, x: int
) -> List[Path]:
    """Generates the dataset once per test case."""
    if base_dir.exists():
        shutil.rmtree(base_dir)
    base_dir.mkdir(parents=True)

    files = []
    for i in range(num_files):
        p = base_dir / f"img_{i:04d}.tif"
        create_synthetic_tiff(p, t, c, z, y, x)
        files.append(p)
    return files

@pytest.fixture(scope="session")
def benchmark_csv_path() -> Path:
    """
    Initializes the results CSV.
    Crucially: DELETES the old file at the start of the session to prevent duplication.
    """
    test_dir = Path(__file__).parent
    results_dir = test_dir / "benchmark_results"
    results_dir.mkdir(exist_ok=True)
    csv_path = results_dir / "benchmark_results.csv"

    # CLEAR OLD DATA
    if csv_path.exists():
        print(f"\n[Init] Clearing previous benchmark results at {csv_path}")
        os.remove(csv_path)

    # Write Header
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "test_name", "iteration", "num_files",
            "t_size", "c_size", "z_size", "y_size", "x_size",
            "processing_time", "import_time", "widget_time",
            "n_records", "n_columns"
        ])

    return csv_path

# ============================================================================
# The Benchmark Engine
# ============================================================================

class BenchmarkRunner:
    """
    Handles the execution lifecycle: Setup -> Warmup -> Measure -> Teardown.
    """
    def __init__(self, csv_path: Path, tmp_path: Path):
        self.csv_path = csv_path
        self.tmp_path = tmp_path

    def run_scenario(
        self,
        test_name: str,
        num_files: int,
        t: int, c: int, z: int, y: int, x: int
    ):
        print(f"\n{'='*60}")
        print(f"SCENARIO: {test_name}")
        print(f"Config: {num_files} files, Dims(T={t}, C={c}, Z={z}, Y={y}, X={x})")
        print(f"{'='*60}")

        # 1. DATASET GENERATION (Once per scenario)
        images_dir = self.tmp_path / f"source_images_{test_name}"
        print(f"  [Setup] Generating {num_files} synthetic images...")
        generate_image_dataset(images_dir, num_files, t, c, z, y, x)

        # 2. WARMUP
        print(f"  [Warmup] Running {WARMUP_CYCLES} warmup cycle(s)...")
        for i in range(WARMUP_CYCLES):
            self._execute_cycle(test_name, i, images_dir, num_files, t, c, z, y, x, save=False)

        # 3. MEASUREMENT
        print(f"  [Measure] Running {TEST_ITERATIONS} test iteration(s)...")
        for i in range(TEST_ITERATIONS):
            self._execute_cycle(test_name, i+1, images_dir, num_files, t, c, z, y, x, save=True)

        # 4. CLEANUP (Optional: remove images to save disk space if storage is tight)
        # shutil.rmtree(images_dir)

    def _execute_cycle(
        self, test_name, iteration, images_dir, num_files, t, c, z, y, x, save: bool
    ):
        """Runs one complete lifecycle: Process -> Widget -> Export -> Import."""

        # Force GC to ensure clean state
        gc.collect()

        # --- A. Measure Processing ---
        # We must create a FRESH project instance to measure processing accurately
        project_name = f"proj_{test_name}_{iteration}"
        project = api.create_project(project_name, base_dir=images_dir, loader="bioio")
        project.set_settings(Settings(selected_file_extensions={"tif"}))

        t0 = time.perf_counter()
        project.process_records()
        t_process = time.perf_counter() - t0

        if project.records_df is None or project.records_df.is_empty():
            print("    [Error] No records processed!")
            return

        n_records = project.records_df.height
        n_cols = len(project.records_df.columns)

        # --- B. Measure Widget Load ---
        try:
            t0 = time.perf_counter()
            app = create_app(project)
            _ = app.layout # Trigger layout creation
            t_widget = time.perf_counter() - t0
            del app
        except Exception:
            t_widget = float('inf')

        # --- C. Measure Import (via Export) ---
        try:
            zip_path = self.tmp_path / f"export_{test_name}_{iteration}.zip"
            # We don't time export as it's not a requested metric, but we need the file
            api.export_project(project, zip_path)

            # Clear memory before import
            del project
            gc.collect()

            t0 = time.perf_counter()
            _ = api.import_project(zip_path)
            t_import = time.perf_counter() - t0

            # Cleanup zip
            if zip_path.exists():
                os.remove(zip_path)
        except Exception:
            t_import = float('inf')

        # --- Output & Save ---
        if save:
            print(f"    Run {iteration}: Proc={t_process:.3f}s | Imp={t_import:.3f}s | Wid={t_widget:.3f}s")
            self._write_result(
                test_name, iteration, num_files, t, c, z, y, x,
                t_process, t_import, t_widget, n_records, n_cols
            )

    def _write_result(self, *args):
        with open(self.csv_path, 'a', newline='') as f:
            csv.writer(f).writerow(args)

# ============================================================================
# Parameterized Tests
# ============================================================================

@pytest.mark.parametrize("num_files", FILE_COUNTS)
def test_scaling_files(num_files, benchmark_csv_path, tmp_path):
    """Scaling Test: Varying file counts (Constant Dimensions)."""
    runner = BenchmarkRunner(benchmark_csv_path, tmp_path)
    runner.run_scenario(
        test_name="scaling_files",
        num_files=num_files,
        t=1, c=1, z=1, y=DEFAULT_X, x=DEFAULT_Y
    )

@pytest.mark.parametrize("xy_size", DIMENSION_SIZES_XY)
def test_scaling_xy(xy_size, benchmark_csv_path, tmp_path):
    """Scaling Test: Varying Image Resolution (Constant File Count)."""
    runner = BenchmarkRunner(benchmark_csv_path, tmp_path)
    runner.run_scenario(
        test_name="scaling_xy",
        num_files=10,
        t=1, c=3, z=1, y=xy_size, x=xy_size
    )

@pytest.mark.parametrize("tz_size", DIMENSION_SIZES_TZ)
def test_scaling_tz(tz_size, benchmark_csv_path, tmp_path):
    """Scaling Test: Varying Time/Z Stacks (Constant File Count)."""
    runner = BenchmarkRunner(benchmark_csv_path, tmp_path)
    runner.run_scenario(
        test_name="scaling_tz",
        num_files=10,
        t=tz_size, c=3, z=tz_size, y=DEFAULT_X, x=DEFAULT_Y
    )

def test_summary(benchmark_csv_path):
    """Prints a summary of where the data is."""
    if benchmark_csv_path.exists():
        print(f"\n{'='*60}")
        print(f"BENCHMARK COMPLETE")
        print(f"Raw data saved to: {benchmark_csv_path}")
        print(f"Run your plotting script on this file to see Mean/StdDev.")
        print(f"{'='*60}")
