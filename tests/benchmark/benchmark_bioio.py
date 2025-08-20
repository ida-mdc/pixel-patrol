# /// script
# dependencies = [
#   "numpy",
#   "pillow",
#   "tifffile",
#   "imageio",
#   "bioio",
#   "bioio-tifffile",
#   "bioio-imageio",
#   "matplotlib",
# ]
# ///

import os
import unittest
import time
import numpy as np
from PIL import Image
import tifffile
import imageio
import bioio
import bioio_tifffile.reader
import statistics
import cProfile
import pstats
import io
import matplotlib.pyplot as plt

# --- Configuration ---
NUM_RUNS_PER_FILE = 50
PROFILE_NUM_RUNS = 100

IMAGE_SIZES = [
    (100, 100),
    (500, 500),
    (1000, 1000),
    (2000, 2000),
    (4000, 4000),
    (8000, 8000),
]

PROFILE_IMAGE_SIZE = (4000, 4000)
TEMP_IMAGE_DIR = "temp_benchmark_images"

# --- Define Readers and their associated functions ---
READERS_CONFIG = {
    "png": {
        "native_name": "imageio",
        "create_func": lambda filename, w, h: create_dummy_png(filename, w, h),
        "native_load_func": lambda fp: imageio.imread(fp),
        "bioio_load_func": lambda fp: bioio.BioImage(fp).data,
    },
    "tiff": {
        "native_name": "tifffile",
        "create_func": lambda filename, w, h: create_dummy_tiff(filename, w, h),
        "native_load_func": lambda fp: tifffile.imread(fp),
        "bioio_load_func": lambda fp: bioio.BioImage(fp, reader=bioio_tifffile.reader.Reader).data,
    },
}


# --- File Generation Functions (remain largely the same, but now called via config) ---
def create_dummy_png(filename, width, height, color=(255, 0, 0, 255)):
    """Creates a simple PNG file with the given dimensions and color."""
    img_array = np.zeros((height, width, 4), dtype=np.uint8)
    img_array[:, :, :] = color
    img = Image.fromarray(img_array)
    img.save(filename)

def create_dummy_tiff(filename, width, height, dtype=np.uint8):
    """Creates a simple grayscale TIFF file with the given dimensions."""
    img_array = np.random.randint(0, 256, size=(height, width), dtype=dtype)
    tifffile.imwrite(filename, img_array)

def setup_test_images(image_format, sizes):
    """Creates dummy images of a specified format using the configured create_func."""
    if not os.path.exists(TEMP_IMAGE_DIR):
        os.makedirs(TEMP_IMAGE_DIR)
    image_files = []
    print(f"\nGenerating dummy {image_format.upper()} files...")

    create_func = READERS_CONFIG[image_format]["create_func"]

    for i, (width, height) in enumerate(sizes):
        filename = os.path.join(TEMP_IMAGE_DIR, f"test_image_{width}x{height}.{image_format}")
        create_func(filename, width, height)
        image_files.append(filename)
        print(f"  Created {os.path.basename(filename)}")
    return image_files

def cleanup_test_images():
    """Removes the temporary image directory and its contents."""
    if os.path.exists(TEMP_IMAGE_DIR):
        import shutil
        print(f"\nCleaning up dummy image files in '{TEMP_IMAGE_DIR}'...")
        shutil.rmtree(TEMP_IMAGE_DIR)

# --- Profiling Functions (simplified) ---

def profile_generic_loading_calls(file_path, num_runs, loader_func):
    """Generic function to be profiled for any loader."""
    for _ in range(num_runs):
        _ = loader_func(file_path)

def run_profiling_session(loader_type, image_format, file_path, num_runs, output_filename_prefix):
    """
    Runs a cProfile session for a given loading function and saves the stats.
    `loader_type` can be 'native' or 'bioio'.
    """
    config = READERS_CONFIG[image_format]
    loader_func = config["native_load_func"] if loader_type == "native" else config["bioio_load_func"]

    profile_output_file = f"{output_filename_prefix}_{image_format}_profile.prof"

    print(f"\n--- Starting profiling for {output_filename_prefix.replace('_', ' ')} ({image_format.upper()}) ---")
    print(f"Profiling {os.path.basename(file_path)} with {num_runs} runs.")

    pr = cProfile.Profile()
    pr.enable()

    profile_generic_loading_calls(file_path, num_runs, loader_func) # Use generic profiler

    pr.disable()

    ps = pstats.Stats(pr)
    ps.dump_stats(profile_output_file)

    print(f"Profile saved to {profile_output_file}.")
    print(f"To visualize, run: snakeviz {profile_output_file}")
    print("--------------------------------------------------")

def run_comparative_profiling(image_format):
    print("\n" + "="*70)
    print(f"               Starting Comparative {image_format.upper()} Profiling               ")
    print("="*70)

    # Clean up any old images first and then set up
    cleanup_test_images()
    profile_filename = os.path.join(TEMP_IMAGE_DIR, f"test_image_{PROFILE_IMAGE_SIZE[0]}x{PROFILE_IMAGE_SIZE[1]}.{image_format}")
    setup_test_images(image_format, [PROFILE_IMAGE_SIZE])

    config = READERS_CONFIG[image_format]

    # Run profiling for native reader
    run_profiling_session(
        "native",
        image_format,
        profile_filename,
        PROFILE_NUM_RUNS,
        config["native_name"]
    )

    # Run profiling for bioio
    run_profiling_session(
        "bioio",
        image_format,
        profile_filename,
        PROFILE_NUM_RUNS,
        "bioio"
    )

    cleanup_test_images()

    print("\n" + "="*70)
    print("Profiling Complete. Use snakeviz to compare the .prof files:")
    print(f"  {config['native_name'].capitalize()} Comparison:")
    print(f"    snakeviz {config['native_name']}_{image_format}_profile.prof")
    print(f"    snakeviz bioio_{image_format}_profile.prof")
    print("="*70)

# --- Benchmarking Class (simplified) ---

# Global variables for benchmark data
BENCHMARK_RESULTS = {}

class TestImageLoadingSpeed(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass # Setup is now handled by run_benchmarking_suite per format

    @classmethod
    def tearDownClass(cls):
        cleanup_test_images() # Clean up at the very end
        # Report printing and plotting is now handled by the main `run_benchmarking_suite` function

    def _benchmark_loader(self, loader_func, file_path, num_runs):
        times = []
        for _ in range(num_runs):
            start_time = time.perf_counter()
            _ = loader_func(file_path)
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        return times

    def run_benchmark_for_format(self, image_format, image_files):
        config = READERS_CONFIG[image_format]
        native_reader_name = config["native_name"]
        native_loader_func = config["native_load_func"]
        bioio_loader_func = config["bioio_load_func"]

        BENCHMARK_RESULTS[image_format] = {
            native_reader_name: {},
            "bioio": {}
        }

        print(f"\n--- Benchmarking {native_reader_name} ({image_format.upper()}) ---")
        for img_file in image_files:
            file_name = os.path.basename(img_file)
            file_size_mb = os.path.getsize(img_file) / (1024 * 1024)
            print(f"Testing {native_reader_name} for {file_name} ({file_size_mb:.2f} MB)")
            times = self._benchmark_loader(native_loader_func, img_file, NUM_RUNS_PER_FILE)
            avg_time = statistics.mean(times)
            stdev_time = statistics.stdev(times) if len(times) > 1 else 0
            BENCHMARK_RESULTS[image_format][native_reader_name][file_name] = {"avg": avg_time, "stdev": stdev_time, "size_mb": file_size_mb}
            print(f"  Avg loading time over {NUM_RUNS_PER_FILE} runs: {avg_time:.6f} s (StDev: {stdev_time:.6f} s)")

        print(f"\n--- Benchmarking bioio ({image_format.upper()}) ---")
        for img_file in image_files:
            file_name = os.path.basename(img_file)
            file_size_mb = os.path.getsize(img_file) / (1024 * 1024)
            print(f"Testing bioio for {file_name} ({file_size_mb:.2f} MB)")
            times = self._benchmark_loader(bioio_loader_func, img_file, NUM_RUNS_PER_FILE)
            avg_time = statistics.mean(times)
            stdev_time = statistics.stdev(times) if len(times) > 1 else 0
            BENCHMARK_RESULTS[image_format]["bioio"][file_name] = {"avg": avg_time, "stdev": stdev_time, "size_mb": file_size_mb}
            print(f"  Avg loading time over {NUM_RUNS_PER_FILE} runs: {avg_time:.6f} s (StDev: {stdev_time:.6f} s)")

def print_comparison_report(image_format):
    print("\n" + "="*70)
    print(f"               {image_format.upper()} Loading Speed Comparison Report              ")
    print("="*70)
    print(f"Number of runs per file: {NUM_RUNS_PER_FILE}\n")

    config = READERS_CONFIG[image_format]
    native_reader_name = config["native_name"]

    header = "{:<25} | {:<10} | {:<15} | {:<15} | {:<10}"
    row_format = "{:<25} | {:<10.2f} | {:<15.6f} | {:<15.6f} | {:<9.2f}%"

    print(header.format("File Name", "Size (MB)", f"{native_reader_name} (s)", "bioio (s)", "% Higher"))
    print("-" * 80)

    results = BENCHMARK_RESULTS[image_format]
    file_names = sorted(results[native_reader_name].keys())

    for file_name in file_names:
        native_data = results[native_reader_name].get(file_name, {"avg": float('nan'), "stdev": float('nan'), "size_mb": 0})
        bioio_data = results["bioio"].get(file_name, {"avg": float('nan'), "stdev": float('nan'), "size_mb": 0})

        size_mb = native_data["size_mb"]
        native_time = native_data["avg"]
        bioio_time = bioio_data["avg"]

        percentage_higher = float('nan')
        if not np.isnan(native_time) and not np.isnan(bioio_time) and native_time > 0:
            percentage_higher = ((bioio_time - native_time) / native_time) * 100

        print(row_format.format(file_name, size_mb, native_time, bioio_time, percentage_higher))

    print("\n" + "="*70)
    print("Detailed Averages & Standard Deviations:")
    print("-" * 70)

    for reader_name, reader_results in results.items():
        print(f"\nLibrary: {reader_name.capitalize()}")
        for file_name, data in reader_results.items():
            print(f"  - {file_name} ({data['size_mb']:.2f} MB): Avg = {data['avg']:.6f} s, StDev = {data['stdev']:.6f} s")

    print("\n" + "="*70)
    print("Overall Summary:")
    print("-" * 70)

    overall_native_total = sum(data['avg'] for data in results[native_reader_name].values())
    overall_bioio_total = sum(data['avg'] for data in results["bioio"].values())

    print(f"Total average loading time across all {image_format.upper()} images ({native_reader_name.capitalize()}): {overall_native_total:.6f} s")
    print(f"Total average loading time across all {image_format.upper()} images (BioIO):    {overall_bioio_total:.6f} s")

    overall_percentage_higher = float('nan')
    if not np.isnan(overall_native_total) and not np.isnan(overall_bioio_total) and overall_native_total > 0:
        overall_percentage_higher = ((overall_bioio_total - overall_native_total) / overall_native_total) * 100

    if overall_native_total < overall_bioio_total:
        print(f"\nConclusion: BioIO ({image_format.upper()}) is slower than {native_reader_name.capitalize()} ({image_format.upper()}) by approximately {overall_percentage_higher:.2f}% (Total difference: {(overall_bioio_total - overall_native_total):.6f} s).")
    elif overall_bioio_total < overall_native_total:
        print(f"\nConclusion: BioIO ({image_format.upper()}) is faster than {native_reader_name.capitalize()} ({image_format.upper()}) by approximately {abs(overall_percentage_higher):.2f}% (Total difference: {(overall_native_total - overall_bioio_total):.6f} s).")
    else:
        print(f"\nConclusion: Both libraries performed similarly for {image_format.upper()} loading.")

    print("="*70)

def plot_speedup_factor(image_format):
    results = BENCHMARK_RESULTS.get(image_format)
    if not results:
        print(f"No benchmark results found for {image_format} to plot.")
        return

    config = READERS_CONFIG[image_format]
    native_reader_name = config["native_name"]

    image_sizes_labels = [f"{s[0]}x{s[1]}" for s in IMAGE_SIZES]
    native_times = [results[native_reader_name][f]["avg"] for f in sorted(results[native_reader_name].keys())]
    bioio_times = [results["bioio"][f]["avg"] for f in sorted(results["bioio"].keys())]

    speedup_factors = []
    for i in range(len(native_times)):
        if native_times[i] > 0 and bioio_times[i] > 0: # Ensure no division by zero or NaN issues
            speedup_factors.append(native_times[i] / bioio_times[i])
        else:
            speedup_factors.append(float('nan')) # Handle cases where a time might be zero or NaN

    x = np.arange(len(image_sizes_labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 7))

    ax.bar(x, native_times, width, label=f'{native_reader_name.capitalize()} Loading Time')
    ax.bar(x + width, bioio_times, width, label='BioIO Loading Time')

    ax.set_xlabel('Image Size')
    ax.set_ylabel('Average Loading Time (s)')
    ax.set_title(f'Average Loading Time Comparison for {image_format.upper()} Images')
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(image_sizes_labels, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'loading_times_{image_format}.png')
    plt.show()

    # Plotting speedup factor
    fig_speedup, ax_speedup = plt.subplots(figsize=(12, 7))
    ax_speedup.bar(x, speedup_factors, width=0.5, color='skyblue')
    ax_speedup.axhline(1.0, color='red', linestyle='--', label='Speedup Factor = 1 (Equal Speed)')

    ax_speedup.set_xlabel('Image Size')
    ax_speedup.set_ylabel(f'Speedup Factor ({native_reader_name.capitalize()} Time / BioIO Time)')
    ax_speedup.set_title(f'BioIO Speedup Factor vs. {native_reader_name.capitalize()} for {image_format.upper()} Images')
    ax_speedup.set_xticks(x)
    ax_speedup.set_xticklabels(image_sizes_labels, rotation=45, ha="right")
    ax_speedup.legend()
    ax_speedup.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'speedup_factor_{image_format}.png')
    plt.show()


def run_benchmarking_suite(image_format):
    print("\n" + "="*70)
    print(f"               Starting {image_format.upper()} Benchmarking Suite               ")
    print("="*70)

    # Setup images for the specific format
    image_files = setup_test_images(image_format, IMAGE_SIZES)

    # Instantiate the test class to run benchmarks directly
    tester = TestImageLoadingSpeed()
    tester.run_benchmark_for_format(image_format, image_files)

    # Print report and plot after benchmarking all files for the format
    print_comparison_report(image_format)
    plot_speedup_factor(image_format)

    # Cleanup after reporting and plotting
    cleanup_test_images()

    print("\n" + "="*70)
    print(f"               {image_format.upper()} Benchmarking Suite Complete               ")
    print("="*70)


# --- Main Execution ---
if __name__ == '__main__':
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib not found. Please install it to plot results: pip install matplotlib")
        plt = None

    # List of formats to benchmark
    formats_to_test = ["png", "tiff"]

    print("\nStarting profiling sessions...")
    for fmt in formats_to_test:
        run_comparative_profiling(fmt)

    print("\nStarting benchmarking suites...")
    for fmt in formats_to_test:
        run_benchmarking_suite(fmt)

    print("\nAll benchmarking and profiling complete!")