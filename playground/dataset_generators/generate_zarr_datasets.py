from pathlib import Path
import zarr
import numpy as np
import os
import shutil

# --- Configuration ---

# 1. Define a base directory to store all Zarr datasets
base_dir = Path(__file__).parent / "zarr_image_datasets"

# 2. Define the three groups (subfolders)
# We'll give them descriptive names for clarity.
groups = [
    "group1_v2_mixed_suffix",
    "group2_v3_with_suffix",
    "group3_mixed_format_no_suffix"
]

# --- Setup ---

# Clean up previous runs for a fresh start
if base_dir.exists():
    shutil.rmtree(base_dir)

# Create the base directory and the group subdirectories
base_dir.mkdir()
for group in groups:
    (base_dir / group).mkdir()

# --- Helper Function ---

def create_zarr_dataset(
    name: str,
    group: str,
    shape: tuple,
    dtype: str,
    chunks: tuple,
    description: str,
    zarr_format: int,
    use_suffix: bool = True
):
    """
    Helper function to create and populate a Zarr array with specific configurations.

    Args:
        name: The base name for the dataset.
        group: The subfolder to save the dataset in.
        shape: The shape of the numpy array.
        dtype: The data type of the array.
        chunks: The chunking strategy for the Zarr array.
        description: A text description of the dataset.
        zarr_format: The Zarr format version (2 or 3).
        use_suffix: If True, appends '.zarr' to the folder name.
    """
    # Conditionally add the .zarr suffix
    folder_name = f"{name}.zarr" if use_suffix else name
    path = base_dir / group / folder_name

    print(f"Creating: {description}")
    print(f"  Path: {path}")
    print(f"  Format: zarr v{zarr_format}, Suffix used: {use_suffix}")

    # Ensure chunk size is not larger than the shape itself
    effective_chunks = tuple(min(s, c) for s, c in zip(shape, chunks))

    try:
        # Open the Zarr array with the specified format
        z_array = zarr.open(
            store=path,
            mode='w',
            shape=shape,
            dtype=dtype,
            chunks=effective_chunks,
            zarr_format=zarr_format  # Specify the zarr format here
        )
        # Fill with some simple data
        if len(shape) > 0:
            if dtype == 'bool':
                z_array[:] = np.random.rand(*shape) > 0.5
            else:
                z_array[:] = (np.arange(np.prod(shape)).reshape(shape) % 256).astype(dtype)
        print(f"  Shape: {z_array.shape}, Dtype: {z_array.dtype}, Chunks: {z_array.chunks}\n")
    except Exception as e:
        print(f"  Error creating {name}: {e}\n")


# --- Dataset Creation ---

# We define all datasets in a list of dictionaries for easy management.

datasets = [
    # Group 1: All datasets will use Zarr v2 format, with mixed use of the .zarr suffix.
    {'name': "signal_1d", 'shape': (1000,), 'dtype': 'float32', 'chunks': (100,), 'desc': "1D Signal", 'group': groups[0], 'format': 2, 'suffix': True},
    {'name': "spectrum_1d", 'shape': (512,), 'dtype': 'uint16', 'chunks': (64,), 'desc': "1D Spectrum", 'group': groups[0], 'format': 2, 'suffix': False},
    {'name': "grayscale_image_2d", 'shape': (256, 256), 'dtype': 'uint8', 'chunks': (64, 64), 'desc': "2D Grayscale Image", 'group': groups[0], 'format': 2, 'suffix': True},
    {'name': "binary_mask_2d", 'shape': (128, 128), 'dtype': 'bool', 'chunks': (32, 32), 'desc': "2D Binary Mask", 'group': groups[0], 'format': 2, 'suffix': False},
    {'name': "hires_grayscale_2d", 'shape': (2048, 2048), 'dtype': 'uint16', 'chunks': (256, 256), 'desc': "2D High-Res Grayscale", 'group': groups[0], 'format': 2, 'suffix': True},

    # Group 2: All datasets will use the modern Zarr v3 format and will have the .zarr suffix.
    {'name': "volume_3d", 'shape': (64, 128, 128), 'dtype': 'uint8', 'chunks': (16, 32, 32), 'desc': "3D Volume", 'group': groups[1], 'format': 3, 'suffix': True},
    {'name': "time_2d_grayscale", 'shape': (100, 200, 200), 'dtype': 'uint8', 'chunks': (10, 50, 50), 'desc': "Time-series (T, Y, X)", 'group': groups[1], 'format': 3, 'suffix': True},
    {'name': "rgb_volume_3d", 'shape': (32, 64, 64, 3), 'dtype': 'uint8', 'chunks': (8, 16, 16, 3), 'desc': "3D RGB Volume", 'group': groups[1], 'format': 3, 'suffix': True},
    {'name': "multi_channel_3d", 'shape': (10, 50, 100, 100), 'dtype': 'uint16', 'chunks': (1, 10, 25, 25), 'desc': "Multi-channel 3D Volume", 'group': groups[1], 'format': 3, 'suffix': True},
    {'name': "time_volume_4d", 'shape': (50, 32, 64, 64), 'dtype': 'uint8', 'chunks': (5, 8, 16, 16), 'desc': "4D Time-series of Volumes", 'group': groups[1], 'format': 3, 'suffix': True},

    # Group 3: Mixed Zarr v2/v3 formats, and none will have the .zarr suffix.
    {'name': "time_rgb_4d", 'shape': (200, 100, 100, 3), 'dtype': 'uint8', 'chunks': (20, 25, 25, 3), 'desc': "4D Time-series of RGB Images", 'group': groups[2], 'format': 2, 'suffix': False},
    {'name': "multi_spectral_3d_volume", 'shape': (8, 40, 80, 80), 'dtype': 'float32', 'chunks': (1, 10, 20, 20), 'desc': "4D Multi-spectral 3D Volume", 'group': groups[2], 'format': 3, 'suffix': False},
    {'name': "time_multi_channel_3d", 'shape': (20, 5, 30, 60, 60), 'dtype': 'uint16', 'chunks': (2, 1, 5, 15, 15), 'desc': "5D Time-series, Multi-channel, 3D", 'group': groups[2], 'format': 2, 'suffix': False},
    {'name': "position_time_2d", 'shape': (5, 100, 150, 150), 'dtype': 'uint8', 'chunks': (1, 10, 30, 30), 'desc': "5D Positional Scan, Time-series, 2D", 'group': groups[2], 'format': 3, 'suffix': False},
    {'name': "tczyxp_6d", 'shape': (10, 2, 20, 40, 40, 2), 'dtype': 'uint16', 'chunks': (1, 1, 5, 10, 10, 1), 'desc': "6D Dataset: T,C,Z,Y,X,Polarization", 'group': groups[2], 'format': 2, 'suffix': False},
]

# Loop through the list and create each dataset
for params in datasets:
    create_zarr_dataset(
        name=params['name'],
        group=params['group'],
        shape=params['shape'],
        dtype=params['dtype'],
        chunks=params['chunks'],
        description=params['desc'],
        zarr_format=params['format'],
        use_suffix=params['suffix']
    )

print("--- Script finished ---")