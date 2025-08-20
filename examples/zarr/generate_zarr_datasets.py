from pathlib import Path

import zarr
import numpy as np
import os
import shutil

# Define a base directory to store the Zarr datasets
base_dir = Path(__file__).parent / "zarr_image_datasets"
if os.path.exists(base_dir):
    shutil.rmtree(base_dir)  # Clean up previous runs for fresh examples
os.makedirs(base_dir)


def create_zarr_dataset(name, shape, dtype, chunks, description):
    """Helper function to create and populate a Zarr array."""
    path = os.path.join(base_dir, f"{name}.zarr")
    print(f"Creating: {description} at {path}")

    # Ensure chunk size is compatible with shape
    effective_chunks = tuple(min(s, c) for s, c in zip(shape, chunks))

    try:
        z_array = zarr.open(
            path,
            mode='w',
            shape=shape,
            dtype=dtype,
            chunks=effective_chunks
        )
        # Fill with some simple data to make it a valid "image"
        if len(shape) > 0:
            if dtype == 'bool':
                z_array[:] = np.random.rand(*shape) > 0.5
            else:
                z_array[:] = np.arange(np.prod(shape)).reshape(shape).astype(dtype) % 256
        print(f"  Shape: {z_array.shape}, Dtype: {z_array.dtype}, Chunks: {z_array.chunks}")
    except Exception as e:
        print(f"  Error creating {name}: {e}")


# --- 1D Datasets ---

# 1. 1D Signal
create_zarr_dataset(
    "signal_1d", (1000,), 'float32', (100,),
    "1D Signal: A simple time-series or line scan data."
)

# 2. 1D Spectrum (e.g., spectral line)
create_zarr_dataset(
    "spectrum_1d", (512,), 'uint16', (64,),
    "1D Spectrum: Intensity values across wavelengths."
)

# --- 2D Datasets ---

# 3. Grayscale Image
create_zarr_dataset(
    "grayscale_image_2d", (256, 256), 'uint8', (64, 64),
    "2D Grayscale Image: Standard 8-bit image."
)

# 4. Binary Mask
create_zarr_dataset(
    "binary_mask_2d", (128, 128), 'bool', (32, 32),
    "2D Binary Mask: Boolean image for segmentation."
)

# 5. High-Resolution Grayscale Image
create_zarr_dataset(
    "hires_grayscale_2d", (2048, 2048), 'uint16', (256, 256),
    "2D High-Resolution Grayscale Image: 16-bit for more intensity levels."
)

# --- 3D Datasets ---

# 6. Volume (e.g., CT scan, microscopy stack)
create_zarr_dataset(
    "volume_3d", (64, 128, 128), 'uint8', (16, 32, 32),
    "3D Volume: Z-stack of grayscale images."
)

# 7. Time-series of 2D Grayscale Images
create_zarr_dataset(
    "time_2d_grayscale", (100, 200, 200), 'uint8', (10, 50, 50),
    "Time-series (T, Y, X): 100 frames of 200x200 grayscale images."
)

# 8. RGB Volume
create_zarr_dataset(
    "rgb_volume_3d", (32, 64, 64, 3), 'uint8', (8, 16, 16, 3),
    "3D Volume with RGB channels (Z, Y, X, C): Color volume data."
)

# 9. Multi-channel 3D Volume (e.g., confocal microscopy)
create_zarr_dataset(
    "multi_channel_3d", (10, 50, 100, 100), 'uint16', (1, 10, 25, 25),
    "Multi-channel 3D Volume (C, Z, Y, X): 10 channels, 50 Z-slices."
)

# --- 4D Datasets ---

# 10. Time-series of 3D Volumes
create_zarr_dataset(
    "time_volume_4d", (50, 32, 64, 64), 'uint8', (5, 8, 16, 16),
    "4D Time-series of Volumes (T, Z, Y, X)."
)

# 11. Time-series of RGB Images
create_zarr_dataset(
    "time_rgb_4d", (200, 100, 100, 3), 'uint8', (20, 25, 25, 3),
    "4D Time-series of RGB Images (T, Y, X, C)."
)

# 12. Multi-spectral 3D Volume
create_zarr_dataset(
    "multi_spectral_3d_volume", (8, 40, 80, 80), 'float32', (1, 10, 20, 20),
    "4D Multi-spectral 3D Volume (S, Z, Y, X): 8 spectral bands."
)

# --- 5D Datasets ---

# 13. Time-series of Multi-channel 3D Volumes (e.g., live-cell imaging)
create_zarr_dataset(
    "time_multi_channel_3d", (20, 5, 30, 60, 60), 'uint16', (2, 1, 5, 15, 15),
    "5D Time-series, Multi-channel, 3D (T, C, Z, Y, X)."
)

# 14. Positional Scans of Time-series 2D Images
create_zarr_dataset(
    "position_time_2d", (5, 100, 150, 150), 'uint8', (1, 10, 30, 30),
    "5D Positional Scan, Time-series, 2D (P, T, Y, X): 5 different positions."
)

# --- Higher Dimensionalities (Illustrative) ---

# 15. Time, Channel, Z, Y, X, Polarization (6D)
create_zarr_dataset(
    "tczyxp_6d", (10, 2, 20, 40, 40, 2), 'uint16', (1, 1, 5, 10, 10, 1),
    "6D Dataset: Time, Channel, Z, Y, X, Polarization."
)
