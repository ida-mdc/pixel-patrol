"""Mean intensity profiles along X and Y, e.g. to spot dead sensor rows or columns at the image borders."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from pixel_patrol_base.core.contracts import ChunkKind
from pixel_patrol_base.core.record import Record
from pixel_patrol_base.core.specs import RecordSpec

PROFILE_AXES = ("X", "Y")


def get_profile_column(axis: str) -> str:
    return f"{axis.lower()}_profile_mean"


def get_profile_count_key(axis: str) -> str:
    """Private leaf-row key: read by the aggregation only, never written to the table."""
    return f"_{axis.lower()}_profile_count"


def compute_axis_profile(arr: np.ndarray, axis_index: int) -> tuple[np.ndarray, np.ndarray]:
    """Mean over all axes except axis_index, ignoring NaN/Inf; returns (means, finite counts) per coordinate."""
    other_axes = tuple(index for index in range(arr.ndim) if index != axis_index)
    if np.issubdtype(arr.dtype, np.floating):
        finite = np.isfinite(arr)
        sums = np.sum(arr, axis=other_axes, where=finite, dtype=np.float64)
        counts = np.sum(finite, axis=other_axes, dtype=np.int64)
    else:
        sums = np.sum(arr, axis=other_axes, dtype=np.float64)
        counts = np.full(arr.shape[axis_index], arr.size // arr.shape[axis_index], dtype=np.int64)
    with np.errstate(invalid="ignore", divide="ignore"):
        means = sums / counts
    return means, counts


def merge_axis_profiles(rows: list[dict[str, Any]], axis: str) -> np.ndarray | None:
    """Count-weighted mean profile over leaf rows, each placed at its global offset (dim_x / dim_y) along the axis.

    Leaves may cover only part of the axis (memory chunks split along X/Y, or a user slice_size on X/Y); every
    coordinate is averaged over the leaves that contain it.
    """
    column = get_profile_column(axis)
    count_key = get_profile_count_key(axis)
    offset_key = f"dim_{axis.lower()}"
    rows = [row for row in rows if column in row and count_key in row]
    if not rows:
        return None
    starts = [int(row.get(offset_key) or 0) for row in rows]
    origin = min(starts)
    stop = max(start + len(row[column]) for start, row in zip(starts, rows))
    sums = np.zeros(stop - origin, dtype=np.float64)
    counts = np.zeros(stop - origin, dtype=np.int64)
    for start, row in zip(starts, rows):
        means = np.asarray(row[column], dtype=np.float64)
        row_counts = np.asarray(row[count_key], dtype=np.int64)
        window = slice(start - origin, start - origin + len(means))
        with np.errstate(invalid="ignore"):
            sums[window] += np.where(row_counts > 0, means * row_counts, 0.0)
        counts[window] += row_counts
    with np.errstate(invalid="ignore", divide="ignore"):
        return (sums / counts).astype(np.float32)


class AxisProfileProcessor:
    """Per-column (X) and per-row (Y) mean intensity, averaged over all other dimensions of the image.

    Each leaf block contributes its profile means and finite-pixel counts; the aggregation adds them up at their
    global X/Y offsets, so no slicing along X or Y is needed. The profiles are only stored in the global row
    (obs_level 0), not per slice.
    """

    NAME = "axis-profile"
    DESCRIPTION = (
        "Computes mean intensity profiles along X (per column) and Y (per row), averaged over all other "
        "dimensions, e.g. to spot dead sensor rows or columns at the image borders."
    )
    CHUNK_KIND = ChunkKind.LEAF
    INPUT = RecordSpec(axes={"X", "Y"}, kinds={"intensity"})
    OUTPUT = "features"

    OUTPUT_SCHEMA: dict[str, Any] = {
        get_profile_column("X"): np.ndarray,  # x_profile_mean
        get_profile_column("Y"): np.ndarray,  # y_profile_mean
    }
    OUTPUT_SCHEMA_DESCRIPTIONS: dict[str, str] = {
        get_profile_column("X"): (
            "Mean intensity of every column (one value per X coordinate), averaged over Y and all other dimensions "
            "(NaN/Inf ignored). Dead or hot columns show up as dips or spikes. Only per image, not per slice."
        ),
        get_profile_column("Y"): (
            "Mean intensity of every row (one value per Y coordinate), averaged over X and all other dimensions "
            "(NaN/Inf ignored). Dead or hot rows show up as dips or spikes. Only per image, not per slice."
        ),
    }
    OUTPUT_SCHEMA_PATTERNS: list[tuple] = []

    def run_chunk(self, record: Record) -> dict[str, Any]:
        arr = record.data.compute() if hasattr(record.data, "compute") else np.asarray(record.data)
        if arr.size == 0:
            return {}
        row: dict[str, Any] = {}
        for axis in PROFILE_AXES:
            axis_idx = record.dim_order.index(axis)
            means, counts = compute_axis_profile(arr, axis_idx)
            row[get_profile_column(axis)] = means
            row[get_profile_count_key(axis)] = counts
        return row

    def get_aggregation(self, name: str) -> Callable[[list[dict[str, Any]], dict[str, Any]], Any] | None:
        axis = next((axis for axis in PROFILE_AXES if get_profile_column(axis) == name), None)
        if axis is None:
            return None
        return lambda rows, group_dims: None if group_dims else merge_axis_profiles(rows, axis)
