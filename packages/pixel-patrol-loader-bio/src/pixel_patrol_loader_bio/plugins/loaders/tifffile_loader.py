"""TIFF / OME-TIFF loader using tifffile + Zarr store → Dask (alternative to BioIO)."""

from __future__ import annotations

import logging
import math
import re
import string
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import dask.array as da
import numpy as np
import polars as pl
import tifffile
import zarr
from bioio_base.exceptions import UnsupportedFileFormatError

from pixel_patrol_base.core.record import record_from
from pixel_patrol_loader_bio.plugins.loaders.bioio_loader import normalize_metadata

logger = logging.getLogger(__name__)


def _page0_description(tf: tifffile.TiffFile) -> Optional[str]:
    try:
        desc = tf.pages[0].description
        if desc is None:
            return None
        if isinstance(desc, bytes):
            return desc.decode(errors="ignore")
        return str(desc)
    except Exception:
        return None


def _letters_for_axes(n: int) -> str:
    if n <= 0:
        return ""
    if n <= len(string.ascii_uppercase):
        return string.ascii_uppercase[:n]
    return "".join(string.ascii_uppercase[i % 26] for i in range(n))


def _channel_names(tf: tifffile.TiffFile, series: tifffile.TiffPageSeries, axes_u: str) -> List[str]:
    if "C" not in axes_u:
        return []

    ij = getattr(tf, "imagej_metadata", None) or {}
    labels = ij.get("Labels") or ij.get("labels")
    if isinstance(labels, (list, tuple)) and labels:
        return [str(x) for x in labels]

    ome = getattr(tf, "ome_metadata", None)
    if isinstance(ome, dict):
        ch = ome.get("Channels") or ome.get("channels")
        if isinstance(ch, list):
            names = []
            for c in ch:
                if isinstance(c, dict) and c.get("Name"):
                    names.append(str(c["Name"]))
                elif isinstance(c, str):
                    names.append(c)
            if names:
                return names

    xml = _page0_description(tf)
    if xml and "<Channel " in xml:
        names = re.findall(r'Name="([^"]*)"', xml)
        if names:
            return names[: series.shape[axes_u.index("C")]]

    n_c = series.shape[axes_u.index("C")]
    return [f"Channel_{i}" for i in range(int(n_c))]


def _physical_pixel_sizes(tf: tifffile.TiffFile, axes_u: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    xml = _page0_description(tf)
    if not isinstance(xml, str):
        return out
    for letter in ("X", "Y", "Z"):
        if letter not in axes_u:
            continue
        m = re.search(rf'PhysicalSize{letter}="([0-9.+-eE]+)"', xml)
        if m:
            try:
                out[f"pixel_size_{letter}"] = float(m.group(1))
            except ValueError:
                continue
    return out


def _extract_metadata(
    tf: tifffile.TiffFile,
    series: tifffile.TiffPageSeries,
    series_index: int,
) -> Dict[str, Any]:
    axes = getattr(series, "axes", None)
    if not axes or not isinstance(axes, str):
        axes = _letters_for_axes(series.ndim)
    axes_u = axes.upper()

    shape = tuple(int(x) for x in series.shape)
    meta: Dict[str, Any] = {
        "dim_order": axes_u,
        "dim_names": list(axes_u),
        "shape": np.array(shape),
        "ndim": len(shape),
        "num_pixels": math.prod(shape) if shape else 0,
        "dtype": str(np.dtype(series.dtype)),
        "n_images": len(tf.series),
        "channel_names": _channel_names(tf, series, axes_u),
    }
    meta.update(_physical_pixel_sizes(tf, axes_u))
    for i, letter in enumerate(axes_u):
        meta[f"{letter}_size"] = int(shape[i])

    return meta


def _dask_from_series(series: tifffile.TiffPageSeries) -> da.Array:
    try:
        store = series.aszarr()
        za = zarr.open(store, mode="r")
        return da.from_zarr(za)
    except Exception as e:
        logger.warning(
            "tifffile aszarr/Zarr failed (%s); falling back to in-memory array + Dask auto chunks.",
            e,
        )
        arr = series.asarray()
        return da.from_array(np.asarray(arr), chunks="auto")


class TifffileLoader:
    """Load TIFF / OME-TIFF via tifffile; metadata aligned with :class:`BioIoLoader`."""

    NAME = "tifffile"

    SUPPORTED_EXTENSIONS: Set[str] = {"tif", "tiff", "ome.tif"}

    OUTPUT_SCHEMA: Dict[str, Any] = {
        "dim_order": str,
        "dim_names": list,
        "n_images": int,
        "num_pixels": int,
        "shape": pl.Array,
        "ndim": int,
        "channel_names": list,
        "dtype": str,
    }

    OUTPUT_SCHEMA_PATTERNS: List[tuple[str, Any]] = [
        (r"^pixel_size_[A-Za-z]$", float),
        (r"^[A-Za-z]_size$", int),
    ]

    def load(self, source: str):
        path = Path(source)
        if not path.is_file():
            raise UnsupportedFileFormatError(self.NAME, path=str(source))

        try:
            tf = tifffile.TiffFile(path)
        except Exception as e:
            raise UnsupportedFileFormatError(self.NAME, path=str(source)) from e

        try:
            if not tf.series:
                raise UnsupportedFileFormatError(self.NAME, path=str(source))
            if len(tf.series) == 1:
                return self._build_record(tf, 0)
            return {str(i): self._build_record(tf, i) for i in range(len(tf.series))}
        finally:
            tf.close()

    @staticmethod
    def _build_record(tf: tifffile.TiffFile, series_index: int):
        series = tf.series[series_index]
        meta = _extract_metadata(tf, series, series_index)
        meta = normalize_metadata(meta)
        data = _dask_from_series(series)
        data = da.squeeze(data)
        return record_from(data, meta, kind="intensity")
