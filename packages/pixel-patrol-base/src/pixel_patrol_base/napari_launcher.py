"""Open one image - or the exact slice / tile / chunk a viewer row represents - in napari.

Run as a subprocess by the viewer server's ``/api/open-napari`` endpoint so napari's
Qt event loop stays isolated from the HTTP server thread.  It reuses the very loader
the processing pipeline used, takes its lazy Dask array, and indexes the same block
the clicked row was measured on - so what you see in napari is exactly what Pixel
Patrol reported for that row.

    python -m pixel_patrol_base.napari_launcher \
        --path /data/img.ome.tif --loader tifffile \
        --slices '{"Z": [3, 4]}' --child-id 0 --name img.ome.tif
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pixel_patrol_base.core.record import Record

logger = logging.getLogger(__name__)


def _resolve_loader(loader_name: Optional[str], file_path: Path):
    """Return the loader instance to use: the recorded one, else one that claims the extension."""
    from pixel_patrol_base.plugin_registry import discover_loader, discover_plugins_from_entrypoints

    if loader_name:
        return discover_loader(loader_id=loader_name)

    ext = file_path.name.lower().split(".", 1)[-1]
    for loader_cls in discover_plugins_from_entrypoints("pixel_patrol.loader_plugins"):
        supported = {e.lower() for e in getattr(loader_cls, "SUPPORTED_EXTENSIONS", set()) or set()}
        if ext in supported or file_path.suffix.lstrip(".").lower() in supported:
            return loader_cls()
    raise RuntimeError(f"No loader records for '{file_path.name}' and none claims its extension.")


def _load_record(loader, file_path: Path, child_id: Optional[str]) -> Record:
    """Load the whole-file Record, or a single sub-image for container formats."""
    if child_id in (None, ""):
        return loader.load(file_path)
    index = int(child_id)
    for _sub_id, record in loader.load_range(file_path, index, index + 1):
        return record
    raise ValueError(f"Sub-image {child_id} not found in {file_path}")


def _slice_tuple(dim_order: str, axis_slices: Dict[str, List[int]], shape: Tuple[int, ...]) -> Tuple[slice, ...]:
    """Build one slice per axis: the row's [start, stop) where pinned, whole axis otherwise.

    ``axis_slices`` is keyed by upper-case axis letter; stops are clamped to the loaded
    array's real extent so a stale row can never index out of bounds.
    """
    out: List[slice] = []
    for i, axis in enumerate(dim_order):
        span = axis_slices.get(axis.upper())
        if span is None:
            out.append(slice(None))
            continue
        start = max(0, min(int(span[0]), shape[i] - 1))
        stop = max(start + 1, min(int(span[1]), shape[i]))
        out.append(slice(start, stop))
    return tuple(out)


def open_in_napari(
    file_path: Path,
    loader_name: Optional[str],
    axis_slices: Dict[str, List[int]],
    child_id: Optional[str] = None,
    name: Optional[str] = None,
) -> None:
    """Load the block the row describes and hand it (lazily) to napari."""
    import napari

    loader = _resolve_loader(loader_name, file_path)
    record = _load_record(loader, file_path, child_id)
    block = record.data[_slice_tuple(record.dim_order, axis_slices, tuple(record.data.shape))]

    viewer = napari.Viewer()
    viewer.add_image(block, name=name or file_path.name)
    if len(record.dim_order) == block.ndim:
        viewer.dims.axis_labels = tuple(record.dim_order)
    napari.run()


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Open an image (or one of its slices) in napari.")
    parser.add_argument("--path", required=True)
    parser.add_argument("--loader", default="")
    parser.add_argument("--slices", default="{}", help="JSON: axis letter -> [start, stop]")
    parser.add_argument("--child-id", default=None)
    parser.add_argument("--name", default=None)
    args = parser.parse_args(argv)

    open_in_napari(
        file_path=Path(args.path),
        loader_name=args.loader or None,
        axis_slices=json.loads(args.slices),
        child_id=args.child_id,
        name=args.name,
    )


if __name__ == "__main__":
    main()
