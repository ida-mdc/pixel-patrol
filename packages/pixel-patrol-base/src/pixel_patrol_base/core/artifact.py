from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Mapping, Set, Literal, Dict

ArrayKind = Literal["intensity", "label", "rgb", "multichannel", "any"]

@dataclass(frozen=True)
class Artifact:
    data: Any                    # typically a dask.array.Array (lazy is fine)
    axes: Set[str]               # e.g. {"X","Y"} (+ maybe "Z","C","T")
    kind: ArrayKind              # semantic type
    meta: Mapping[str, Any]      # includes 'dim_order', sizes, etc.
    capabilities: Set[str]               # capabilities (e.g. {"spatial-2d","temporal"})

def artifact_from(array: Any, meta: Mapping[str, Any], *, kind: ArrayKind = "intensity") -> Artifact:
    dim_order = (meta or {}).get("dim_order", "")
    axes = set(dim_order) if isinstance(dim_order, str) else set()
    capabilities: Set[str] = set()
    if "X" in axes and "Y" in axes: capabilities.add("spatial-2d")
    if "Z" in axes: capabilities.add("spatial-3d")
    if "T" in axes: capabilities.add("temporal")
    if "C" in axes: capabilities.add("multichannel")
    return Artifact(data=array, axes=axes, kind=kind, meta=meta or {}, capabilities=capabilities)