from dataclasses import dataclass
from typing import Any, Mapping, Set

Kind = str

@dataclass(frozen=True)
class Artifact:
    data: Any
    axes: Set[str]
    kind: Kind
    meta: Mapping[str, Any]
    capabilities: Set[str]

def artifact_from(array: Any, meta: Mapping[str, Any], *, kind: Kind = "image/intensity") -> Artifact:
    dim_order = (meta or {}).get("dim_order", "")
    axes = set(dim_order) if isinstance(dim_order, str) else set()
    capabilities: Set[str] = set()
    if "X" in axes and "Y" in axes: capabilities.add("spatial-2d")
    if "Z" in axes: capabilities.add("spatial-3d")
    if "T" in axes: capabilities.add("temporal")
    if "C" in axes: capabilities.add("multichannel")
    return Artifact(data=array, axes=axes, kind=kind, meta=meta or {}, capabilities=capabilities)