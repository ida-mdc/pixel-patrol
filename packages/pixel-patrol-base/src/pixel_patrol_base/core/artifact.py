from dataclasses import dataclass
from typing import Any, Mapping, Set, List
import string

Kind = str

@dataclass(frozen=True)
class Artifact:
    data: Any
    dim_order: str
    dim_names: List[str]
    kind: Kind
    meta: Mapping[str, Any]
    capabilities: Set[str]

def _infer_dim_order(array: Any, meta: Mapping[str, Any]) -> str:
    order = meta.get("dim_order", "")
    if isinstance(order, str) and order.isalpha(): return order
    n = meta.get("ndim")
    if not isinstance(n, int):
        n = getattr(array, "ndim", None)
        if not isinstance(n, int):
            shape = getattr(array, "shape", None)
            n = len(shape) if isinstance(shape, (list, tuple)) else 0
    return ''.join(string.ascii_uppercase[i] for i in range(min(int(n or 0), 26)))

def _infer_dim_names(order: str, meta: Mapping[str, Any]) -> List[str]:
    names = meta.get("dim_names")
    if isinstance(names, list) and len(names) == len(order) and all(isinstance(x, str) for x in names):
        return names
    return [f"dim{i+1}" for i in range(len(order))]

def artifact_from(array: Any, meta: Mapping[str, Any], *, kind: Kind = "image/intensity") -> Artifact:
    mm = dict(meta or {})
    dim_order = _infer_dim_order(array, mm)
    dim_names = _infer_dim_names(dim_order, mm)
    mm["dim_order"] = dim_order
    mm["dim_names"] = dim_names
    capabilities: Set[str] = set()
    if 'X' in dim_order and 'Y' in dim_order: capabilities.add('spatial-2d')
    if 'Z' in dim_order: capabilities.add('spatial-3d')
    if 'T' in dim_order: capabilities.add('temporal')
    if 'C' in dim_order: capabilities.add('multichannel')
    return Artifact(data=array, dim_order=dim_order, dim_names=dim_names, kind=kind, meta=mm, capabilities=capabilities)