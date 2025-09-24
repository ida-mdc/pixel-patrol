from dataclasses import dataclass
from typing import Any, Mapping, Set, List
import string

Kind = str

@dataclass(frozen=True)
class Record:
    data: Any
    dim_order: str
    dim_names: List[str]
    kind: Kind
    meta: Mapping[str, Any]
    capabilities: Set[str]

# _infer_dim_order: accept only single-letter codes and (if known) match ndim, else fallback ABC...
def _infer_dim_order(array: Any, meta: Mapping[str, Any]) -> str:
    order = meta.get("dim_order", "")
    n = meta.get("ndim")
    if not isinstance(n, int):
        n = getattr(array, "ndim", None) or len(getattr(array, "shape", []) or [])
    if isinstance(order, str) and order.isalpha() and (not n or len(order) == int(n)):
        return order
    return ''.join(string.ascii_uppercase[i] for i in range(min(int(n or 0), 26)))

# _infer_dim_names: prefer meta dim_names; otherwise derive from dim_order letters (lowercase)
def _infer_dim_names(order: str, meta: Mapping[str, Any]) -> List[str]:
    names = meta.get("dim_names")
    if isinstance(names, list) and len(names) == len(order) and all(isinstance(x, str) for x in names):
        return names
    if isinstance(order, str) and order:
        return [ch.lower() for ch in order]
    return [f"dim{i+1}" for i in range(len(order))]


def record_from(array: Any, meta: Mapping[str, Any], *, kind: Kind = "image/intensity") -> Record:
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
    return Record(data=array, dim_order=dim_order, dim_names=dim_names, kind=kind, meta=mm, capabilities=capabilities)