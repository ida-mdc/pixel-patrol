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

from typing import Mapping, List, Any

def _infer_dim_names(order: str, meta: Mapping[str, Any]) -> List[str]:
    """
    Decide human-readable dim names.
      - If meta['dim_names'] is valid, use it as-is.
      - Else if meta['dim_order'] equals 'order' and is single-letter, use letters (lowercase).
      - Else fallback to ['dimA','dimB',...] from 'order'; if no order, use ['dim1',...].
    """
    # 1) explicit names from metadata win
    names = meta.get("dim_names")
    if isinstance(names, list) and len(names) == len(order) and all(isinstance(x, str) for x in names):
        return names

    # 2) if the order came from metadata and is single-letter, derive names from it (no 'dim' prefix)
    mo = meta.get("dim_order")
    if isinstance(mo, str) and mo == order and order.isalpha() and order.isupper():
        return [ch.lower() for ch in order]

    # 3) fallback from order → 'dimA','dimB',...
    if isinstance(order, str) and order:
        return [f"dim{ch.upper()}" for ch in order]

    # 4) no order → numeric dims using meta['ndim']
    n = meta.get("ndim")
    n = int(n) if isinstance(n, int) else 0
    return [f"dim{i+1}" for i in range(n)]


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