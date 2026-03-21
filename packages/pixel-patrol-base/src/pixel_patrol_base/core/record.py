from dataclasses import dataclass
from typing import Any, Mapping, Set, List
import string
import logging
logger = logging.getLogger(__name__)

Kind = str

@dataclass(frozen=True)
class Record:
    data: Any
    dim_order: str
    dim_names: List[str]
    kind: Kind
    meta: Mapping[str, Any]
    capabilities: Set[str]

def as_list(v):
    if v is None:
        return []
    if hasattr(v, "tolist"):
        return v.tolist()
    return list(v)


def _validate_and_fix_meta(array: Any, meta: Mapping[str, Any]) -> Mapping[str, Any]:
    meta = dict(meta)  # Create a mutable copy of the input meta mapping

    meta_ndim = meta.get("ndim")
    if isinstance(meta_ndim, str) and meta_ndim.strip().lstrip("+-").isdigit():
        meta_ndim = int(meta_ndim)
    elif not isinstance(meta_ndim, int):
        meta_ndim = None

    meta_shape = as_list(meta.get("shape"))
    arr_shape = as_list(getattr(array, "shape", None))
    arr_ndim = getattr(array, "ndim", None) or len(arr_shape)

    if arr_ndim is not None and (meta_ndim is None or meta_ndim != arr_ndim):
        if meta_ndim is not None:
            logger.warning(f"meta['ndim'] != array.ndim - {meta_ndim} != {arr_ndim}; using array.ndim")
        meta['ndim'] = arr_ndim

    if not meta_shape and arr_shape:
        meta['shape'] = list(arr_shape)
    elif meta_shape and arr_shape and tuple(meta_shape) != tuple(arr_shape):
        logger.warning(f"meta['shape'] != array.shape - {meta_shape} != {arr_shape}; using array.shape")
        meta['shape'] = list(arr_shape)
    meta['shape'] = list(meta['shape'])

    if "dim_order" in meta:
        if not isinstance(meta['dim_order'], str) or len(meta['dim_order']) != arr_ndim:
            logger.warning(
                f"meta['dim_order'] {meta['dim_order']} is invalid or length mismatch ({len(meta['dim_order'])} != {arr_ndim}); removing from meta")
            del meta["dim_order"]

    if "dim_names" in meta:
        if isinstance(meta['dim_names'], tuple):
            meta['dim_names'] = list(meta['dim_names'])
        if not isinstance(meta['dim_names'], list):
            logger.warning(
                f"meta['dim_names'] must be list or tuple, found {type(meta['dim_names']).__name__}; removing from meta")
            del meta["dim_names"]
        elif len(meta['dim_names']) != arr_ndim:
                logger.warning(
                    f"meta['dim_names'] length mismatch ({len(meta['dim_names'])} != {arr_ndim}); removing from meta")
                del meta["dim_names"]

    if "channel_names" in meta:
        cn = as_list(meta["channel_names"])
        meta["channel_names"] = [
            (x.item() if hasattr(x, "item") else x) if isinstance(x, str) else str(x)
            for x in cn
        ]

    return meta


def _infer_dim_order(meta: Mapping[str, Any]) -> str:
    meta_order = meta.get("dim_order")
    meta_names = meta.get("dim_names")
    meta_ndim = meta.get("ndim")

    if meta_order and isinstance(meta_order, str):
        return meta_order

    if (meta_names and isinstance(meta_names, list)
            and all(isinstance(x, str) for x in meta_names)
            and all(len(x) == 1 for x in meta_names)):
        return "".join(meta_names)
    else:
        return ''.join(string.ascii_uppercase[i] for i in range(min(int(meta_ndim or 0), 26)))


def _infer_dim_names(order: str, meta: Mapping[str, Any]) -> List[str]:

    names = meta.get("dim_names")
    if isinstance(names, list) and all(isinstance(x, str) for x in names):
        return names

    meta_order = meta.get("dim_order")
    if isinstance(meta_order, str) and meta_order == order:
        return list(order)

    if isinstance(order, str) and order:
        return [f"dim{ch.upper()}" for ch in order]

    return []


def record_from(array: Any, meta: Mapping[str, Any], *, kind: Kind = "image/intensity") -> Record:
    mm = dict(meta or {})
    mm = _validate_and_fix_meta(array, mm)
    dim_order = _infer_dim_order(mm)
    dim_names = _infer_dim_names(dim_order, mm)
    mm["dim_order"] = dim_order
    mm["dim_names"] = dim_names
    capabilities: Set[str] = set()
    if 'X' in dim_order and 'Y' in dim_order: capabilities.add('spatial-2d')
    if 'Z' in dim_order: capabilities.add('spatial-3d')
    if 'T' in dim_order: capabilities.add('temporal')
    if 'C' in dim_order: capabilities.add('multichannel')
    rgb_cap = _infer_rgb_capability(dim_order, mm, getattr(array, 'dtype', None))
    if rgb_cap:
        capabilities.add(rgb_cap)
    return Record(data=array, dim_order=dim_order, dim_names=dim_names, kind=kind, meta=mm, capabilities=capabilities)


def _infer_rgb_capability(dim_order: str, meta: Mapping[str, Any], dtype=None) -> str | None:
    """
    Returns an 'rgb:<dim>' capability string if the image represents an RGB or RGBA image,
    otherwise None.

    Only uint8 arrays can be RGB — non-uint8 values are not in the 0-255 range
    expected for direct color display.

    The S (samples) dimension unambiguously signals RGB/RGBA in OME-style formats.
    A C dimension is treated as RGB only when the channel names explicitly identify
    the channels as red, green, blue (and optionally alpha).
    """
    if dtype is not None and getattr(dtype, 'name', str(dtype)) != 'uint8':
        return None

    shape = meta.get('shape', [])
    if not shape:
        return None

    if 'S' in dim_order:
        s_size = shape[dim_order.index('S')]
        if s_size in (3, 4):
            return 'rgb:S'

    if 'C' in dim_order:
        c_size = shape[dim_order.index('C')]
        channel_names = meta.get('channel_names', [])
        _rgb_names = {'R', 'G', 'B', 'A', 'RED', 'GREEN', 'BLUE', 'ALPHA'}
        if c_size in (3, 4) and channel_names and all(
            str(n).upper() in _rgb_names for n in channel_names
        ):
            return 'rgb:C'

    return None