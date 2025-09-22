
from dataclasses import dataclass
from typing import Optional, Set, List, Pattern, Mapping, Any, Literal, Union

from pixel_patrol_base.core.record import Record

Features = Mapping[str, Any]

# What a processor returns:
# - "features": a flat dict of columns to merge into the table
# - "record": a new Record (with free-form .kind)
ProcessorOutput = Literal["features", "record"]

# The actual return value
ProcessResult = Union[Features, Record]


@dataclass(frozen=True)
class RecordSpec:
    axes: Optional[Set[str]] = None
    kinds: Optional[Set[str]] = None       # {"text"}, {"audio/*"}, {"*"}, etc.
    capabilities: Optional[Set[str]] = None
    kind_patterns: Optional[List[Pattern[str]]] = None  # optional regexes

def is_kind_match(art_kind: str,
                  kinds: Optional[Set[str]],
                  patterns: Optional[List[Pattern[str]]]) -> bool:
    if kinds is None or "*" in kinds:
        return True
    if art_kind in kinds:
        return True
    # prefix match: "audio/*" matches "audio/waveform", "audio/mel", ...
    for k in kinds:
        if k.endswith("/*") and art_kind.startswith(k[:-2] + "/"):
            return True
    if patterns and any(p.search(art_kind) for p in patterns):
        return True
    return False

def is_record_matching_processor(art, processor_input_spec: RecordSpec) -> bool:
    if processor_input_spec.axes:
        art_axes = set(getattr(art, 'dim_order', '') or '')
        if not processor_input_spec.axes.issubset(art_axes): return False
    if not is_kind_match(art.kind, processor_input_spec.kinds, processor_input_spec.kind_patterns): return False
    if processor_input_spec.capabilities and not processor_input_spec.capabilities.issubset(art.capabilities): return False
    return True
