
from dataclasses import dataclass
from typing import Optional, Set, List, Pattern, Mapping, Any, Literal, Union

from pixel_patrol_base.core.artifact import Artifact

Features = Mapping[str, Any]

# What a processor returns:
# - "features": a flat dict of columns to merge into the table
# - "artifact": a new Artifact (with free-form .kind)
ProcessorOutput = Literal["features", "artifact"]

# The actual return value
ProcessResult = Union[Features, Artifact]


@dataclass(frozen=True)
class ArtifactSpec:
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

def is_artifact_matching_processor(art, processor_input_spec: ArtifactSpec) -> bool:
    if processor_input_spec.axes and not processor_input_spec.axes.issubset(art.axes): return False
    if not is_kind_match(art.kind, processor_input_spec.kinds, processor_input_spec.kind_patterns): return False
    if processor_input_spec.capabilities and not processor_input_spec.capabilities.issubset(art.capabilities): return False
    return True