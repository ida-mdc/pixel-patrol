from __future__ import annotations
from dataclasses import dataclass, field
from typing import Set, Literal

from pixel_patrol_base.core.artifact import ArrayKind, Artifact


@dataclass(frozen=True)
class ArtifactSpec:
    axes: Set[str] = field(default_factory=set)           # required axes subset
    kinds: Set[ArrayKind] = field(default_factory=lambda: {"any"})
    capabilities: Set[str]  = field(default_factory=set)

OutputKind = Literal["features","artifact"]

def can_accept(artifact: Artifact, spec: ArtifactSpec) -> bool:
    if not spec.axes.issubset(artifact.axes): return False
    if "any" not in spec.kinds and artifact.kind not in spec.kinds: return False
    if not spec.capabilities.issubset(artifact.capabilities): return False
    return True