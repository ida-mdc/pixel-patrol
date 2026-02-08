"""
Configuration for file processing operations (slicing and processor selection).
These are runtime parameters, not persisted project settings.
"""

from dataclasses import dataclass, field
from typing import Set


@dataclass
class ProcessingConfig:
    """Configuration for file processing operations."""
    # Slicing configuration: If False, slicing is disabled entirely (only full-image stats computed)
    slicing_enabled: bool = True
    # Dimensions to include in slicing (e.g., {"T", "C"}). If non-empty, only these dimensions are sliced.
    # If empty, slicing_dimensions_excluded is used instead.
    slicing_dimensions_included: Set[str] = field(default_factory=set)
    # Dimensions to exclude from slicing (e.g., {"X", "Y", "Z"}). Default: {"X", "Y"} are never sliced.
    # Only used if slicing_dimensions_included is empty.
    slicing_dimensions_excluded: Set[str] = field(default_factory=lambda: {"X", "Y"})
    processors_included: Set[str] = field(default_factory=set)
    processors_excluded: Set[str] = field(default_factory=set)
