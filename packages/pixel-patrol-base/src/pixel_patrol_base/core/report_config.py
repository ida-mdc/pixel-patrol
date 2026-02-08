"""
Configuration for report generation (widget selection and global filters/grouping).
These are runtime parameters, not persisted project settings.
"""

from dataclasses import dataclass, field
from typing import Set, Dict, Optional


@dataclass
class ReportConfig:
    """Configuration for report generation."""
    widgets_included: Set[str] = field(default_factory=set)
    widgets_excluded: Set[str] = field(default_factory=set)
    # Global configuration for filters, grouping, and dimensions
    group_col: Optional[str] = None  # Column name to group by (e.g., "imported_path_short")
    filter: Optional[Dict] = None  # Filter dict, e.g., {"file_extension": {"op": "in", "value": "tif, png"}}
    dimensions: Optional[Dict[str, str]] = None  # Dimension filters, e.g., {"T": "0", "Z": "1"}
    is_show_significance: bool = False  # Whether to show statistical significance annotations
    
    def to_dict(self) -> Dict:
        """Convert to dict format for Dash Store serialization."""
        result = {}
        if self.group_col is not None:
            result["group_col"] = self.group_col
        if self.filter is not None:
            result["filter"] = self.filter
        if self.dimensions is not None:
            result["dimensions"] = self.dimensions
        if self.is_show_significance:
            result["is_show_significance"] = True
        return result
    
    @classmethod
    def from_dict(cls, d: Optional[Dict], **kwargs) -> "ReportConfig":
        """Create ReportConfig from dict format. Merges dict fields with kwargs."""
        if d is None:
            return cls(**kwargs)
        return cls(
            group_col=d.get("group_col"),
            filter=d.get("filter"),
            dimensions=d.get("dimensions"),
            is_show_significance=d.get("is_show_significance", False),
            **kwargs
        )
