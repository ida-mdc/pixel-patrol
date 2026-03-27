"""
Provenance metadata attached to a processed project's parquet file.
Stored in the parquet footer — zero overhead on data read/write.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional
from importlib.metadata import version, PackageNotFoundError


def _get_package_version() -> str:
    try:
        return version("pixel_patrol_base")
    except PackageNotFoundError:
        return ""


@dataclass
class ProjectMetadata:
    project_name: str = "Imported Project"
    flavor: str = ""
    authors: str = ""       # free-form, e.g. "ella, deborah"
    version: str = field(default_factory=_get_package_version)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    base_dir: Optional[str] = None          # stored for future project reconstruction
    paths: List[str] = field(default_factory=list)  # stored for future project reconstruction

    def to_parquet_meta(self) -> dict[str, str]:
        """Serialise to parquet footer key-value pairs."""
        return {
            "pp_project_name": self.project_name,
            "pp_flavor":       self.flavor,
            "pp_authors":      self.authors,
            "pp_version":      self.version,
            "pp_created_at":   self.created_at,
            "pp_base_dir":     self.base_dir or "",
            "pp_paths":        json.dumps(self.paths),
        }

    @classmethod
    def from_parquet_meta(cls, meta: dict[str, str]) -> "ProjectMetadata":
        """Reconstruct from parquet footer key-value pairs."""
        raw_paths = meta.get("pp_paths", "[]")
        try:
            paths = json.loads(raw_paths)
        except json.JSONDecodeError:
            paths = []

        raw_base = meta.get("pp_base_dir", "")

        return cls(
            project_name=meta.get("pp_project_name", "Imported Project"),
            flavor=meta.get("pp_flavor", ""),
            authors=meta.get("pp_authors", ""),
            version=meta.get("pp_version", ""),
            created_at=meta.get("pp_created_at", ""),
            base_dir=raw_base if raw_base else None,
            paths=paths,
        )

    def populate_from_project(self, project: "Project") -> "ProjectMetadata":  # type: ignore[name-defined]
        """Fill project_name, base_dir and paths from a live Project instance."""
        self.project_name = project.name
        self.base_dir = str(project.base_dir) if project.base_dir else None
        self.paths = [str(p) for p in project.paths]
        return self
