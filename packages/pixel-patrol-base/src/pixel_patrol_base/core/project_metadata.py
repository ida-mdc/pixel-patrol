"""
Provenance metadata attached to a processed project's parquet file.
Stored in the parquet footer - zero overhead on data read/write.
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
    description: str = ""   # free-form, e.g. "Authors: Annona Buddha and Banana Java"
    version: str = field(default_factory=_get_package_version)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    loader: Optional[str] = None            # NAME of the loader plugin used
    base_dir: Optional[str] = None          # stored for future project reconstruction
    paths: List[str] = field(default_factory=list)  # stored for future project reconstruction
    processing_stats: dict = field(default_factory=dict)  # timing/throughput from build_records_df
    omit_base_dir: bool = False  # when True, pp_base_dir is not written to parquet metadata

    def to_parquet_meta(self) -> dict[str, str]:
        """Serialise to parquet footer key-value pairs."""
        meta = {
            "pp_project_name":     self.project_name,
            "pp_flavor":           self.flavor,
            "pp_description":      self.description,
            "pp_version":          self.version,
            "pp_created_at":       self.created_at,
            "pp_processing_stats": json.dumps(self.processing_stats),
            "pp_paths":            json.dumps(self.paths),
            "pp_loader":           self.loader or "",
        }
        if not self.omit_base_dir:
            meta["pp_base_dir"] = self.base_dir or ""
        return meta

    @classmethod
    def from_parquet_meta(cls, meta: dict[str, str]) -> "ProjectMetadata":
        """Reconstruct from parquet footer key-value pairs."""
        raw_paths = meta.get("pp_paths", "[]")
        try:
            paths = json.loads(raw_paths)
        except json.JSONDecodeError:
            paths = []

        raw_stats = meta.get("pp_processing_stats", "{}")
        try:
            processing_stats = json.loads(raw_stats)
        except json.JSONDecodeError:
            processing_stats = {}

        raw_base = meta.get("pp_base_dir", "")

        return cls(
            project_name=meta.get("pp_project_name", "Imported Project"),
            flavor=meta.get("pp_flavor", ""),
            description=meta.get("pp_description", ""),
            version=meta.get("pp_version", ""),
            created_at=meta.get("pp_created_at", ""),
            loader=meta.get("pp_loader") or None,
            base_dir=raw_base if raw_base else None,
            paths=paths,
            processing_stats=processing_stats,
        )

    def populate_from_project(self, project: "Project") -> "ProjectMetadata":  # type: ignore[name-defined]
        """Fill project_name, base_dir and paths from a live Project instance."""
        from pathlib import Path
        self.project_name = project.name
        self.loader = project.loader.NAME if project.loader else None
        self.base_dir = str(project.base_dir) if project.base_dir else None
        if project.base_dir:
            rel = [str(Path(p).relative_to(project.base_dir)) for p in project.paths]
            self.paths = [p for p in rel if p != "."]
        else:
            self.paths = [str(p) for p in project.paths]
        return self
