"""
The report manager's library of reports.

A report is a single ``.parquet`` file. Files inside ``PIXEL_PATROL_REPORTS_DIR``
are discovered automatically; reports imported from elsewhere are remembered by
absolute path in ``reports_index.json`` inside that directory. Each report's
card data is read from the file itself (parquet footer + a few aggregate
queries + the baked-in thumbnail) and cached per ``(path, mtime)``.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from pixel_patrol_base.viewer_server import _setup_duckdb

logger = logging.getLogger(__name__)

REPORTS_DIR = Path(
    os.environ.get("PIXEL_PATROL_REPORTS_DIR", str(Path.home() / "pixel-patrol-reports"))
).expanduser()

_THUMBNAIL_PX = 96

# id -> resolved parquet path, filled whenever a report is listed or opened so
# that /api/query?report=<id> can find its file.
_registry: Dict[str, Path] = {}
_registry_lock = threading.Lock()

_meta_cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}
_meta_lock = threading.Lock()

_index_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Ids
# ---------------------------------------------------------------------------

def report_id(path: Path) -> str:
    return hashlib.sha1(str(path).encode()).hexdigest()[:16]


def register_report(path: Path) -> str:
    rid = report_id(path)
    with _registry_lock:
        _registry[rid] = path
    return rid


def resolve_report(rid: str) -> Optional[Path]:
    with _registry_lock:
        return _registry.get(rid)


# ---------------------------------------------------------------------------
# Index of imported (external) reports
# ---------------------------------------------------------------------------

def _index_file() -> Path:
    return REPORTS_DIR / "reports_index.json"


def _load_index() -> List[str]:
    try:
        data = json.loads(_index_file().read_text())
    except (OSError, ValueError):
        return []
    if not isinstance(data, list):
        return []
    return [item for item in data if isinstance(item, str) and item.strip()]


def _save_index(paths: List[str]) -> None:
    """Atomically replace the index file. Callers hold ``_index_lock``."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = _index_file().with_suffix(".json.tmp")
    tmp.write_text(json.dumps(sorted(set(paths)), indent=2))
    os.replace(tmp, _index_file())


def _add_to_index(path: Path) -> None:
    with _index_lock:
        paths = _load_index()
        if str(path) not in paths:
            _save_index([*paths, str(path)])


def _remove_from_index(path: Path) -> None:
    with _index_lock:
        paths = _load_index()
        kept = [p for p in paths if str(Path(p).resolve()) != str(path)]
        if len(kept) != len(paths):
            _save_index(kept)


def is_internal(path: Path) -> bool:
    """True if the path lives inside REPORTS_DIR (auto-discovered, not imported)."""
    try:
        path.resolve().relative_to(REPORTS_DIR.resolve())
        return True
    except (ValueError, OSError):
        return False


# ---------------------------------------------------------------------------
# Card metadata
# ---------------------------------------------------------------------------

def _format_size(n_bytes: int) -> str:
    size = float(n_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.0f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def _thumbnail_b64(conn) -> Optional[str]:
    """First baked thumbnail as a small JPEG (stored as a flat square RGBA uint8 canvas)."""
    try:
        row = conn.execute("SELECT thumbnail FROM pp_data WHERE thumbnail IS NOT NULL LIMIT 1").fetchone()
        raw = bytes(row[0]) if row and row[0] else b""
        side = int(round((len(raw) / 4) ** 0.5))
        if side <= 0 or side * side * 4 != len(raw):
            return None
        arr = np.frombuffer(raw, dtype=np.uint8).reshape(side, side, 4)
        img = Image.fromarray(arr, "RGBA").convert("RGB")
        img.thumbnail((_THUMBNAIL_PX, _THUMBNAIL_PX), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=72)
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        logger.debug("Could not decode thumbnail", exc_info=True)
        return None


def _columns(conn) -> set:
    return {r[0] for r in conn.execute("SELECT column_name FROM information_schema.columns WHERE table_name = 'pp_data'").fetchall()}


def _file_type_counts(conn) -> Dict[str, int]:
    rows = conn.execute(
        "SELECT file_extension, count(*) AS c FROM pp_data GROUP BY file_extension ORDER BY c DESC LIMIT 8"
    ).fetchall()
    return {str(ext): int(c) for ext, c in rows if ext}


def _subpaths(kv: Dict[str, str]) -> list:
    try:
        value = json.loads(kv.get("pp_paths", "[]"))
    except ValueError:
        return []
    return value if isinstance(value, list) else []


def _created_at(kv: Dict[str, str], path: Path) -> str:
    if kv.get("pp_created_at"):
        return kv["pp_created_at"]
    return datetime.fromtimestamp(path.stat().st_mtime).isoformat()


def _compute_meta(path: Path) -> Dict[str, Any]:
    conn, kv = _setup_duckdb(path)
    try:
        cols = _columns(conn)
        n_files = int(conn.execute("SELECT count(*) FROM pp_data").fetchone()[0] or 0)
        total_size = 0
        if "size_bytes" in cols:
            total_size = int(conn.execute("SELECT coalesce(sum(size_bytes), 0) FROM pp_data").fetchone()[0] or 0)
        return {
            "project_name": kv.get("pp_project_name") or "",
            "base_dir": kv.get("pp_base_dir") or "",
            "description": kv.get("pp_description") or "",
            "flavor": kv.get("pp_flavor") or "",
            "created_at": _created_at(kv, path),
            "subpaths": _subpaths(kv),
            "n_files": n_files,
            "total_size_bytes": total_size,
            "size_readable": _format_size(total_size),
            "file_type_counts": _file_type_counts(conn) if "file_extension" in cols else {},
            "thumbnail_b64": _thumbnail_b64(conn) if "thumbnail" in cols else None,
        }
    finally:
        conn.close()


def _read_meta(path: Path) -> Dict[str, Any]:
    key = str(path)
    mtime = path.stat().st_mtime
    with _meta_lock:
        cached = _meta_cache.get(key)
        if cached and cached[0] == mtime:
            return cached[1]
    try:
        meta = _compute_meta(path)
    except Exception as exc:
        logger.warning("Could not read report metadata for %s: %s", path, exc)
        meta = {}
    with _meta_lock:
        _meta_cache[key] = (mtime, meta)
    return meta


def invalidate_meta(path: Optional[Path] = None) -> None:
    """Forget cached card data for one report, or for all of them."""
    with _meta_lock:
        if path is None:
            _meta_cache.clear()
        else:
            _meta_cache.pop(str(path), None)


# ---------------------------------------------------------------------------
# Listing, importing, deleting
# ---------------------------------------------------------------------------

def _entry(path: Path, source: str) -> Dict[str, Any]:
    """One report card: metadata if the file is present, a stub if it is gone."""
    resolved = path.resolve()
    exists = resolved.is_file()
    return {
        "id": register_report(resolved),
        "path": str(resolved),
        "filename": resolved.name,
        "source": source,
        "exists": exists,
        "parquet_size_bytes": resolved.stat().st_size if exists else 0,
        **(_read_meta(resolved) if exists else {}),
    }


def _sort_key(report: Dict[str, Any]):
    # Existing before missing, imported before internal, newest first.
    return (report["exists"], report["source"] == "imported", report.get("created_at", ""))


def scan_reports() -> List[Dict[str, Any]]:
    """Merge auto-discovered REPORTS_DIR reports with imported external ones."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    seen: Dict[str, Dict[str, Any]] = {}
    for parquet in REPORTS_DIR.glob("*.parquet"):
        entry = _entry(parquet, "internal")
        seen[entry["path"]] = entry
    for external in _load_index():
        key = str(Path(external).resolve())
        if key not in seen:
            seen[key] = _entry(Path(external), "imported")
    return sorted(seen.values(), key=_sort_key, reverse=True)


def _known_paths() -> Dict[str, Path]:
    """Every report the manager knows (REPORTS_DIR contents + imported), by resolved path string."""
    known = {str(p.resolve()): p.resolve() for p in REPORTS_DIR.glob("*.parquet")}
    for external in _load_index():
        resolved = Path(external).resolve()
        known[str(resolved)] = resolved
    return known


def find_known(raw_path: str) -> Optional[Path]:
    """The manager's own path for a requested report, or None if it is not one it knows.

    The returned path comes from the manager's list, never from the request.
    """
    try:
        requested = str(Path(raw_path).expanduser().resolve())
    except OSError:
        return None
    return _known_paths().get(requested)


def import_report(path: Path) -> Dict[str, Any]:
    """Add an existing parquet (typically outside REPORTS_DIR) to the index."""
    try:
        resolved = path.expanduser().resolve()
    except OSError:
        return {"error": f"Invalid path: {path}", "status": 400}
    if not resolved.exists() or resolved.suffix.lower() != ".parquet":
        return {"error": f"Not a parquet report: {resolved}", "status": 404}
    if not is_internal(resolved):
        _add_to_index(resolved)
    invalidate_meta(resolved)
    return {"status": "ok", "id": register_report(resolved)}


def delete_report(path: Path) -> bool:
    """Delete an internal report from disk; for imported ones only drop the index entry."""
    resolved = path.resolve()
    if is_internal(resolved):
        try:
            resolved.unlink(missing_ok=True)
        except OSError as exc:
            logger.warning("Could not delete report %s: %s", resolved, exc)
            return False
    _remove_from_index(resolved)
    invalidate_meta(resolved)
    with _registry_lock:
        _registry.pop(report_id(resolved), None)
    return True


def default_output_path(base_directory: str, project_name: str) -> Path:
    """Auto-name an output parquet inside REPORTS_DIR for a new processing run."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    base = project_name.strip() or Path(base_directory).name or "project"
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in base)
    return REPORTS_DIR / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{safe}.parquet"
