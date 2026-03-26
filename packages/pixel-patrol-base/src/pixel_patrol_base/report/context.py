"""Report context for widgets - get project/df by zip path (avoids circular imports)."""
from pathlib import Path
from typing import Tuple, Optional, Any
from urllib.parse import parse_qs, unquote

# Cache for project loaded from zip path (Project type; avoid import to prevent circular)
_project_cache: dict[str, Any] = {}
_current_report_project: Any = None


def set_current_project(project: Any) -> None:
    """Set the current project (used by standalone report)."""
    global _current_report_project
    _current_report_project = project


def get_report_context(zip_path: str | None = None) -> Tuple[Optional[Any], Optional[Any], Optional[str]]:
    """
    Return (df, project, project_name) for report callbacks.
    When zip_path is provided (embedded mode), load from path.
    Otherwise use _current_report_project (standalone mode).
    """
    if zip_path:
        from pixel_patrol_base import api
        path_key = str(Path(zip_path).resolve())
        if path_key not in _project_cache:
            _project_cache[path_key] = api.import_project(Path(path_key))
        p = _project_cache[path_key]
    else:
        p = _current_report_project
    if p is None:
        return None, None, None
    return p.records_df, p, p.name


def clear_for_test() -> None:
    """Clear caches (for tests)."""
    global _project_cache, _current_report_project
    _project_cache.clear()
    _current_report_project = None


def zip_path_from_url_search(search: str | None) -> str | None:
    """Extract zip path from Location search string (?zip=...)."""
    if not search:
        return None
    params = parse_qs(search.lstrip("?"))
    zip_param = params.get("zip", [None])[0]
    return unquote(zip_param) if zip_param else None
