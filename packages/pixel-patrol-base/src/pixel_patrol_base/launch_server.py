"""
Local HTTP server for the PixelPatrol report manager.

Serves a small static JS/HTML frontend (``launch_assets/``) that lists the
reports in ``PIXEL_PATROL_REPORTS_DIR`` (see ``report_library``), runs new
processing in a background thread, and opens any report in the viewer.

The viewer is served from this same server under ``/report?path=...``, reusing
``viewer_server._ViewerHandler``: its SQL queries go to
``/api/query?report=<id>``, answered by a native DuckDB connection per report.
One port keeps the whole experience working on an HPC node reachable through a
single forwarded port.
"""

from __future__ import annotations

import importlib.metadata
import json
import logging
import os
import platform
import shutil
import socket
import subprocess
import sys
import threading
import urllib.request
import webbrowser
from urllib.parse import parse_qs, quote, urlsplit
from collections import deque
from datetime import datetime
from functools import lru_cache
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from packaging.version import InvalidVersion, Version

from pixel_patrol_base import api
from pixel_patrol_base import report_library as library
from pixel_patrol_base.viewer_server import (
    _discover_installed_extensions,
    _js,
    _mime,
    _setup_duckdb,
    _ViewerHandler,
    build_viewer_url_params,
    find_viewer_dist,
)

logger = logging.getLogger(__name__)

ASSETS_DIR = (Path(__file__).parent / "launch_assets").resolve()

# ~/.pixel-patrol is created by the PixelPatrol launcher (deploy/launcher);
# its presence as our venv root is how we detect a launcher-managed install.
_LAUNCHER_HOME = Path.home() / ".pixel-patrol"
_LAUNCHER_VENV = _LAUNCHER_HOME / "venv"
_PYPI_URL = "https://pypi.org/pypi/pixel-patrol/json"

# ---------------------------------------------------------------------------
# Processing state (single in-flight job, mirrors the previous Dash app)
# ---------------------------------------------------------------------------

_state_lock = threading.Lock()
_state: Dict[str, Any] = {
    "status": "idle",  # idle, running, completed, error
    "progress": 0,
    "message": "",
    "processed_files": 0,
    "total_files": 0,
    "error": None,
    "output_parquet": None,
}


def get_state() -> Dict[str, Any]:
    with _state_lock:
        return dict(_state)


def update_state(**kwargs: Any) -> None:
    with _state_lock:
        _state.update(kwargs)


class ProcessingCancelled(Exception):
    """Raised from the progress callback to abort a running processing job."""


_cancel_event = threading.Event()


# ---------------------------------------------------------------------------
# Warning capture (so errors/warnings during processing surface in the UI)
# ---------------------------------------------------------------------------

_warning_queue: deque = deque(maxlen=100)
_console_lines: deque = deque(maxlen=500)


class _WarningCaptureHandler(logging.Handler):
    """Collects INFO+ records as console lines, and WARNING+ records as UI alerts."""

    def emit(self, record: logging.LogRecord) -> None:
        stamp = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
        _console_lines.append(f"{stamp} {record.levelname:<7} {record.getMessage()}")
        if record.levelno >= logging.WARNING:
            _warning_queue.append({
                "level": record.levelname,
                "message": record.getMessage(),
                "timestamp": record.created,
                "module": record.module,
            })


def _install_warning_capture() -> None:
    base_logger = logging.getLogger("pixel_patrol_base")
    if not any(isinstance(h, _WarningCaptureHandler) for h in base_logger.handlers):
        handler = _WarningCaptureHandler()
        handler.setLevel(logging.INFO)
        base_logger.addHandler(handler)
        if base_logger.getEffectiveLevel() > logging.INFO:
            base_logger.setLevel(logging.INFO)


def get_warnings() -> list:
    return list(_warning_queue)


def clear_warnings() -> None:
    _warning_queue.clear()
    _console_lines.clear()


def _state_with_messages() -> Dict[str, Any]:
    state = get_state()
    state["warnings"] = get_warnings()
    state["console"] = list(_console_lines)
    return state


# ---------------------------------------------------------------------------
# Loaders / processors discovery
# ---------------------------------------------------------------------------

def _get_available_loaders() -> list:
    """List of available loaders with their names and supported extensions."""
    from pixel_patrol_base.plugin_registry import discover_plugins_from_entrypoints

    loaders = []

    for loader_class in sorted(discover_plugins_from_entrypoints("pixel_patrol.loader_plugins"), key=lambda c: c.NAME.lower()):
        try:
            extensions = sorted(getattr(loader_class, "SUPPORTED_EXTENSIONS", set()) or [])
        except Exception as e:
            logger.warning(f"Could not get extensions for loader {loader_class.NAME}: {e}")
            extensions = []
        loaders.append({"label": loader_class.NAME, "value": loader_class.NAME, "extensions": extensions})

    loaders.append({"label": "None (basic file info only)", "value": "", "extensions": []})
    return loaders


def _get_available_processors() -> list:
    """List of available processors as {id, name}."""
    from pixel_patrol_base.plugin_registry import discover_processor_plugins

    return [{"id": p.NAME, "name": p.NAME} for p in discover_processor_plugins()]


# ---------------------------------------------------------------------------
# Version check / self-update (launcher-managed installs only)
# ---------------------------------------------------------------------------

def _is_managed_install() -> bool:
    """True if this process is running from the launcher-managed venv."""
    try:
        return Path(sys.prefix).resolve() == _LAUNCHER_VENV.resolve()
    except OSError:
        return False


def _latest_pixel_patrol_version() -> Optional[str]:
    try:
        with urllib.request.urlopen(_PYPI_URL, timeout=3) as resp:
            data = json.loads(resp.read())
        return data["info"]["version"]
    except Exception:
        logger.debug("Could not check latest pixel-patrol version on PyPI", exc_info=True)
        return None


def _installed_version() -> Optional[str]:
    """Version of pixel-patrol, falling back to pixel-patrol-base if the
    full bundle isn't installed (e.g. dev/test environments)."""
    for name in ("pixel-patrol", "pixel-patrol-base"):
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


def _get_version_info() -> Dict[str, Any]:
    current = _installed_version()
    latest = _latest_pixel_patrol_version()

    update_available = False
    if latest and current:
        try:
            update_available = Version(latest) > Version(current)
        except InvalidVersion:
            update_available = latest != current

    return {
        "current": current,
        "latest": latest,
        "update_available": update_available,
        "managed": _is_managed_install(),
        "pypi_url": "https://pypi.org/project/pixel-patrol/",
    }


def _find_uv() -> Optional[Path]:
    hit = shutil.which("uv")
    if hit:
        return Path(hit)
    suffix = ".exe" if platform.system() == "Windows" else ""
    bundled = _LAUNCHER_HOME / "uv-bin" / f"uv{suffix}"
    return bundled if bundled.exists() else None


def _update_pixel_patrol() -> Dict[str, Any]:
    """Upgrade pixel-patrol (and configured loaders) in the managed venv."""
    uv = _find_uv()
    if uv is None:
        return {"error": "Could not find the uv package manager."}

    loader_pkgs: list = []
    try:
        config = json.loads((_LAUNCHER_HOME / "config.json").read_text())
        loader_pkgs = config.get("loader_pkgs", [])
    except Exception:
        pass

    pkgs = ["pixel-patrol", *loader_pkgs]
    try:
        subprocess.run(
            [str(uv), "pip", "install", "--python", sys.executable, "--upgrade", *pkgs],
            check=True, capture_output=True, text=True, timeout=300,
        )
    except subprocess.CalledProcessError as exc:
        return {"error": exc.stderr or str(exc)}
    except Exception as exc:
        return {"error": str(exc)}

    return {"status": "ok"}


def _list_directory(path: Path, with_files: bool = True) -> Dict[str, Any]:
    """List subdirectories (and .parquet files unless with_files is off), for the file picker."""
    entries = []
    for child in sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
        if child.is_dir():
            entries.append({"name": child.name, "is_dir": True})
        elif with_files and child.suffix.lower() == ".parquet":
            entries.append({"name": child.name, "is_dir": False})
    parent = path.parent
    return {
        "path": str(path),
        "parent": str(parent) if parent != path else None,
        "entries": entries,
    }


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------

def _parse_csv(value: Optional[str]) -> list:
    if not value:
        return []
    return [v.strip() for v in value.split(",") if v.strip()]


def _parse_slice_size(value: Optional[str]) -> Optional[Dict[str, int]]:
    """Parse 'Z=1, C=2' -> {'Z': 1, 'C': 2}. Raises ValueError on bad input."""
    if not value:
        return None
    result: Dict[str, int] = {}
    for item in _parse_csv(value):
        if "=" not in item:
            raise ValueError(f"Expected format DIM=SIZE (e.g. Z=1), got: {item!r}")
        dim, size = item.split("=", 1)
        try:
            result[dim.strip()] = int(size.strip())
        except ValueError:
            raise ValueError(f"Slice size must be an integer, got: {size!r}")
    return result


def _parse_dims(value: Optional[str]) -> Optional[Dict[str, str]]:
    """Parse 'z=0, c=1' -> {'z': '0', 'c': '1'}. Raises ValueError on bad input."""
    if not value:
        return None
    result: Dict[str, str] = {}
    for item in _parse_csv(value):
        if "=" not in item:
            raise ValueError(f"Expected format key=value (e.g. z=1), got: {item!r}")
        k, v = item.split("=", 1)
        result[k.strip()] = v.strip()
    return result or None


def _parse_filter(col: Optional[str], op: Optional[str], value: Optional[str]) -> Optional[Dict[str, Any]]:
    """Build a filter_by dict from form fields, or None if incomplete."""
    col, op, value = (col or "").strip(), (op or "").strip(), (value or "").strip()
    if col and op and value:
        return {col: {"op": op, "value": value}}
    return None


def _start_processing(payload: Dict[str, Any]) -> None:
    """Validate the request and start processing in a background thread."""
    state = get_state()
    if state["status"] == "running":
        return

    base_directory = (payload.get("base_directory") or "").strip()
    if not base_directory:
        update_state(status="error", error="Base directory is required")
        return

    output_path = (payload.get("output_path") or "").strip()
    if not output_path:
        output_path = str(library.default_output_path(base_directory, payload.get("project_name") or ""))
    payload["output_path"] = output_path

    try:
        slice_size = _parse_slice_size(payload.get("slice_size"))
    except ValueError as e:
        update_state(status="error", error=str(e))
        return

    clear_warnings()
    _cancel_event.clear()
    update_state(
        status="running",
        progress=0,
        message="Starting processing...",
        processed_files=0,
        total_files=0,
        error=None,
        output_parquet=None,
    )

    thread = threading.Thread(target=_run_processing, args=(payload, slice_size), daemon=True)
    thread.start()


def _run_processing(payload: Dict[str, Any], slice_size: Optional[Dict[str, int]]) -> None:
    try:
        base_dir = Path(payload["base_directory"]).resolve()
        if not base_dir.exists():
            update_state(status="error", error=f"Base directory does not exist: {payload['base_directory']}")
            return

        output_path = Path(payload["output_path"]).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        path_list = _parse_csv(payload.get("paths"))
        extensions = set(_parse_csv(payload.get("file_extensions"))) or "all"
        loader = payload.get("loader") or None
        project_name = (payload.get("project_name") or "").strip() or base_dir.name

        update_state(status="running", progress=5, message="Creating project...")

        project = api.create_project(project_name, str(base_dir), loader=loader, output_path=output_path)
        if path_list:
            api.add_paths(project, path_list)
        else:
            api.add_paths(project, base_dir)

        update_state(status="running", progress=15, message="Processing files...")

        def progress_callback(current: int, total: int) -> None:
            if _cancel_event.is_set():
                raise ProcessingCancelled()
            update_state(
                status="running",
                progress=min(20 + current, 85),
                message=f"Processing record {current}...",
                processed_files=current,
                total_files=max(total, 0),
            )

        scheduler = (payload.get("scheduler") or "").strip()
        max_workers = payload.get("max_workers")

        process_kwargs = dict(
            progress_callback=progress_callback,
            processors_included=set(payload.get("processors_include") or []) or None,
            processors_excluded=set(payload.get("processors_exclude") or []) or None,
            selected_file_extensions=extensions,
            mb_per_task=payload.get("mb_per_task"),
            max_images_per_task=payload.get("max_images_per_task"),
            slice_size=slice_size,
            rows_per_part=payload.get("rows_per_part"),
            parquet_row_group_size=payload.get("parquet_row_group_size"),
            flavor=(payload.get("flavor") or "").strip(),
            description=(payload.get("description") or "").strip(),
            log_file=bool(payload.get("log_file")),
        )

        if scheduler:
            from dask.distributed import Client
            with Client(scheduler):
                api.process_files(project, max_workers=None, **process_kwargs)
        else:
            api.process_files(project, max_workers=max_workers, **process_kwargs)

        update_state(status="running", progress=90, message="Processing complete. Finalizing...")

        final_parquet = project.output_path
        if not final_parquet.exists():
            raise FileNotFoundError(f"Processing completed but output file not found at '{final_parquet}'")
        if final_parquet.stat().st_size == 0:
            raise ValueError(f"Output file is empty: {final_parquet}")

        resolved = str(final_parquet)
        library.import_report(final_parquet)  # outputs saved elsewhere join the list
        update_state(
            status="completed",
            progress=100,
            message=f"Project saved to {resolved}",
            output_parquet=resolved,
        )
        logger.info(f"Processing completed. Output saved to: {resolved}")

    except ProcessingCancelled:
        logger.info("Processing cancelled by user")
        update_state(status="cancelled", message="Processing cancelled.")

    except Exception as e:
        logger.exception("Error during processing")
        update_state(status="error", error=f"Processing failed: {e}")


# ---------------------------------------------------------------------------
# Viewer support: one native DuckDB connection per report, opened on demand
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _viewer_dist() -> Path:
    return find_viewer_dist()


@lru_cache(maxsize=1)
def _extension_dirs() -> List[Path]:
    return _discover_installed_extensions()


_ReportConn = Tuple[Any, threading.Lock, Dict[str, str]]  # (connection, query lock, parquet meta)
_report_conns: Dict[Path, _ReportConn] = {}
_conns_lock = threading.Lock()


def _get_report_conn(path: Path) -> _ReportConn:
    with _conns_lock:
        if path not in _report_conns:
            conn, meta = _setup_duckdb(path)
            _report_conns[path] = (conn, threading.Lock(), meta)
        return _report_conns[path]


def _close_report_conn(path: Path) -> None:
    with _conns_lock:
        entry = _report_conns.pop(path, None)
    if entry:
        entry[0].close()


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

class _LaunchHandler(_ViewerHandler):
    """Report manager API + static pages, plus the inherited viewer endpoints.

    The viewer endpoints (``/api/query``, ``/api/export-parquet``, the viewer
    page itself) act on one report at a time; ``_bind_report`` points the
    inherited per-report attributes at the requested one.
    """

    report_id: Optional[str] = None

    @property
    def dist_dir(self) -> Path:
        return _viewer_dist()

    @property
    def extension_dirs(self) -> List[Path]:
        return _extension_dirs()

    def do_HEAD(self) -> None:
        if not self._allow_request():
            return
        file_path = self._find_asset(self.path.split("?")[0]) or ASSETS_DIR / "index.html"
        self.send_response(200)
        self._common_headers(_mime(file_path.suffix), file_path.stat().st_size)
        self.end_headers()

    def do_GET(self) -> None:
        if not self._allow_request():
            return
        path, _, query_string = self.path.partition("?")
        if path == "/api/loaders":
            self._send_json(_get_available_loaders())
        elif path == "/api/processors":
            self._send_json(_get_available_processors())
        elif path == "/api/status":
            self._send_json(_state_with_messages())
        elif path == "/api/version":
            self._send_json(_get_version_info())
        elif path == "/api/reports":
            self._handle_reports(query_string)
        elif path == "/api/browse":
            self._handle_browse()
        elif path == "/report":
            self._serve_report_page(query_string)
        elif path == "/api/export-parquet":
            if self._bind_report(query_string):
                self._handle_export_parquet(query_string)
        elif path.startswith("/extension/"):
            self._serve_extension_file(path)
        else:
            self._serve_asset(path)

    def do_POST(self) -> None:
        if not self._allow_request():
            return
        path, _, query_string = self.path.partition("?")
        if path == "/api/process":
            payload = self._read_json()
            if payload is not None:
                _start_processing(payload)
            self._send_json(_state_with_messages())
        elif path == "/api/cancel":
            if get_state()["status"] == "running":
                _cancel_event.set()
            self._send_json(_state_with_messages())
        elif path == "/api/report-url":
            self._handle_report_url(self._read_json() or {})
        elif path == "/api/import-report":
            self._handle_import_report(self._read_json() or {})
        elif path == "/api/delete-report":
            self._handle_delete_report(self._read_json() or {})
        elif path == "/api/query":
            if self._bind_report(query_string):
                self._serve_query()
        elif path == "/api/update":
            if not _is_managed_install():
                self._send_json(
                    {"error": "Update is only available for installations managed by the PixelPatrol launcher."},
                    status=400,
                )
            else:
                result = _update_pixel_patrol()
                self._send_json(result, status=200 if "error" not in result else 500)
        else:
            self.send_error(404)

    # ------------------------------------------------------------------
    # Report library
    # ------------------------------------------------------------------

    def _handle_reports(self, query_string: str) -> None:
        if parse_qs(query_string).get("refresh", ["0"])[0] in ("1", "true"):
            library.invalidate_meta()
        self._send_json({"reports_dir": str(library.REPORTS_DIR), "reports": library.scan_reports()})

    def _handle_import_report(self, payload: Dict[str, Any]) -> None:
        target = (payload.get("path") or "").strip()
        if not target:
            self._send_json({"error": "No report path given."}, status=400)
            return
        result = library.import_report(Path(target))
        self._send_json(result, status=result.pop("status", 200) if "error" in result else 200)

    def _handle_delete_report(self, payload: Dict[str, Any]) -> None:
        target = (payload.get("path") or "").strip()
        if not target:
            self._send_json({"error": "No report path given."}, status=400)
            return
        known = library.find_known(target)
        if known is None:
            self._send_json({"error": f"Unknown report: {target}"}, status=404)
            return
        _close_report_conn(known)
        if library.delete_report(known):
            self._send_json({"status": "ok"})
        else:
            self._send_json({"error": f"Could not delete report: {target}"}, status=400)

    # ------------------------------------------------------------------
    def _handle_browse(self) -> None:
        query = parse_qs(urlsplit(self.path).query)
        raw_path = query.get("path", [str(Path.home())])[0]

        target = Path(raw_path).expanduser()
        try:
            target = target.resolve()
        except OSError:
            self._send_json({"error": f"Invalid path: {raw_path}"}, status=400)
            return

        if not target.exists() or not target.is_dir():
            self._send_json({"error": f"Not a directory: {target}"}, status=404)
            return

        try:
            self._send_json(_list_directory(target, with_files=query.get("files", ["1"])[0] != "0"))
        except PermissionError:
            self._send_json({"error": f"Permission denied: {target}"}, status=403)

    # ------------------------------------------------------------------
    # Viewer
    # ------------------------------------------------------------------

    def _handle_report_url(self, payload: Dict[str, Any]) -> None:
        """Build the in-app ``/report?path=...`` URL, encoding the initial viewer state."""
        output_parquet = payload.get("output_parquet") or payload.get("path")
        if not output_parquet:
            self._send_json({"error": "No report path given."}, status=400)
            return

        parquet_path = library.find_known(output_parquet)
        if parquet_path is None or not parquet_path.exists():
            self._send_json({"error": f"Report not found: {output_parquet}"}, status=404)
            return

        try:
            dimensions = _parse_dims(payload.get("dimensions"))
        except ValueError as exc:
            self._send_json({"error": str(exc)}, status=400)
            return

        viewer_qs = build_viewer_url_params(
            group_col=(payload.get("group_by") or "").strip() or None,
            filter_by=_parse_filter(payload.get("filter_col"), payload.get("filter_op"), payload.get("filter_value")),
            dimensions=dimensions,
            widgets_excluded=set(_parse_csv(payload.get("widgets_exclude"))) or None,
            is_show_significance=bool(payload.get("is_show_significance")),
            palette=(payload.get("palette") or "").strip() or None,
        )
        url = f"/report?path={quote(str(parquet_path))}"
        self._send_json({"url": f"{url}&{viewer_qs}" if viewer_qs else url})

    def _bind_report(self, query_string: str) -> bool:
        """Point the viewer handler at the report named by ``?report=<id>``."""
        path = library.resolve_report(parse_qs(query_string).get("report", [""])[0])
        if path is None:
            self._send_error_text(404, "Unknown report")
            return False
        return self._use_report(path)

    def _use_report(self, path: Path) -> bool:
        try:
            self.duck_conn, self.query_lock, self.parquet_meta = _get_report_conn(path)
        except Exception as exc:
            logger.exception("Failed to open report")
            self._send_error_text(500, f"Failed to open report: {exc}")
            return False
        self.parquet_path = path
        self.report_id = library.register_report(path)
        self.project_name = self.parquet_meta.get("pp_project_name") or None
        self.description = self.parquet_meta.get("pp_description") or None
        return True

    def _serve_report_page(self, query_string: str) -> None:
        raw_path = parse_qs(query_string).get("path", [""])[0]
        if not raw_path:
            self.send_response(302)
            self.send_header("Location", "/")
            self.end_headers()
            return
        parquet_path = library.find_known(raw_path)
        if parquet_path is None or not parquet_path.is_file():
            self._send_error_text(404, f"Report not found: {raw_path}")
            return
        if self._use_report(parquet_path):
            self._serve_static("/index.html")

    def _inject_server_config(self, html: bytes) -> bytes:
        """Add the report id so the viewer routes its queries to this report."""
        html = super()._inject_server_config(html)
        script = f"<script>window.__PP_REPORT_ID = {_js(self.report_id)};</script>\n".encode()
        return html.replace(b"</head>", script + b"</head>", 1)

    # ------------------------------------------------------------------
    # Static file serving: manager assets first, then the viewer's dist assets
    # ------------------------------------------------------------------

    def _find_asset(self, url_path: str) -> Optional[Path]:
        rel = url_path.lstrip("/") or "index.html"
        roots = [ASSETS_DIR]
        try:
            roots.append(self.dist_dir)
        except FileNotFoundError:
            pass  # viewer not built: the manager itself still works
        for root in roots:
            candidate = os.path.normpath(os.path.join(root, rel))
            if candidate.startswith(str(root) + os.sep) and os.path.isfile(candidate):
                return Path(candidate)
        return None

    def _serve_asset(self, url_path: str) -> None:
        file_path = self._find_asset(url_path) or ASSETS_DIR / "index.html"
        data = file_path.read_bytes()
        self.send_response(200)
        self._common_headers(_mime(file_path.suffix), len(data))
        self.end_headers()
        self.wfile.write(data)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _read_json(self) -> Optional[Dict[str, Any]]:
        try:
            length = int(self.headers.get("Content-Length", 0))
            return json.loads(self.rfile.read(length))
        except Exception as exc:
            self._send_json({"error": f"Bad request: {exc}"}, status=400)
            return None

    def _send_json(self, data: Any, status: int = 200) -> None:
        body = json.dumps(data).encode()
        self.send_response(status)
        self._common_headers("application/json", len(body))
        self.end_headers()
        self.wfile.write(body)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def serve_launch(port: int = 8051, open_browser: bool = True) -> None:
    """Start the report manager server and (optionally) open it in the browser."""
    _install_warning_capture()
    library.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    chosen_port = port
    try:
        server = ThreadingHTTPServer(("127.0.0.1", chosen_port), _LaunchHandler)
    except OSError as exc:
        if exc.errno not in {getattr(socket, "EADDRINUSE", 98), 98}:
            raise
        server = ThreadingHTTPServer(("127.0.0.1", 0), _LaunchHandler)

    chosen_port = int(server.server_address[1])
    url = f"http://127.0.0.1:{chosen_port}/"

    import click
    click.echo(f"PixelPatrol report manager: {url}")
    click.echo(f"Reports directory: {library.REPORTS_DIR}")
    click.echo("Press Ctrl+C to stop.\n")

    if open_browser:
        threading.Timer(0.6, webbrowser.open, args=[url]).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


__all__ = ["serve_launch"]
