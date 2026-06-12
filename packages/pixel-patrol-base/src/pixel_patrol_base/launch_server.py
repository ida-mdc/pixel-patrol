"""
Local HTTP server for the Pixel Patrol processing launch page.

Serves a small static JS/HTML frontend (``launch_assets/``) that configures
and monitors a Pixel Patrol processing run, then hands off to the existing
JS report viewer (``viewer_server.serve_viewer``) once a parquet file has
been produced.

Mirrors the architecture of ``viewer_server.py``: a plain
``http.server.ThreadingHTTPServer`` serving static assets plus a small JSON
API, no Dash/Flask dependency required.
"""

from __future__ import annotations

import importlib.metadata
import json
import logging
import platform
import shutil
import socket
import subprocess
import sys
import threading
import urllib.request
import webbrowser
from urllib.parse import parse_qs, urlsplit
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional

from packaging.version import InvalidVersion, Version

from pixel_patrol_base import api
from pixel_patrol_base.viewer_server import build_viewer_url_params, serve_viewer

logger = logging.getLogger(__name__)

ASSETS_DIR = (Path(__file__).parent / "launch_assets").resolve()

# ~/.pixel-patrol is created by the Pixel Patrol launcher (deploy/launcher);
# its presence as our venv root is how we detect a launcher-managed install.
_LAUNCHER_HOME = Path.home() / ".pixel-patrol"
_LAUNCHER_VENV = _LAUNCHER_HOME / "venv"
_PYPI_URL = "https://pypi.org/pypi/pixel-patrol/json"

_MIME = {
    ".html": "text/html; charset=utf-8",
    ".js":   "application/javascript",
    ".css":  "text/css",
    ".json": "application/json",
    ".png":  "image/png",
    ".svg":  "image/svg+xml",
    ".ico":  "image/x-icon",
}


def _mime(suffix: str) -> str:
    return _MIME.get(suffix.lower(), "application/octet-stream")


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


class _WarningCaptureHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
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
        handler.setLevel(logging.WARNING)
        base_logger.addHandler(handler)


def get_warnings() -> list:
    return list(_warning_queue)


def clear_warnings() -> None:
    _warning_queue.clear()


# ---------------------------------------------------------------------------
# Loaders / processors discovery
# ---------------------------------------------------------------------------

def _get_available_loaders() -> list:
    """List of available loaders with their names and supported extensions."""
    from pixel_patrol_base.plugin_registry import discover_plugins_from_entrypoints

    loaders = [{"label": "None (basic file info only)", "value": "", "extensions": []}]

    for loader_class in discover_plugins_from_entrypoints("pixel_patrol.loader_plugins"):
        try:
            extensions = sorted(getattr(loader_class, "SUPPORTED_EXTENSIONS", set()) or [])
        except Exception as e:
            logger.warning(f"Could not get extensions for loader {loader_class.NAME}: {e}")
            extensions = []
        loaders.append({"label": loader_class.NAME, "value": loader_class.NAME, "extensions": extensions})

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


def _list_directory(path: Path) -> Dict[str, Any]:
    """List subdirectories and .parquet files in path, for the report file picker."""
    entries = []
    for child in sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
        if child.is_dir():
            entries.append({"name": child.name, "is_dir": True})
        elif child.suffix.lower() == ".parquet":
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
    output_path = (payload.get("output_path") or "").strip()
    if not base_directory or not output_path:
        update_state(status="error", error="Base directory and output parquet path are required")
        return

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
# Viewer launching - one viewer server per parquet file, started on demand
# ---------------------------------------------------------------------------

_viewer_lock = threading.Lock()
_viewer_servers: Dict[str, str] = {}  # resolved parquet path -> viewer URL


def _get_or_launch_viewer(parquet_path: Path) -> str:
    key = str(parquet_path)
    with _viewer_lock:
        url = _viewer_servers.get(key)
        if url is not None:
            return url

        ready = threading.Event()
        result: Dict[str, str] = {}

        def on_ready(_port: int, viewer_url: str) -> None:
            result["url"] = viewer_url
            ready.set()

        thread = threading.Thread(
            target=serve_viewer,
            kwargs=dict(parquet_path=parquet_path, port=0, open_browser=False, ready_callback=on_ready),
            daemon=True,
        )
        thread.start()

        if not ready.wait(timeout=15):
            raise RuntimeError("Timed out starting the viewer server")

        url = result["url"]
        _viewer_servers[key] = url
        return url


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

class _LaunchHandler(BaseHTTPRequestHandler):

    def do_HEAD(self) -> None:
        path = self.path.split("?")[0]
        rel = path.lstrip("/") or "index.html"
        file_path = (ASSETS_DIR / rel).resolve()
        try:
            file_path.relative_to(ASSETS_DIR)
        except ValueError:
            self.send_error(403)
            return
        if not file_path.exists() or not file_path.is_file():
            file_path = ASSETS_DIR / "index.html"
        self.send_response(200)
        self.send_header("Content-Type", _mime(file_path.suffix))
        self.send_header("Content-Length", str(file_path.stat().st_size))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

    def do_GET(self) -> None:
        path = self.path.split("?")[0]
        if path == "/api/loaders":
            self._send_json(_get_available_loaders())
        elif path == "/api/processors":
            self._send_json(_get_available_processors())
        elif path == "/api/status":
            state = get_state()
            state["warnings"] = get_warnings()
            self._send_json(state)
        elif path == "/api/version":
            self._send_json(_get_version_info())
        elif path == "/api/browse":
            self._handle_browse()
        else:
            self._serve_static(path)

    def do_POST(self) -> None:
        if self.path == "/api/process":
            payload = self._read_json()
            if payload is not None:
                _start_processing(payload)
            state = get_state()
            state["warnings"] = get_warnings()
            self._send_json(state)
        elif self.path == "/api/cancel":
            if get_state()["status"] == "running":
                _cancel_event.set()
            state = get_state()
            state["warnings"] = get_warnings()
            self._send_json(state)
        elif self.path == "/api/open-viewer":
            payload = self._read_json() or {}
            self._handle_open_viewer(payload)
        elif self.path == "/api/update":
            if not _is_managed_install():
                self._send_json(
                    {"error": "Update is only available for installations managed by the Pixel Patrol launcher."},
                    status=400,
                )
            else:
                result = _update_pixel_patrol()
                self._send_json(result, status=200 if "error" not in result else 500)
        else:
            self.send_error(404)

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
            self._send_json(_list_directory(target))
        except PermissionError:
            self._send_json({"error": f"Permission denied: {target}"}, status=403)

    # ------------------------------------------------------------------
    def _handle_open_viewer(self, payload: Dict[str, Any]) -> None:
        output_parquet = payload.get("output_parquet")
        if not output_parquet:
            self._send_json({"error": "No output file available. Please run processing first."}, status=400)
            return

        parquet_path = Path(output_parquet).resolve()
        if not parquet_path.exists():
            self._send_json({"error": f"Output file not found: {parquet_path}"}, status=404)
            return

        try:
            url = _get_or_launch_viewer(parquet_path)
        except Exception as exc:
            logger.exception("Failed to launch viewer")
            self._send_json({"error": f"Failed to launch viewer: {exc}"}, status=500)
            return

        try:
            dimensions = _parse_dims(payload.get("dimensions"))
        except ValueError as exc:
            self._send_json({"error": str(exc)}, status=400)
            return

        query = build_viewer_url_params(
            group_col=(payload.get("group_by") or "").strip() or None,
            filter_by=_parse_filter(payload.get("filter_col"), payload.get("filter_op"), payload.get("filter_value")),
            dimensions=dimensions,
            widgets_excluded=set(_parse_csv(payload.get("widgets_exclude"))) or None,
            is_show_significance=bool(payload.get("is_show_significance")),
            palette=(payload.get("palette") or "").strip() or None,
        )
        if query:
            url = f"{url}?{query}"

        self._send_json({"url": url})

    # ------------------------------------------------------------------
    # Static file serving
    # ------------------------------------------------------------------

    def _serve_static(self, url_path: str) -> None:
        rel = url_path.lstrip("/") or "index.html"
        file_path = (ASSETS_DIR / rel).resolve()

        try:
            file_path.relative_to(ASSETS_DIR)
        except ValueError:
            self.send_error(403)
            return

        if not file_path.exists() or not file_path.is_file():
            file_path = ASSETS_DIR / "index.html"

        data = file_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", _mime(file_path.suffix))
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
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
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args) -> None:
        pass


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def serve_launch(port: int = 8051, open_browser: bool = True) -> None:
    """Start the processing launch server and (optionally) open it in the browser."""
    _install_warning_capture()

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
    click.echo(f"Processing dashboard URL: {url}")
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
