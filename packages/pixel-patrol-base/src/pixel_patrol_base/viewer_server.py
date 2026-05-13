"""
Local HTTP server for the static Pixel Patrol viewer.

Serves the pre-built viewer SPA and the requested parquet file over HTTP,
including byte-range support so DuckDB WASM can stream only the row groups
it actually needs rather than downloading the whole file up front.

When a parquet file is served locally, all SQL queries are executed by a
native (multi-threaded) DuckDB connection in Python instead of DuckDB WASM
in the browser.  The browser viewer detects window.__PP_SERVER and routes
every conn.query() call to POST /api/query, which returns Arrow IPC bytes.
This is significantly faster than WASM for large datasets.

COOP/COEP headers are still sent so that if the user opens the viewer on a
remote static host (GitHub Pages etc.) the multi-threaded WASM bundle can
also be used.
"""

from __future__ import annotations

import json
import socket
import threading
import warnings
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional
import importlib.resources
from urllib.parse import urlencode

# ---------------------------------------------------------------------------
# Installed extension discovery
# ---------------------------------------------------------------------------

def _discover_installed_extensions() -> list[Path]:
    """
    Return viewer extension directories declared by installed packages.

    Each entry point in the ``pixel_patrol.viewer_extensions`` group must be a
    callable that returns the path to a directory containing ``extension.json``.
    """
    try:
        from importlib.metadata import entry_points
        dirs: list[Path] = []
        eps = sorted(entry_points(group="pixel_patrol.viewer_extensions"), key=lambda ep: (ep.name, ep.value))
        for ep in eps:
            try:
                fn = ep.load()
                d  = Path(fn())
                if d.is_dir() and (d / "extension.json").exists():
                    dirs.append(d)
                else:
                    warnings.warn(
                        f"[pixel-patrol] viewer extension {ep.name!r}: "
                        f"{d} is not a directory containing extension.json"
                    )
            except Exception as exc:
                warnings.warn(f"[pixel-patrol] viewer extension {ep.name!r} failed to load: {exc}")
        return dirs
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Viewer dist discovery
# ---------------------------------------------------------------------------

def find_viewer_dist() -> Path:
    """
    Return the path to the built viewer ``dist/`` directory.

    Search order:
    1. Installed package data  (``pixel_patrol_base/viewer_dist/``)
    2. Source-tree sibling     (``<repo>/viewer/dist/``) — editable installs
    """
    try:
        pkg   = importlib.resources.files("pixel_patrol_base").joinpath("viewer_dist")
        index = pkg.joinpath("index.html")
        if index.is_file():
            with importlib.resources.as_file(pkg) as p:
                return p.resolve()
    except Exception:
        pass

    here      = Path(__file__).resolve().parent
    repo_root = here.parents[3]
    candidate = repo_root / "viewer" / "dist"
    if (candidate / "index.html").exists():
        return candidate.resolve()

    raise FileNotFoundError(
        "Viewer not built.\n"
        "Run:  cd viewer && npm install && npm run build\n"
        "Then retry the command."
    )


# ---------------------------------------------------------------------------
# Native DuckDB setup
# ---------------------------------------------------------------------------

def _setup_duckdb(parquet_path: Path):
    """
    Open a native DuckDB connection with pp_data pre-registered as a view.
    Returns (conn, project_name | None, description | None).
    """
    import duckdb

    conn = duckdb.connect()
    escaped = str(parquet_path).replace("'", "''")
    conn.execute(
        f"CREATE VIEW pp_data AS "
        f"SELECT *, row_number() OVER () - 1 AS file_row_number "
        f"FROM read_parquet('{escaped}')"
    )

    project_name, description = _read_parquet_meta(conn, escaped)
    return conn, project_name, description


def _read_parquet_meta(conn, escaped_path: str):
    """Read pp_project_name and pp_description from the parquet file's KV metadata."""
    try:
        rows = conn.execute(
            f"SELECT decode(key)::VARCHAR AS k, decode(value)::VARCHAR AS v "
            f"FROM parquet_kv_metadata('{escaped_path}') "
            f"WHERE decode(key)::VARCHAR IN ('pp_project_name', 'pp_description')"
        ).fetchall()
        meta = {k: v for k, v in rows}
        return meta.get("pp_project_name") or None, meta.get("pp_description") or None
    except Exception:
        return None, None


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

class _ViewerHandler(BaseHTTPRequestHandler):
    """
    Serves the viewer SPA from ``dist_dir`` and the parquet file at
    ``/data.parquet``, with byte-range support and the COOP/COEP headers
    needed for SharedArrayBuffer.

    When running locally, POST /api/query executes SQL via a native DuckDB
    connection and returns the result as an Arrow IPC stream.
    """

    dist_dir:         Path
    parquet_path:     Path
    duck_conn:        object   # duckdb.DuckDBPyConnection
    query_lock:       threading.Lock
    project_name:     Optional[str]
    description:      Optional[str]
    extension_dirs:   list  # list[Path] — each dir contains extension.json + plugin JS files

    # ------------------------------------------------------------------
    def do_HEAD(self) -> None:
        path = self.path.split("?")[0]
        if path == "/data.parquet":
            self._send_parquet_head(self.parquet_path)
        else:
            self._send_static_head(path)

    def do_GET(self) -> None:
        path, _, query_string = self.path.partition("?")
        if path == "/data.parquet":
            self._serve_parquet(self.parquet_path)
        elif path == "/api/export-parquet":
            self._handle_export_parquet(query_string)
        elif path.startswith("/extension/"):
            self._serve_extension_file(path)
        else:
            self._serve_static(path)


    def do_POST(self) -> None:
        if self.path == "/api/query":
            self._serve_query()
        else:
            self.send_error(404)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self._common_headers("text/plain", 0)
        self.end_headers()

    # ------------------------------------------------------------------
    # /api/query  — native DuckDB execution
    # ------------------------------------------------------------------

    def _serve_query(self) -> None:
        import pyarrow as pa
        import pyarrow.ipc as ipc

        try:
            length = int(self.headers.get("Content-Length", 0))
            body   = json.loads(self.rfile.read(length))
            sql    = body["sql"]
        except Exception as exc:
            self._send_error_text(400, f"Bad request: {exc}")
            return

        try:
            with self.query_lock:
                arrow_table = self.duck_conn.execute(sql).fetch_arrow_table()

            sink   = pa.BufferOutputStream()
            writer = ipc.new_stream(sink, arrow_table.schema)
            writer.write_table(arrow_table)
            writer.close()
            data = bytes(sink.getvalue())

            self.send_response(200)
            self._common_headers("application/vnd.apache.arrow.stream", len(data))
            self.end_headers()
            self.wfile.write(data)

        except Exception as exc:
            self._send_error_text(400, str(exc))


    # ------------------------------------------------------------------
    # /api/export-parquet  — filtered parquet download with metadata
    # ------------------------------------------------------------------

    def _handle_export_parquet(self, query_string: str) -> None:
        import tempfile
        import urllib.parse
        from pixel_patrol_base.io.parquet_io import reattach_parquet_metadata

        try:
            params = urllib.parse.parse_qs(query_string)
            where = params.get("where", [""])[0]

            with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
                tmp_path = f.name

            sql = (
                f"COPY ("
                f"  SELECT * EXCLUDE (file_row_number) FROM pp_data {where}"
                f") TO '{tmp_path}' "
                f"(FORMAT parquet, COMPRESSION snappy, ROW_GROUP_SIZE 2048)"
            )
            with self.query_lock:
                self.duck_conn.execute(sql)

            reattach_parquet_metadata(Path(tmp_path), self.parquet_path)

            data = Path(tmp_path).read_bytes()
            Path(tmp_path).unlink(missing_ok=True)

            stem = self.parquet_path.stem
            filename = f"{stem}_filtered.parquet" if where else f"{stem}.parquet"

            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(data)

        except Exception as exc:
            self._send_error_text(500, f"Parquet export failed: {exc}")

    def _send_error_text(self, code: int, msg: str) -> None:
        body = msg.encode()
        self.send_response(code)
        self._common_headers("text/plain; charset=utf-8", len(body))
        self.end_headers()
        self.wfile.write(body)

    # ------------------------------------------------------------------
    # Parquet endpoint (with byte-range support)
    # ------------------------------------------------------------------

    def _serve_parquet(self, path: Path) -> None:
        size         = path.stat().st_size
        range_header = self.headers.get("Range")

        if range_header:
            start, end = _parse_range(range_header, size)
            length      = end - start + 1
            self.send_response(206)
            self._common_headers("application/octet-stream", length)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            with open(path, "rb") as f:
                f.seek(start)
                self._write_chunks(f, length)
        else:
            self.send_response(200)
            self._common_headers("application/octet-stream", size)
            self.end_headers()
            with open(path, "rb") as f:
                self._write_chunks(f, size)

    def _send_parquet_head(self, path: Path) -> None:
        size = path.stat().st_size
        self.send_response(200)
        self._common_headers("application/octet-stream", size)
        self.end_headers()

    # ------------------------------------------------------------------
    # Static file serving  (injects __PP_SERVER config into index.html)
    # ------------------------------------------------------------------

    def _serve_static(self, url_path: str) -> None:
        rel       = url_path.lstrip("/") or "index.html"
        file_path = (self.dist_dir / rel).resolve()

        try:
            file_path.relative_to(self.dist_dir)
        except ValueError:
            self.send_error(403)
            return

        if not file_path.exists() or not file_path.is_file():
            file_path = self.dist_dir / "index.html"

        content_type = _mime(file_path.suffix)
        data         = file_path.read_bytes()

        if file_path.name == "index.html":
            data = self._inject_server_config(data)
            content_type = "text/html; charset=utf-8"

        self.send_response(200)
        self._common_headers(content_type, len(data))
        self.end_headers()
        self.wfile.write(data)

    def _send_static_head(self, url_path: str) -> None:
        rel       = url_path.lstrip("/") or "index.html"
        file_path = self.dist_dir / rel
        if not file_path.is_file():
            file_path = self.dist_dir / "index.html"
        size = len(self._inject_server_config(file_path.read_bytes())) \
               if file_path.name == "index.html" else file_path.stat().st_size
        self.send_response(200)
        self._common_headers(_mime(file_path.suffix), size)
        self.end_headers()

    def _serve_extension_file(self, url_path: str) -> None:
        """Serve extension.json or a plugin JS file from extension_dirs[idx].

        URL format: /extension/{idx}/{filename}
        """
        parts = url_path.strip("/").split("/")  # ["extension", "{idx}", "{filename}"]
        if len(parts) != 3:
            self.send_error(404)
            return
        try:
            idx = int(parts[1])
        except ValueError:
            self.send_error(404)
            return
        if idx < 0 or idx >= len(self.extension_dirs):
            self.send_error(404)
            return

        ext_dir   = self.extension_dirs[idx]
        filename  = parts[2]
        file_path = (ext_dir / filename).resolve()
        try:
            file_path.relative_to(ext_dir)
        except ValueError:
            self.send_error(403)
            return
        if not file_path.is_file():
            self.send_error(404)
            return
        data         = file_path.read_bytes()
        content_type = "application/json" if file_path.suffix == ".json" else "application/javascript"
        self.send_response(200)
        self._common_headers(content_type, len(data))
        self.end_headers()
        self.wfile.write(data)

    def _inject_server_config(self, html: bytes) -> bytes:
        """Inject window.__PP_* config variables before </head>."""
        extension_urls = [f"/extension/{i}/extension.json" for i in range(len(self.extension_dirs))]
        script = (
            "<script>\n"
            "window.__PP_SERVER = true;\n"
            f"window.__PP_FILENAME = {json.dumps(self.parquet_path.name)};\n"
            f"window.__PP_PROJECT_NAME = {json.dumps(self.project_name)};\n"
            f"window.__PP_DESCRIPTION = {json.dumps(self.description)};\n"
            f"window.__PP_EXTENSION_URLS = {json.dumps(extension_urls)};\n"
            "</script>\n"
        ).encode()
        return html.replace(b"</head>", script + b"</head>", 1)

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _common_headers(self, content_type: str, length: int) -> None:
        self.send_header("Content-Type",                 content_type)
        self.send_header("Content-Length",               str(length))
        self.send_header("Accept-Ranges",                "bytes")
        self.send_header("Cross-Origin-Opener-Policy",   "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Access-Control-Allow-Origin",  "*")
        self.send_header("Access-Control-Allow-Headers", "Range, Content-Type")
        self.send_header("Cache-Control",                "no-cache")

    def _write_chunks(self, f, length: int, chunk: int = 1 << 20) -> None:
        remaining = length
        while remaining > 0:
            data = f.read(min(chunk, remaining))
            if not data:
                break
            self.wfile.write(data)
            remaining -= len(data)

    def log_message(self, fmt: str, *args) -> None:
        pass


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def serve_viewer(
    parquet_path:  Path | str,
    port:          int            = 8052,
    open_browser:  bool           = True,
    dist_dir:      Optional[Path] = None,
    extension:     Optional[Path] = None,
    group_col: Optional[str] = None,
    filter_by: Optional[dict] = None,
    dimensions: Optional[dict] = None,
    widgets_excluded: Optional[set] = None,
    is_show_significance: bool = False,
    palette: Optional[str] = None,
) -> None:
    """
    Start a local HTTP server and (optionally) open the viewer in the browser.

    Parameters
    ----------
    parquet_path:
        Path to the ``.parquet`` file to view.
    port:
        TCP port to listen on (default 8052).
    open_browser:
        Open the default browser automatically (default True).
    dist_dir:
        Override the viewer dist directory (useful for testing).
    extension:
        Path to a local directory containing ``extension.json`` and plugin JS
        files.  Added on top of any extensions auto-discovered from installed
        packages (``pixel_patrol.viewer_extensions`` entry-point group).
        Mirrors the ``?extension=<url>`` URL parameter used when the viewer is
        hosted remotely.
    group_col, filter_by, dimensions, widgets_excluded, is_show_significance, palette:
        Initial viewer state, encoded into the viewer URL as query parameters.
    """
    parquet_path = Path(parquet_path).resolve()
    if not parquet_path.exists():
        raise FileNotFoundError(parquet_path)

    dist_dir = dist_dir or find_viewer_dist()

    # Collect extension directories: auto-discovered first, then explicit.
    extension_dirs: list[Path] = _discover_installed_extensions()
    if extension:
        extension_dirs.append(Path(extension).resolve())

    import click
    click.echo(f"Serving parquet  : {parquet_path}")
    for d in extension_dirs:
        click.echo(f"Extension        : {d}")

    duck_conn, project_name, description = _setup_duckdb(parquet_path)
    query_lock = threading.Lock()

    handler = type(
        "_Handler",
        (_ViewerHandler,),
        {
            "dist_dir":       dist_dir,
            "parquet_path":   parquet_path,
            "duck_conn":      duck_conn,
            "query_lock":     query_lock,
            "project_name":   project_name,
            "description":    description,
            "extension_dirs": extension_dirs,
        },
    )

    chosen_port = port
    try:
        server = ThreadingHTTPServer(("127.0.0.1", chosen_port), handler)
    except OSError as exc:
        # If the requested port is occupied, transparently fall back to an
        # ephemeral port so `pixel-patrol view` still starts.
        if exc.errno not in {getattr(socket, "EADDRINUSE", 98), 98}:
            raise

        click.echo(f"Port {chosen_port} is in use; selecting a free port.")
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        chosen_port = int(server.server_address[1])

    qs = build_viewer_url_params(
        group_col=group_col,
        filter_by=filter_by,
        dimensions=dimensions,
        widgets_excluded=widgets_excluded,
        is_show_significance=is_show_significance,
        palette=palette,
    )
    viewer_url = f"http://127.0.0.1:{chosen_port}/{'?' + qs if qs else ''}"

    click.echo(f"Viewer URL       : {viewer_url}")
    click.echo("Press Ctrl+C to stop.\n")

    if open_browser:
        threading.Timer(0.6, webbrowser.open, args=[viewer_url]).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        duck_conn.close()


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _parse_range(header: str, total: int) -> tuple[int, int]:
    """Parse ``Range: bytes=start-end`` and clamp to file size."""
    spec          = header.split("=", 1)[1]
    start_s, _, end_s = spec.partition("-")
    start = int(start_s) if start_s else 0
    end   = int(end_s)   if end_s   else total - 1
    start = max(0, min(start, total - 1))
    end   = max(start, min(end, total - 1))
    return start, end


_MIME = {
    ".html": "text/html; charset=utf-8",
    ".js":   "application/javascript",
    ".mjs":  "application/javascript",
    ".css":  "text/css",
    ".json": "application/json",
    ".wasm": "application/wasm",
    ".svg":  "image/svg+xml",
    ".png":  "image/png",
    ".ico":  "image/x-icon",
}

def _mime(suffix: str) -> str:
    return _MIME.get(suffix.lower(), "application/octet-stream")


def build_viewer_url_params(
    group_col: Optional[str] = None,
    filter_by: Optional[dict] = None,
    dimensions: Optional[dict] = None,
    widgets_excluded: Optional[set] = None,
    is_show_significance: bool = False,
    palette: Optional[str] = None,
) -> str:
    """
    Build a URL query string encoding viewer state.
    Matches the param names in viewer/src/url-params.js.
    Returns an empty string when no state is set.
    """

    params = {}

    if group_col:
        params["group"] = group_col

    if filter_by:
        col = next(iter(filter_by))
        params["fc"] = col
        params["fo"] = filter_by[col]["op"]
        params["fv"] = filter_by[col]["value"]

    if dimensions:
        # {"t": "0", "c": "1"} → "t0.c1"
        params["dims"] = ".".join(f"{k}{v}" for k, v in dimensions.items())

    if is_show_significance:
        params["sig"] = "1"

    if palette and palette != "tab10":
        params["palette"] = palette

    if widgets_excluded:
        params["hidden"] = ".".join(sorted(widgets_excluded))

    return urlencode(params)

