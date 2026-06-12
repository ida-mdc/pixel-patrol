#!/usr/bin/env python3
"""Pixel Patrol Launcher
------------------------
Double-click this binary to run Pixel Patrol.

• First run  : serves a local setup wizard in your browser, downloads uv,
               creates a managed venv at ~/.pixel-patrol/venv, installs
               packages, then hands off to pixel-patrol launch.
• Later runs : calls pixel-patrol launch directly.
"""

from __future__ import annotations

import base64
import io
import json
import os
import platform
import stat
import subprocess
import sys
import tarfile
import threading
import urllib.request
import webbrowser
import zipfile
from pathlib import Path

from flask import Flask, Response, request, stream_with_context

# ── Embedded assets ────────────────────────────────────────────────────────────
# Images are read relative to this file at build time so PyInstaller can bundle
# them; fall back to empty strings if not found (binary still runs fine).

def _b64_png(rel_path: str) -> str:
    here = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    repo_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
    # also look relative to the source tree root, where the viewer and launch
    # page already ship the same branding assets
    candidates = [
        os.path.join(here, rel_path),
        os.path.join(repo_root, "viewer", "public", os.path.basename(rel_path)),
        os.path.join(repo_root, "packages", "pixel-patrol-base", "src",
                      "pixel_patrol_base", "launch_assets", os.path.basename(rel_path)),
    ]
    for p in candidates:
        try:
            with open(p, "rb") as f:
                return "data:image/png;base64," + base64.b64encode(f.read()).decode()
        except FileNotFoundError:
            pass
    return ""

_IMG_ICON      = _b64_png("icon.png")
_IMG_LOGO      = _b64_png("prevalidation.png")
_IMG_HELMHOLTZ = _b64_png("Helmholtz-Imaging_Mark.png")

# ── Version ────────────────────────────────────────────────────────────────────

# Kept in sync with packages/pixel-patrol's version by tools/bump_toml_version.py.
LAUNCHER_VERSION = "0.7.0"

# ── App directories ────────────────────────────────────────────────────────────

APP_DIR     = Path.home() / ".pixel-patrol"
VENV_DIR    = APP_DIR / "venv"
CONFIG_FILE = APP_DIR / "config.json"
UV_BIN_DIR  = APP_DIR / "uv-bin"

# ── Package catalogue ──────────────────────────────────────────────────────────

# pixel-patrol (the full bundle: base + image quality + bioio loaders) is
# always installed silently.
BASE_PACKAGES = ["pixel-patrol"]

# Selectable add-on packages — shown as cards in the setup wizard, for
# packages not already included in the pixel-patrol bundle above.
# Keys:
#   id          unique string used in the HTML form
#   label       display name
#   package     PyPI package to install
#   use_case    one-liner shown below the title
#   processes   what the processor analyses (None if no processor)
#   widgets     list of report widget names added by this package
#   extensions  input file extensions supported (empty list if no loader)
#   default     pre-checked in the UI
# Empty for now - the pixel-patrol bundle above covers the packages relevant
# to most users. Add entries here for optional add-ons worth surfacing to a
# general audience (e.g. niche loaders for specific projects/communities).
PACKAGES: list[dict] = []

# ── uv helpers ─────────────────────────────────────────────────────────────────

def _uv_download_url() -> str:
    system  = platform.system()
    machine = platform.machine().lower()
    if system == "Linux":
        arch = "aarch64" if "aarch64" in machine or "arm64" in machine else "x86_64"
        return f"https://github.com/astral-sh/uv/releases/latest/download/uv-{arch}-unknown-linux-musl.tar.gz"
    if system == "Darwin":
        arch = "aarch64" if "arm" in machine else "x86_64"
        return f"https://github.com/astral-sh/uv/releases/latest/download/uv-{arch}-apple-darwin.tar.gz"
    if system == "Windows":
        return "https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip"
    raise RuntimeError(f"Unsupported platform: {system}")


def _find_uv() -> Path | None:
    import shutil
    hit = shutil.which("uv")
    if hit:
        return Path(hit)
    suffix  = ".exe" if platform.system() == "Windows" else ""
    bundled = UV_BIN_DIR / f"uv{suffix}"
    return bundled if bundled.exists() else None


def _download_uv(progress_cb: callable | None = None) -> Path:
    url = _uv_download_url()
    UV_BIN_DIR.mkdir(parents=True, exist_ok=True)
    suffix  = ".exe" if platform.system() == "Windows" else ""
    uv_bin  = UV_BIN_DIR / f"uv{suffix}"

    if progress_cb:
        progress_cb("Downloading uv package manager…")

    with urllib.request.urlopen(url) as resp:
        data = resp.read()

    if url.endswith(".tar.gz"):
        with tarfile.open(fileobj=io.BytesIO(data)) as tar:
            for member in tar.getmembers():
                if member.name.endswith("/uv") or member.name == "uv":
                    uv_bin.write_bytes(tar.extractfile(member).read())
                    break
    else:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            for name in zf.namelist():
                if name.endswith("uv.exe"):
                    uv_bin.write_bytes(zf.read(name))
                    break

    uv_bin.chmod(uv_bin.stat().st_mode | stat.S_IEXEC)
    return uv_bin


def get_uv(progress_cb: callable | None = None) -> Path:
    uv = _find_uv()
    return uv if uv else _download_uv(progress_cb)


# ── Environment helpers ────────────────────────────────────────────────────────

def _venv_python() -> Path:
    return VENV_DIR / ("Scripts/python.exe" if platform.system() == "Windows" else "bin/python")

def _venv_pp_bin() -> Path:
    return VENV_DIR / ("Scripts/pixel-patrol.exe" if platform.system() == "Windows" else "bin/pixel-patrol")

def env_is_ready() -> bool:
    return VENV_DIR.exists() and _venv_pp_bin().exists()

def _run_streaming(cmd: list[str], line_cb: callable) -> None:
    """Run a command and feed each output line to line_cb. Raises on non-zero exit."""
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    for line in proc.stdout:
        line_cb(line.rstrip())
    proc.wait()
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)


def setup_environment(uv: Path, loader_pkgs: list[str], line_cb: callable | None = None) -> None:
    cb = line_cb or (lambda _: None)

    cb(">>> Creating Python environment…")
    _run_streaming([str(uv), "venv", "--python", "3.12", "--clear", str(VENV_DIR)], cb)

    pkgs = BASE_PACKAGES + loader_pkgs
    cb(f">>> Installing: {', '.join(pkgs)}")
    _run_streaming(
        [str(uv), "pip", "install", "--python", str(_venv_python()), *pkgs],
        cb,
    )


# ── Config ─────────────────────────────────────────────────────────────────────

def load_config() -> dict | None:
    if CONFIG_FILE.exists():
        try:
            return json.loads(CONFIG_FILE.read_text())
        except Exception:
            pass
    return None

def save_config(loader_pkgs: list[str]) -> None:
    APP_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(json.dumps({"loader_pkgs": loader_pkgs}, indent=2))


# ── Launch Pixel Patrol ────────────────────────────────────────────────────────

def launch_pixel_patrol() -> None:
    kwargs: dict = {}
    if platform.system() == "Windows":
        kwargs["creationflags"] = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    subprocess.Popen([str(_venv_pp_bin()), "launch"], **kwargs)


# ── HTML ───────────────────────────────────────────────────────────────────────

def _package_cards_html() -> str:
    cards = []
    for pkg in PACKAGES:
        checked = "checked" if pkg["default"] else ""

        # Processes row
        processes_row = ""
        if pkg["processes"]:
            processes_row = (
                f'<div class="pkg-row">'
                f'<span class="pkg-row-label">Analyses</span>'
                f'<span class="pkg-row-val">{pkg["processes"]}</span>'
                f'</div>'
            )

        # Widgets row
        widgets_row = ""
        if pkg["widgets"]:
            widget_pills = "".join(
                f'<span class="pill pill-widget">{w}</span>' for w in pkg["widgets"]
            )
            widgets_row = (
                f'<div class="pkg-row">'
                f'<span class="pkg-row-label">Widgets</span>'
                f'<div class="pill-wrap">{widget_pills}</div>'
                f'</div>'
            )

        # File types row
        ext_row = ""
        if pkg["extensions"]:
            ext_pills = "".join(
                f'<span class="pill pill-ext">{e}</span>' for e in pkg["extensions"]
            )
            ext_row = (
                f'<div class="pkg-row">'
                f'<span class="pkg-row-label">File types</span>'
                f'<div class="pill-wrap">{ext_pills}</div>'
                f'</div>'
            )

        cards.append(f"""
        <label class="loader-card" for="pkg-{pkg['id']}">
          <input type="checkbox" id="pkg-{pkg['id']}"
                 name="loader" value="{pkg['id']}" {checked}>
          <div class="card-body">
            <div class="card-title">
              {pkg['label']}
              <span class="card-subtitle">{pkg['use_case']}</span>
            </div>
            {processes_row}{widgets_row}{ext_row}
          </div>
        </label>""")
    return "\n".join(cards)


SETUP_HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Pixel Patrol – Setup</title>
{"<link rel='icon' type='image/png' href='" + _IMG_ICON + "'>" if _IMG_ICON else ""}
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css">
<style>
  body {{
    background: #f8f9fa;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
  }}

  /* ── Header (matches the Pixel Patrol launch page) ── */
  .pp-navbar {{
    background: #fff;
    border-bottom: 1px solid #dee2e6;
    padding: 24px 0;
  }}
  .pp-navbar .container-fluid {{
    max-width: 1400px;
    padding: 0 24px;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 16px;
  }}
  .pp-navbar img.logo {{
    height: 64px;
    width: auto;
    object-fit: contain;
  }}
  .pp-navbar h1 {{
    font-size: 1.75rem;
    font-weight: 700;
    color: #212529;
    margin: 0;
  }}

  /* ── Main content ── */
  .pp-main {{
    flex: 1;
    padding: 32px 0 24px;
  }}
  .pp-main .container-fluid {{
    max-width: 1400px;
    padding: 0 24px;
  }}

  /* ── Loader cards ── */
  .loader-card {{
    display: flex;
    align-items: flex-start;
    gap: 14px;
    padding: 14px 18px;
    border: 1.5px solid #dee2e6;
    border-radius: 8px;
    background: #fff;
    cursor: pointer;
    transition: border-color .15s, box-shadow .15s;
    margin-bottom: 10px;
    user-select: none;
  }}
  .loader-card:hover {{ border-color: #0d6efd; }}
  .loader-card:has(input:checked) {{
    border-color: #0d6efd;
    box-shadow: 0 0 0 3px rgba(13,110,253,.12);
  }}
  .loader-card input[type=checkbox] {{
    width: 17px; height: 17px;
    accent-color: #0d6efd;
    flex-shrink: 0;
    margin-top: 3px;
    cursor: pointer;
  }}
  .loader-card .card-body {{ display: flex; flex-direction: column; gap: 6px; flex: 1; padding: 0; }}
  .loader-card .card-title {{
    font-size: 15px; font-weight: 600; color: #212529;
    display: flex; align-items: baseline; gap: 10px; flex-wrap: wrap; margin: 0;
  }}
  .loader-card .card-subtitle {{ font-size: 12.5px; color: #6c757d; font-weight: 400; }}
  /* Package detail rows */
  .pkg-row {{
    display: flex;
    align-items: flex-start;
    gap: 10px;
    font-size: 12.5px;
  }}
  .pkg-row-label {{
    color: #6c757d;
    flex-shrink: 0;
    width: 72px;
    padding-top: 2px;
  }}
  .pkg-row-val {{ color: #495057; }}
  .pill-wrap {{ display: flex; flex-wrap: wrap; gap: 5px; }}
  .pill {{
    font-size: 11.5px;
    padding: 2px 8px;
    border-radius: 4px;
  }}
  .pill-ext {{
    font-family: SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    background: #e9ecef;
    color: #495057;
  }}
  .pill-widget {{
    background: #e9ecef;
    color: #495057;
  }}
  .loader-card:has(input:checked) .pill-ext {{
    background: #cfe2ff;
    color: #084298;
  }}
  .loader-card:has(input:checked) .pill-widget {{
    background: #d1e7dd;
    color: #0a3622;
  }}

  /* ── Action bar ── */
  .pp-actions {{
    display: flex;
    align-items: center;
    justify-content: flex-end;
    gap: 8px;
    margin-bottom: 24px;
  }}

  /* ── Progress overlay ── */
  #overlay {{
    display: none;
    position: fixed; inset: 0;
    background: rgba(0,0,0,.5);
    backdrop-filter: blur(4px);
    align-items: center;
    justify-content: center;
    z-index: 1050;
  }}
  #overlay.show {{ display: flex; }}
  .pp-modal {{
    background: #fff;
    border-radius: 12px;
    padding: 28px 32px 24px;
    width: 580px;
    max-width: 95vw;
    box-shadow: 0 20px 60px rgba(0,0,0,.25);
  }}
  .pp-modal .modal-title-row {{
    display: flex; align-items: center; gap: 12px; margin-bottom: 6px;
  }}
  .spinner-border-sm {{ width: 1.25rem; height: 1.25rem; flex-shrink: 0; }}
  #done-icon {{ display: none; color: #198754; font-size: 1.25rem; flex-shrink: 0; }}
  .pp-modal h5 {{ font-weight: 700; margin: 0; }}
  #progress-msg {{ font-size: 13px; color: #6c757d; margin-bottom: 12px; line-height: 1.5; }}

  /* Terminal log */
  #log {{
    background: #0d1117;
    color: #e6edf3;
    border-radius: 6px;
    padding: 12px 14px;
    font-family: SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    font-size: 12px;
    line-height: 1.6;
    height: 220px;
    overflow-y: auto;
    white-space: pre-wrap;
    word-break: break-all;
    margin-bottom: 12px;
  }}
  .log-status {{ color: #58a6ff; font-weight: 600; }}
  .log-err    {{ color: #f85149; }}

  #error-box {{
    display: none;
    background: #f8d7da;
    border: 1px solid #f5c2c7;
    border-radius: 6px;
    padding: 12px 14px;
    font-size: 12.5px;
    color: #842029;
    white-space: pre-wrap;
    word-break: break-word;
  }}

  /* ── Footer (matches the Pixel Patrol viewer/launch pages) ── */
  .pp-footer {{
    background-color: #212529;
    margin-top: auto;
  }}
  .pp-footer .container-fluid {{
    max-width: 1400px;
    padding: 0 24px;
  }}
  .pp-footer-inner {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 16px 0;
  }}
  .pp-footer-left {{ display: flex; align-items: center; }}
  .pp-footer-left img {{ height: 36px; margin-right: 14px; vertical-align: middle; }}
  .pp-footer-left .text {{ font-size: 12px; line-height: 1.4; }}
  .pp-footer-left .text span {{ color: #adb5bd; }}
  .pp-footer-left .text a {{ color: #dee2e6; text-decoration: none; }}
  .pp-footer-right {{ display: flex; align-items: center; gap: 16px; }}
  .pp-footer-right a {{ color: #adb5bd; text-decoration: none; font-size: 12px; }}
  .pp-footer-right a:hover {{ color: #dee2e6; }}
</style>
</head>
<body>

<!-- Navbar / header -->
<nav class="pp-navbar">
  <div class="container-fluid">
    {"<img class='logo' src='" + _IMG_LOGO + "' alt='Pixel Patrol'>" if _IMG_LOGO else ""}
    <h1>Pixel Patrol Setup</h1>
  </div>
</nav>

<!-- Main -->
<div class="pp-main">
  <div class="container-fluid">
    {f'''<p class="text-muted mb-3" style="font-size:13px;">
      Select additional packages to install. You can change this later by
      deleting <code>~/.pixel-patrol/</code> and relaunching.
    </p>
    {_package_cards_html()}''' if PACKAGES else '''<p class="text-muted mb-3" style="font-size:13px;">
      Pixel Patrol is ready to install.
    </p>'''}

    <!-- Action bar -->
    <div class="pp-actions">
      <button class="btn btn-outline-secondary btn-sm" onclick="window.close()">Cancel</button>
      <button class="btn btn-primary btn-sm" id="install-btn" onclick="startInstall()">
        <i class="bi bi-download me-1"></i>Install &amp; Launch
      </button>
    </div>
  </div>
</div>

<!-- Progress overlay -->
<div id="overlay">
  <div class="pp-modal">
    <div class="modal-title-row">
      <div class="spinner-border spinner-border-sm text-primary" id="spinner" role="status"></div>
      <i class="bi bi-check-circle-fill" id="done-icon"></i>
      <h5 id="modal-title">Setting up Pixel Patrol</h5>
    </div>
    <p id="progress-msg">Preparing…</p>
    <div id="log"></div>
    <div id="error-box"></div>
  </div>
</div>

<!-- Footer -->
<footer class="pp-footer">
  <div class="container-fluid">
    <div class="pp-footer-inner">
      <div class="pp-footer-left">
        {"<a href='https://helmholtz-imaging.de' target='_blank'><img src='" + _IMG_HELMHOLTZ + "' alt='Helmholtz Imaging'></a>" if _IMG_HELMHOLTZ else ""}
        <div class="text">
          <span>Pixel Patrol is developed by </span>
          <a href="https://helmholtz-imaging.de" target="_blank">Helmholtz Imaging</a>
          <span>. Launcher v{LAUNCHER_VERSION}</span>
        </div>
      </div>
      <div class="pp-footer-right">
        <a href="mailto:support@helmholtz-imaging.de">support@helmholtz-imaging.de</a>
        <a href="https://github.com/ida-mdc/pixel-patrol" target="_blank">
          <i class="bi bi-github me-1"></i>GitHub
        </a>
      </div>
    </div>
  </div>
</footer>

<script>
  const installBtn = document.getElementById('install-btn');
  const logEl      = document.getElementById('log');

  function selectedLoaders() {{
    return [...document.querySelectorAll('input[name=loader]:checked')].map(el => el.value);
  }}

  function appendLog(text, cls) {{
    const line = document.createElement('div');
    if (cls) line.className = cls;
    line.textContent = text;
    logEl.appendChild(line);
    logEl.scrollTop = logEl.scrollHeight;
  }}

  function startInstall() {{
    installBtn.disabled = true;
    document.getElementById('overlay').classList.add('show');

    const es = new EventSource('/install?loaders=' + encodeURIComponent(selectedLoaders().join(',')));

    es.addEventListener('status', e => {{
      document.getElementById('progress-msg').textContent = e.data;
      appendLog(e.data, 'log-status');
    }});

    es.addEventListener('log', e => {{ appendLog(e.data); }});

    es.addEventListener('done', () => {{
      es.close();
      document.getElementById('spinner').style.display = 'none';
      document.getElementById('done-icon').style.display = 'inline';
      document.getElementById('modal-title').textContent = 'All done!';
      document.getElementById('progress-msg').textContent =
        'Pixel Patrol is launching in your browser. You can close this tab.';
    }});

    es.addEventListener('error_msg', e => {{
      es.close();
      document.getElementById('spinner').style.display = 'none';
      document.getElementById('modal-title').textContent = 'Setup failed';
      document.getElementById('progress-msg').textContent = 'An error occurred:';
      document.getElementById('error-box').textContent = e.data;
      document.getElementById('error-box').style.display = 'block';
      appendLog(e.data, 'log-err');
    }});
  }}
</script>
</body>
</html>"""


# ── Flask app ──────────────────────────────────────────────────────────────────

def create_flask_app() -> Flask:
    app = Flask(__name__)
    app.logger.disabled = True

    # Suppress Flask startup banner
    import logging
    log = logging.getLogger("werkzeug")
    log.setLevel(logging.ERROR)

    _shutdown_event = threading.Event()

    @app.get("/")
    def index():
        return SETUP_HTML

    @app.get("/install")
    def install():
        # comma-separated list of selected package IDs
        selected_ids = set(request.args.get("loaders", "").split(","))
        seen: set[str] = set()
        loader_pkgs: list[str] = []
        for pkg in PACKAGES:
            if pkg["id"] in selected_ids and pkg["package"]:
                p = pkg["package"]
                if p not in seen:
                    seen.add(p)
                    loader_pkgs.append(p)

        def generate():
            import queue as _queue

            q: _queue.Queue = _queue.Queue()
            _DONE = object()

            def _worker() -> None:
                try:
                    def cb(line: str) -> None:
                        q.put(("log", line))

                    def dl_cb(msg: str) -> None:
                        q.put(("status", msg))

                    uv = get_uv(progress_cb=dl_cb)
                    setup_environment(uv, loader_pkgs, line_cb=cb)
                    save_config(loader_pkgs)
                    q.put(("status", f"Installed to {APP_DIR}"))
                    q.put(("status", "Launching Pixel Patrol…"))
                    launch_pixel_patrol()
                    q.put(("done", ""))
                except Exception as exc:
                    q.put(("error_msg", str(exc)))
                finally:
                    q.put(_DONE)

            threading.Thread(target=_worker, daemon=True).start()

            # Stream events until worker finishes — no timeout
            while True:
                item = q.get()
                if item is _DONE:
                    break
                event, data = item
                for line in data.splitlines() or [""]:
                    yield f"event: {event}\ndata: {line}\n"
                yield "\n"

            threading.Timer(2.0, _shutdown_event.set).start()

        return Response(
            stream_with_context(generate()),
            content_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    return app, _shutdown_event


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    APP_DIR.mkdir(parents=True, exist_ok=True)

    if env_is_ready() and load_config() is not None:
        # Already set up — just launch.
        print("Pixel Patrol: launching…")
        launch_pixel_patrol()
        return

    # First run: serve the setup wizard.
    import socket

    # Find a free port
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    url = f"http://127.0.0.1:{port}/"
    app, shutdown_event = create_flask_app()

    # Open the browser shortly after the server starts
    threading.Timer(0.8, lambda: webbrowser.open(url)).start()

    print(f"Pixel Patrol setup running at {url}")
    print("(Opening in your browser…)")

    server_thread = threading.Thread(
        target=lambda: app.run(host="127.0.0.1", port=port, threaded=True, use_reloader=False),
        daemon=True,
    )
    server_thread.start()

    # Wait until install is done (or Ctrl-C)
    shutdown_event.wait()


if __name__ == "__main__":
    main()
