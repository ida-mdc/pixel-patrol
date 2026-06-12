"""
Tests for the Pixel Patrol setup launcher.

Covers:
- PACKAGES catalogue integrity
- HTML card generation
- Config read / write
- env_is_ready detection
- Flask routes (/, /install SSE stream)
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from unittest.mock import patch

import pytest

import launcher


# ── PACKAGES catalogue ─────────────────────────────────────────────────────────

def test_packages_have_required_keys():
    required = {"id", "label", "package", "use_case", "processes", "widgets", "extensions", "default"}
    for pkg in launcher.PACKAGES:
        missing = required - pkg.keys()
        assert not missing, f"Package {pkg.get('id')!r} is missing keys: {missing}"


def test_package_ids_are_unique():
    ids = [p["id"] for p in launcher.PACKAGES]
    assert len(ids) == len(set(ids)), f"Duplicate package ids: {ids}"


def test_base_packages_always_contains_full_bundle():
    assert "pixel-patrol" in launcher.BASE_PACKAGES


# ── Version ────────────────────────────────────────────────────────────────────

def test_launcher_version_matches_pixel_patrol_version():
    import tomllib

    pyproject = Path(__file__).resolve().parents[3] / "packages" / "pixel-patrol" / "pyproject.toml"
    pp_version = tomllib.loads(pyproject.read_text())["project"]["version"]
    assert launcher.LAUNCHER_VERSION == pp_version


# ── HTML generation ────────────────────────────────────────────────────────────

def test_package_cards_html_has_checkbox_for_each_package():
    html = launcher._package_cards_html()
    for pkg in launcher.PACKAGES:
        assert f'id="pkg-{pkg["id"]}"' in html
        assert f'value="{pkg["id"]}"' in html
        assert 'type="checkbox"' in html


def test_package_cards_html_defaults_are_checked():
    html = launcher._package_cards_html()
    for pkg in launcher.PACKAGES:
        if pkg["default"]:
            # The checked attribute must appear on the default packages
            assert f'value="{pkg["id"]}" checked' in html


def test_package_cards_html_non_defaults_not_checked():
    html = launcher._package_cards_html()
    for pkg in launcher.PACKAGES:
        if not pkg["default"]:
            assert f'value="{pkg["id"]}" checked' not in html


def test_setup_html_loads_bootstrap():
    assert "bootstrap" in launcher.SETUP_HTML.lower()


def test_setup_html_loads_bootstrap_icons():
    assert "bootstrap-icons" in launcher.SETUP_HTML


def test_setup_html_contains_helmholtz():
    assert "Helmholtz Imaging" in launcher.SETUP_HTML
    assert "helmholtz-imaging.de" in launcher.SETUP_HTML


def test_setup_html_contains_sse_listener():
    assert "EventSource" in launcher.SETUP_HTML


def test_setup_html_contains_github_link():
    assert "github.com" in launcher.SETUP_HTML


# ── Environment setup ──────────────────────────────────────────────────────────

def test_setup_environment_clears_existing_venv(tmp_path, monkeypatch):
    """uv venv must be called with --clear so a broken/partial prior install
    (e.g. from an interrupted setup) doesn't break re-installation."""
    monkeypatch.setattr(launcher, "VENV_DIR", tmp_path / "venv")
    commands: list[list[str]] = []

    def fake_run_streaming(cmd, line_cb):
        commands.append(cmd)

    monkeypatch.setattr(launcher, "_run_streaming", fake_run_streaming)
    launcher.setup_environment(Path("/fake/uv"), [])
    venv_cmd = commands[0]
    assert "--clear" in venv_cmd


# ── Config ─────────────────────────────────────────────────────────────────────

def test_save_and_load_config(tmp_path, monkeypatch):
    monkeypatch.setattr(launcher, "APP_DIR", tmp_path)
    monkeypatch.setattr(launcher, "CONFIG_FILE", tmp_path / "config.json")

    launcher.save_config(["pixel-patrol-demo"])
    result = launcher.load_config()
    assert result == {"loader_pkgs": ["pixel-patrol-demo"]}


def test_save_config_empty_list(tmp_path, monkeypatch):
    monkeypatch.setattr(launcher, "APP_DIR", tmp_path)
    monkeypatch.setattr(launcher, "CONFIG_FILE", tmp_path / "config.json")

    launcher.save_config([])
    result = launcher.load_config()
    assert result == {"loader_pkgs": []}


def test_load_config_returns_none_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(launcher, "CONFIG_FILE", tmp_path / "nonexistent.json")
    assert launcher.load_config() is None


def test_load_config_returns_none_on_corrupt_json(tmp_path, monkeypatch):
    bad = tmp_path / "config.json"
    bad.write_text("not valid json {{{{")
    monkeypatch.setattr(launcher, "CONFIG_FILE", bad)
    assert launcher.load_config() is None


# ── env_is_ready ───────────────────────────────────────────────────────────────

def test_env_is_ready_false_when_venv_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(launcher, "VENV_DIR", tmp_path / "no_venv")
    assert launcher.env_is_ready() is False


def test_env_is_ready_false_when_binary_missing(tmp_path, monkeypatch):
    venv = tmp_path / "venv"
    venv.mkdir()
    monkeypatch.setattr(launcher, "VENV_DIR", venv)
    assert launcher.env_is_ready() is False


def test_env_is_ready_true_when_binary_exists(tmp_path, monkeypatch):
    import platform
    if platform.system() == "Windows":
        pp_bin = tmp_path / "venv" / "Scripts" / "pixel-patrol.exe"
    else:
        pp_bin = tmp_path / "venv" / "bin" / "pixel-patrol"
    pp_bin.parent.mkdir(parents=True)
    pp_bin.touch()
    monkeypatch.setattr(launcher, "VENV_DIR", tmp_path / "venv")
    assert launcher.env_is_ready() is True


# ── Flask routes ───────────────────────────────────────────────────────────────

@pytest.fixture()
def flask_client(monkeypatch):
    """Flask test client with shutdown timer suppressed."""
    monkeypatch.setattr(threading, "Timer", lambda *a, **kw: _NoOpTimer())
    app, _ = launcher.create_flask_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


class _NoOpTimer:
    def start(self):
        pass


def test_index_returns_200(flask_client):
    resp = flask_client.get("/")
    assert resp.status_code == 200


def test_index_content_type_is_html(flask_client):
    resp = flask_client.get("/")
    assert "text/html" in resp.content_type


def test_index_contains_all_package_labels(flask_client):
    html = flask_client.get("/").data.decode()
    for pkg in launcher.PACKAGES:
        assert pkg["label"] in html, f"Package label {pkg['label']!r} not found in setup page"


def test_index_contains_helmholtz_footer(flask_client):
    html = flask_client.get("/").data.decode()
    assert "Helmholtz Imaging" in html
    assert "helmholtz-imaging.de" in html


def test_index_contains_bootstrap(flask_client):
    html = flask_client.get("/").data.decode()
    assert "bootstrap" in html.lower()


def test_index_contains_install_button(flask_client):
    html = flask_client.get("/").data.decode()
    assert "startInstall" in html
    assert "install-btn" in html


def test_install_response_is_event_stream(flask_client, tmp_path, monkeypatch):
    _patch_install_deps(monkeypatch, tmp_path)
    resp = flask_client.get("/install?loaders=")
    assert resp.status_code == 200
    assert "text/event-stream" in resp.content_type


def test_install_streams_done_event(flask_client, tmp_path, monkeypatch):
    _patch_install_deps(monkeypatch, tmp_path)
    resp = flask_client.get("/install?loaders=")
    body = resp.data.decode()
    assert "event: done" in body


def test_install_streams_status_events(flask_client, tmp_path, monkeypatch):
    _patch_install_deps(monkeypatch, tmp_path)
    resp = flask_client.get("/install?loaders=")
    body = resp.data.decode()
    assert "event: status" in body


def test_install_streams_install_dir_before_launch(flask_client, tmp_path, monkeypatch):
    _patch_install_deps(monkeypatch, tmp_path)
    resp = flask_client.get("/install?loaders=")
    body = resp.data.decode()
    install_dir_idx = body.find(f"data: Installed to {tmp_path}")
    launch_idx = body.find("data: Launching Pixel Patrol")
    assert install_dir_idx != -1
    assert launch_idx != -1
    assert install_dir_idx < launch_idx


def test_install_no_loaders_installs_base_only(flask_client, tmp_path, monkeypatch):
    """Selecting no optional packages should still complete without error."""
    installed: list[list[str]] = []

    def fake_setup(uv, pkgs, line_cb=None):
        installed.append(list(pkgs))

    _patch_install_deps(monkeypatch, tmp_path, setup_fn=fake_setup)
    resp = flask_client.get("/install?loaders=")
    assert "event: done" in resp.data.decode()
    # Only base packages, no extras
    assert installed == [[]]


_FAKE_PACKAGE = {
    "id": "demo",
    "label": "Demo package",
    "package": "pixel-patrol-demo",
    "use_case": "demo",
    "processes": [],
    "widgets": [],
    "extensions": [],
    "default": False,
}


def test_install_deduplicates_packages(flask_client, tmp_path, monkeypatch):
    """Sending duplicate IDs must not install the same package twice."""
    installed: list[str] = []

    def fake_setup(uv, pkgs, line_cb=None):
        installed.extend(pkgs)

    monkeypatch.setattr(launcher, "PACKAGES", [_FAKE_PACKAGE])
    _patch_install_deps(monkeypatch, tmp_path, setup_fn=fake_setup)
    # demo and a repeated demo — pixel-patrol-demo must appear at most once
    flask_client.get("/install?loaders=demo,demo")
    assert installed.count("pixel-patrol-demo") <= 1


def test_install_saves_config(flask_client, tmp_path, monkeypatch):
    monkeypatch.setattr(launcher, "PACKAGES", [_FAKE_PACKAGE])
    _patch_install_deps(monkeypatch, tmp_path)
    flask_client.get("/install?loaders=demo")
    config = launcher.load_config()
    assert config is not None
    assert "pixel-patrol-demo" in config["loader_pkgs"]


def test_install_error_streams_error_msg_event(flask_client, tmp_path, monkeypatch):
    monkeypatch.setattr(launcher, "VENV_DIR", tmp_path / "venv")
    monkeypatch.setattr(launcher, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(launcher, "APP_DIR", tmp_path)

    def boom(*a, **kw):
        raise RuntimeError("uv not found")

    monkeypatch.setattr(launcher, "get_uv", boom)
    monkeypatch.setattr(launcher, "launch_pixel_patrol", lambda: None)

    resp = flask_client.get("/install?loaders=")
    body = resp.data.decode()
    assert "event: error_msg" in body
    assert "uv not found" in body


# ── Helpers ────────────────────────────────────────────────────────────────────

def _patch_install_deps(monkeypatch, tmp_path, setup_fn=None):
    """Redirect every home-dir path to tmp_path and stub heavy operations."""
    monkeypatch.setattr(launcher, "APP_DIR",     tmp_path)
    monkeypatch.setattr(launcher, "VENV_DIR",    tmp_path / "venv")
    monkeypatch.setattr(launcher, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(launcher, "UV_BIN_DIR",  tmp_path / "uv-bin")
    monkeypatch.setattr(launcher, "get_uv", lambda *a, **kw: Path("/fake/uv"))
    monkeypatch.setattr(
        launcher, "setup_environment",
        setup_fn if setup_fn is not None else (lambda uv, pkgs, line_cb=None: None),
    )
    monkeypatch.setattr(launcher, "launch_pixel_patrol", lambda: None)


