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
import platform
import stat
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


def test_base_packages_always_contains_base():
    assert "pixel-patrol-base" in launcher.BASE_PACKAGES


def test_image_package_is_default():
    img = next(p for p in launcher.PACKAGES if p["id"] == "image")
    assert img["default"] is True
    assert img["package"] == "pixel-patrol-image"


def test_bio_package_extensions():
    bio = next(p for p in launcher.PACKAGES if p["id"] == "bio")
    for ext in (".tif", ".tiff", ".zarr", ".ome.zarr", ".czi", ".nd2"):
        assert ext in bio["extensions"], f"{ext} missing from bio package extensions"


def test_aqqua_package_extensions():
    aqqua = next(p for p in launcher.PACKAGES if p["id"] == "aqqua")
    assert ".lmdb" in aqqua["extensions"]
    assert ".mdb" in aqqua["extensions"]


def test_image_package_has_widgets():
    img = next(p for p in launcher.PACKAGES if p["id"] == "image")
    assert len(img["widgets"]) > 0


def test_image_package_has_processes():
    img = next(p for p in launcher.PACKAGES if p["id"] == "image")
    assert img["processes"] is not None and len(img["processes"]) > 0


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


def test_package_cards_html_shows_extensions():
    html = launcher._package_cards_html()
    assert ".tif" in html
    assert ".zarr" in html
    assert ".lmdb" in html


def test_package_cards_html_shows_widgets():
    html = launcher._package_cards_html()
    img = next(p for p in launcher.PACKAGES if p["id"] == "image")
    for widget in img["widgets"]:
        assert widget in html


def test_package_cards_html_shows_processes():
    html = launcher._package_cards_html()
    img = next(p for p in launcher.PACKAGES if p["id"] == "image")
    # At least one word from the processes description should appear
    first_word = img["processes"].split(",")[0].strip().split()[0]
    assert first_word in html


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


# ── Config ─────────────────────────────────────────────────────────────────────

def test_save_and_load_config(tmp_path, monkeypatch):
    monkeypatch.setattr(launcher, "APP_DIR", tmp_path)
    monkeypatch.setattr(launcher, "CONFIG_FILE", tmp_path / "config.json")

    launcher.save_config(["pixel-patrol-loader-bio"])
    result = launcher.load_config()
    assert result == {"loader_pkgs": ["pixel-patrol-loader-bio"]}


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
    resp = flask_client.get("/install?loaders=image")
    assert resp.status_code == 200
    assert "text/event-stream" in resp.content_type


def test_install_streams_done_event(flask_client, tmp_path, monkeypatch):
    _patch_install_deps(monkeypatch, tmp_path)
    resp = flask_client.get("/install?loaders=image")
    body = resp.data.decode()
    assert "event: done" in body


def test_install_streams_status_events(flask_client, tmp_path, monkeypatch):
    _patch_install_deps(monkeypatch, tmp_path)
    resp = flask_client.get("/install?loaders=image")
    body = resp.data.decode()
    assert "event: status" in body


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


def test_install_deduplicates_packages(flask_client, tmp_path, monkeypatch):
    """Sending duplicate IDs must not install the same package twice."""
    installed: list[str] = []

    def fake_setup(uv, pkgs, line_cb=None):
        installed.extend(pkgs)

    _patch_install_deps(monkeypatch, tmp_path, setup_fn=fake_setup)
    # bio and a repeated bio — pixel-patrol-loader-bio must appear at most once
    flask_client.get("/install?loaders=bio,bio")
    assert installed.count("pixel-patrol-loader-bio") <= 1


def test_install_saves_config(flask_client, tmp_path, monkeypatch):
    _patch_install_deps(monkeypatch, tmp_path)
    flask_client.get("/install?loaders=bio")
    config = launcher.load_config()
    assert config is not None
    assert "pixel-patrol-loader-bio" in config["loader_pkgs"]


def test_install_error_streams_error_msg_event(flask_client, tmp_path, monkeypatch):
    monkeypatch.setattr(launcher, "VENV_DIR", tmp_path / "venv")
    monkeypatch.setattr(launcher, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(launcher, "APP_DIR", tmp_path)

    def boom(*a, **kw):
        raise RuntimeError("uv not found")

    monkeypatch.setattr(launcher, "get_uv", boom)
    monkeypatch.setattr(launcher, "launch_pixel_patrol", lambda: None)

    resp = flask_client.get("/install?loaders=bio")
    body = resp.data.decode()
    assert "event: error_msg" in body
    assert "uv not found" in body


# ── _pp_cmd ────────────────────────────────────────────────────────────────────

def test_pp_cmd_frozen_uses_venv_binary(tmp_path, monkeypatch):
    import platform
    if platform.system() == "Windows":
        fake_bin = tmp_path / "Scripts" / "pixel-patrol.exe"
    else:
        fake_bin = tmp_path / "bin" / "pixel-patrol"
    fake_bin.parent.mkdir(parents=True)
    fake_bin.touch()
    monkeypatch.setattr(launcher, "VENV_DIR", tmp_path)
    with patch("launcher.sys") as mock_sys:
        mock_sys.frozen = True
        cmd = launcher._pp_cmd()
    assert cmd == [str(fake_bin)]


def test_pp_cmd_dev_mode_uses_current_interpreter():
    with patch("launcher.sys") as mock_sys:
        mock_sys.frozen = False
        mock_sys.executable = "/fake/python"
        cmd = launcher._pp_cmd()
    assert cmd[0] == "/fake/python"
    assert "-m" in cmd
    assert "pixel_patrol_base.cli" in cmd


def test_pp_cmd_dev_mode_does_not_reference_venv():
    """In dev mode the venv binary must NOT appear in the command."""
    with patch("launcher.sys") as mock_sys:
        mock_sys.frozen = False
        mock_sys.executable = "/fake/python"
        cmd = launcher._pp_cmd()
    assert not any(".pixel-patrol" in part for part in cmd)


# ── _mimes_for_loader_pkgs ─────────────────────────────────────────────────────

def test_mimes_for_bio_loader():
    mimes = launcher._mimes_for_loader_pkgs(["pixel-patrol-loader-bio"])
    assert "image/tiff" in mimes
    assert "image/png" in mimes
    assert "image/jpeg" in mimes
    assert "image/x-zeiss-czi" in mimes
    assert "image/x-nikon-nd2" in mimes


def test_mimes_for_empty_loader_pkgs():
    assert launcher._mimes_for_loader_pkgs([]) == []


def test_mimes_for_unknown_package():
    assert launcher._mimes_for_loader_pkgs(["pixel-patrol-nonexistent"]) == []


def test_mimes_for_aqqua_loader():
    mimes = launcher._mimes_for_loader_pkgs(["pixel-patrol-aqqua"])
    assert "application/x-lmdb" in mimes
    assert "application/x-mdb" in mimes


def test_mimes_are_sorted():
    mimes = launcher._mimes_for_loader_pkgs(["pixel-patrol-loader-bio"])
    assert mimes == sorted(mimes)


# ── install_desktop_entry ──────────────────────────────────────────────────────

@pytest.fixture()
def desktop_env(tmp_path, monkeypatch):
    """Redirect all paths and stub subprocess for desktop-entry tests."""
    monkeypatch.setattr(launcher, "APP_DIR", tmp_path / "app")
    monkeypatch.setattr(launcher, "LAUNCHER_DIR", tmp_path / "app" / "launcher")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(launcher.subprocess, "run", lambda *a, **kw: None)
    return tmp_path


def _desktop_file(tmp_path: Path) -> Path:
    return tmp_path / ".local" / "share" / "applications" / "pixel-patrol.desktop"


@pytest.mark.skipif(platform.system() != "Linux", reason="Linux only")
def test_install_desktop_entry_creates_file(desktop_env):
    launcher.install_desktop_entry(["pixel-patrol-loader-bio"])
    assert _desktop_file(desktop_env).exists()


@pytest.mark.skipif(platform.system() != "Linux", reason="Linux only")
def test_install_desktop_entry_exec_line(desktop_env):
    launcher.install_desktop_entry(["pixel-patrol-loader-bio"])
    content = _desktop_file(desktop_env).read_text()
    assert "Exec=" in content
    assert "%f" in content


@pytest.mark.skipif(platform.system() != "Linux", reason="Linux only")
def test_install_desktop_entry_mime_types(desktop_env):
    launcher.install_desktop_entry(["pixel-patrol-loader-bio"])
    content = _desktop_file(desktop_env).read_text()
    assert "MimeType=" in content
    assert "image/tiff" in content
    assert "image/png" in content


@pytest.mark.skipif(platform.system() != "Linux", reason="Linux only")
def test_install_desktop_entry_empty_loader_pkgs(desktop_env):
    """No loader packages → desktop file still created, MimeType is empty."""
    launcher.install_desktop_entry([])
    content = _desktop_file(desktop_env).read_text()
    assert "MimeType=\n" in content


@pytest.mark.skipif(platform.system() != "Linux", reason="Linux only")
def test_install_desktop_entry_execute_bit(desktop_env):
    launcher.install_desktop_entry(["pixel-patrol-loader-bio"])
    f = _desktop_file(desktop_env)
    assert f.stat().st_mode & stat.S_IEXEC, ".desktop file must have execute bit set"


@pytest.mark.skipif(platform.system() != "Linux", reason="Linux only")
def test_install_desktop_entry_required_fields(desktop_env):
    launcher.install_desktop_entry(["pixel-patrol-loader-bio"])
    content = _desktop_file(desktop_env).read_text()
    for field in ("Type=Application", "Name=Pixel Patrol", "Categories="):
        assert field in content, f"Missing field: {field!r}"


@pytest.mark.skipif(platform.system() != "Linux", reason="Linux only")
def test_install_desktop_entry_idempotent(desktop_env):
    """Calling twice must not raise and must overwrite with the same content."""
    launcher.install_desktop_entry(["pixel-patrol-loader-bio"])
    content1 = _desktop_file(desktop_env).read_text()
    launcher.install_desktop_entry(["pixel-patrol-loader-bio"])
    content2 = _desktop_file(desktop_env).read_text()
    assert content1 == content2


def test_install_desktop_entry_noop_on_non_linux(desktop_env):
    """On non-Linux platforms the function must return without creating any file."""
    with patch("launcher.platform.system", return_value="Darwin"):
        launcher.install_desktop_entry(["pixel-patrol-loader-bio"])
    assert not _desktop_file(desktop_env).exists()


@pytest.mark.skipif(platform.system() != "Linux", reason="Linux only")
def test_install_desktop_entry_survives_write_error(desktop_env, monkeypatch):
    """Errors must be caught and not propagate."""
    # Make mkdir raise to trigger the exception path
    monkeypatch.setattr(Path, "mkdir", lambda *a, **kw: (_ for _ in ()).throw(PermissionError("no")))
    # Should not raise
    launcher.install_desktop_entry(["pixel-patrol-loader-bio"])


@pytest.mark.skipif(platform.system() != "Linux", reason="Linux only")
def test_install_route_installs_desktop_entry(flask_client, tmp_path, monkeypatch):
    """/install SSE stream must call install_desktop_entry on Linux."""
    called_with: list = []

    def fake_install(pkgs, progress_cb=None):
        called_with.append(pkgs)

    _patch_install_deps(monkeypatch, tmp_path)
    monkeypatch.setattr(launcher, "install_desktop_entry", fake_install)
    flask_client.get("/install?loaders=bio")
    assert called_with, "install_desktop_entry was not called during /install"
    assert "pixel-patrol-loader-bio" in called_with[0]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _patch_install_deps(monkeypatch, tmp_path, setup_fn=None):
    """Redirect every home-dir path to tmp_path and stub heavy operations."""
    monkeypatch.setattr(launcher, "APP_DIR",      tmp_path)
    monkeypatch.setattr(launcher, "LAUNCHER_DIR", tmp_path / "launcher")
    monkeypatch.setattr(launcher, "VENV_DIR",     tmp_path / "venv")
    monkeypatch.setattr(launcher, "CONFIG_FILE",  tmp_path / "config.json")
    monkeypatch.setattr(launcher, "UV_BIN_DIR",   tmp_path / "uv-bin")
    monkeypatch.setattr(launcher, "get_uv", lambda *a, **kw: Path("/fake/uv"))
    monkeypatch.setattr(
        launcher, "setup_environment",
        setup_fn if setup_fn is not None else (lambda uv, pkgs, line_cb=None: None),
    )
    monkeypatch.setattr(launcher, "launch_pixel_patrol", lambda: None)
    # Stub desktop integration — dedicated tests cover it separately
    monkeypatch.setattr(launcher, "install_desktop_entry", lambda *a, **kw: None)


