"""
Tests for the static viewer builders in ``viewer_pages``.
"""
import json
from pathlib import Path

import pytest

import pixel_patrol_base.viewer_pages as viewer_pages


@pytest.fixture
def fake_dist(tmp_path: Path) -> Path:
    """A minimal viewer dist/ directory (just enough for the builder)."""
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<html><head></head><body></body></html>")
    return dist


@pytest.fixture
def auto_detect_extension(tmp_path: Path) -> Path:
    """An extension dir using auto_detect with a couple of plugin_*.js files."""
    ext = tmp_path / "base_ext"
    ext.mkdir()
    (ext / "extension.json").write_text('{"name": "Base Plugins", "auto_detect": true}')
    (ext / "plugin_summary.js").write_text("// summary")
    (ext / "plugin_histogram.js").write_text("// histogram")
    (ext / "helper.js").write_text("// not a plugin")
    return ext


def test_site_build_bakes_plugins_into_manifest(
    monkeypatch, tmp_path, fake_dist, auto_detect_extension
):
    """The copied extension.json must expose an explicit plugins list.

    Before the fix this manifest was copied verbatim (only auto_detect, no
    plugins), so the client loaded zero widgets.
    """
    monkeypatch.setattr(viewer_pages, "find_viewer_dist", lambda: fake_dist)
    monkeypatch.setattr(
        viewer_pages, "_discover_installed_extensions", lambda: [auto_detect_extension]
    )

    out_dir = viewer_pages.build_github_pages_site(tmp_path / "site")

    manifests = list((out_dir / "viewer" / "extensions").glob("*/extension.json"))
    assert len(manifests) == 1

    manifest = json.loads(manifests[0].read_text())
    # The client reads manifest.plugins directly; auto_detect is unusable statically.
    assert manifest.get("plugins") == ["./plugin_histogram.js", "./plugin_summary.js"]
    assert "auto_detect" not in manifest

    # Every referenced plugin file was actually copied alongside the manifest.
    ext_out = manifests[0].parent
    for rel in manifest["plugins"]:
        assert (ext_out / rel.lstrip("./")).is_file()
