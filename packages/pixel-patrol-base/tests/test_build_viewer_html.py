"""
Tests for the single-file static viewer builder (viewer_pages).

Covers the light (URL-referenced, default) vs offline (fully inlined) modes of
build_single_file_viewer_html and the small helpers they rely on.
"""
import json
from pathlib import Path

import pytest

from pixel_patrol_base import viewer_pages
from pixel_patrol_base.viewer_pages import (
    _DEFAULT_DUCKDB_WASM_VERSION,
    _duckdb_cdn_urls,
    _read_duckdb_wasm_version,
    build_single_file_viewer_html,
)


INDEX_HTML = """<!DOCTYPE html>
<html><head>
  <link rel="icon" href="./icon.png"/>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css"/>
  <script type="module" crossorigin src="./assets/index-ABC123.js"></script>
  <link rel="stylesheet" crossorigin href="./assets/index-DEF456.css">
</head><body>
  <img src="./logo.png"/>
  <a href="#anchor">jump</a>
  <a href="https://example.com/page">external</a>
</body></html>
"""


@pytest.fixture
def fake_dist(tmp_path: Path) -> Path:
    dist = tmp_path / "viewer_dist"
    assets = dist / "assets"
    assets.mkdir(parents=True)
    (dist / "index.html").write_text(INDEX_HTML, encoding="utf-8")
    (dist / "icon.png").write_bytes(b"icon")
    (dist / "logo.png").write_bytes(b"logo")
    # App bundle references the DuckDB worker/wasm by their hashed names at runtime.
    (assets / "index-ABC123.js").write_text(
        "console.log('app');\n"
        'var w="./assets/duckdb-browser-mvp.worker-HASH.js";'
        'var m="./assets/duckdb-mvp-HASH.wasm";',
        encoding="utf-8",
    )
    (assets / "index-DEF456.css").write_text("body{}", encoding="utf-8")
    (assets / "duckdb-browser-mvp.worker-HASH.js").write_text("//worker", encoding="utf-8")
    (assets / "duckdb-mvp-HASH.wasm").write_bytes(b"\x00wasm-bytes" * 100)
    (dist / "pp_build_meta.json").write_text(
        json.dumps({"duckdbWasmVersion": "9.9.9"}), encoding="utf-8"
    )
    return dist


@pytest.fixture
def patched(monkeypatch, fake_dist):
    monkeypatch.setattr(viewer_pages, "find_viewer_dist", lambda: fake_dist)
    monkeypatch.setattr(viewer_pages, "_discover_installed_extensions", lambda: [])
    return fake_dist


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

class TestReadDuckdbWasmVersion:
    def test_reads_from_meta(self, fake_dist):
        assert _read_duckdb_wasm_version(fake_dist) == "9.9.9"

    def test_falls_back_when_missing(self, tmp_path):
        assert _read_duckdb_wasm_version(tmp_path) == _DEFAULT_DUCKDB_WASM_VERSION

    def test_falls_back_on_malformed_meta(self, tmp_path):
        (tmp_path / "pp_build_meta.json").write_text("not json", encoding="utf-8")
        assert _read_duckdb_wasm_version(tmp_path) == _DEFAULT_DUCKDB_WASM_VERSION


class TestDuckdbCdnUrls:
    def test_pins_version_and_uses_mvp_files(self, fake_dist):
        worker, wasm = _duckdb_cdn_urls(fake_dist)
        assert "@duckdb/duckdb-wasm@9.9.9/" in worker
        assert worker.endswith("/duckdb-browser-mvp.worker.js")
        assert wasm.endswith("/duckdb-mvp.wasm")


# ---------------------------------------------------------------------------
# build_single_file_viewer_html
# ---------------------------------------------------------------------------

class TestLightBuild:
    """Default: inline the app bundle but load DuckDB WASM from a CDN."""

    def test_inlines_app_js(self, patched, tmp_path):
        out = build_single_file_viewer_html(tmp_path / "v.html")  # light is default
        html = out.read_text(encoding="utf-8")
        assert "console.log('app');" in html

    def test_injects_duckdb_cdn_globals(self, patched, tmp_path):
        out = build_single_file_viewer_html(tmp_path / "v.html")
        html = out.read_text(encoding="utf-8")
        assert "window.__PP_DUCKDB_WORKER_URL =" in html
        assert "@duckdb/duckdb-wasm@9.9.9/" in html

    def test_does_not_inline_duckdb_wasm(self, patched, tmp_path):
        out = build_single_file_viewer_html(tmp_path / "v.html")
        html = out.read_text(encoding="utf-8")
        # The wasm bytes must not be embedded; the runtime ref is left untouched.
        assert "data:application/wasm" not in html
        assert "duckdb-mvp-HASH.wasm" in html  # original ref still present, not a data URL
        assert "wasm-bytes" not in html

    def test_no_github_pages_refs(self, patched, tmp_path):
        out = build_single_file_viewer_html(tmp_path / "v.html")
        assert "ida-mdc.github.io" not in out.read_text(encoding="utf-8")

    def test_keeps_bootstrap_cdn_link(self, patched, tmp_path):
        out = build_single_file_viewer_html(tmp_path / "v.html")
        html = out.read_text(encoding="utf-8")
        assert "cdn.jsdelivr.net/npm/bootstrap" in html


class TestOfflineBuild:
    def test_inlines_local_js(self, patched, tmp_path):
        out = build_single_file_viewer_html(tmp_path / "v.html", offline=True)
        html = out.read_text(encoding="utf-8")
        assert "console.log('app');" in html

    def test_no_cdn_app_bundle_or_duckdb_globals(self, patched, tmp_path):
        out = build_single_file_viewer_html(tmp_path / "v.html", offline=True)
        html = out.read_text(encoding="utf-8")
        assert "ida-mdc.github.io" not in html
        assert "window.__PP_DUCKDB_WORKER_URL =" not in html

    def test_both_modes_inject_extension_urls(self, patched, tmp_path):
        light = build_single_file_viewer_html(tmp_path / "l.html").read_text("utf-8")
        offline = build_single_file_viewer_html(tmp_path / "o.html", offline=True).read_text("utf-8")
        assert "window.__PP_EXTENSION_URLS =" in light
        assert "window.__PP_EXTENSION_URLS =" in offline
