"""
Tests for the Python-side JS viewer extension discovery system.

Covers:
  - viewer_extensions.get_viewer_extension_dir  (entry-point callable)
  - viewer_server._discover_installed_extensions (entry-point loader)
  - viewer_server.find_viewer_dist               (dist directory lookup)
"""
import json
import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pixel_patrol_base.viewer_extensions import get_viewer_extension_dir
from pixel_patrol_base.viewer_server import _discover_installed_extensions, find_viewer_dist


# ---------------------------------------------------------------------------
# get_viewer_extension_dir
# ---------------------------------------------------------------------------

class TestGetViewerExtensionDir:
    def test_returns_path(self):
        result = get_viewer_extension_dir()
        assert isinstance(result, Path)

    def test_points_to_viewer_directory(self):
        result = get_viewer_extension_dir()
        assert result.name == "viewer"

    def test_directory_exists(self):
        result = get_viewer_extension_dir()
        assert result.is_dir()

    def test_contains_extension_json(self):
        result = get_viewer_extension_dir()
        assert (result / "extension.json").exists()

    def test_extension_json_lists_plugins(self):
        ext_dir = get_viewer_extension_dir()
        data = json.loads((ext_dir / "extension.json").read_text())
        assert "plugins" in data
        assert len(data["plugins"]) > 0

    def test_all_declared_plugins_exist(self):
        ext_dir = get_viewer_extension_dir()
        data = json.loads((ext_dir / "extension.json").read_text())
        for rel_path in data["plugins"]:
            # Paths are relative, starting with "./"
            plugin_file = ext_dir / rel_path.lstrip("./")
            assert plugin_file.exists(), f"Declared plugin missing: {rel_path}"


# ---------------------------------------------------------------------------
# _discover_installed_extensions
# ---------------------------------------------------------------------------

def _make_ep(name: str, fn):
    """Build a mock entry point whose .load() returns fn."""
    ep = MagicMock()
    ep.name = name
    ep.value = f"fake:{name}"
    ep.load.return_value = fn
    return ep


class TestDiscoverInstalledExtensions:
    def test_returns_list(self):
        result = _discover_installed_extensions()
        assert isinstance(result, list)

    def test_includes_base_extension(self):
        """The installed pixel-patrol-base package registers its own extension."""
        result = _discover_installed_extensions()
        expected = get_viewer_extension_dir()
        assert expected in result

    def test_skips_ep_whose_directory_has_no_extension_json(self, tmp_path):
        bad_dir = tmp_path / "no_json"
        bad_dir.mkdir()

        ep = _make_ep("bad_ext", lambda: bad_dir)

        with patch("importlib.metadata.entry_points", return_value=[ep]):
            with pytest.warns(UserWarning, match="not a directory containing extension.json"):
                result = _discover_installed_extensions()

        assert bad_dir not in result

    def test_skips_ep_that_raises_on_load(self):
        ep = MagicMock()
        ep.name = "broken_ext"
        ep.value = "broken:broken"
        ep.load.side_effect = RuntimeError("simulated load failure")

        with patch("importlib.metadata.entry_points", return_value=[ep]):
            with pytest.warns(UserWarning, match="failed to load"):
                result = _discover_installed_extensions()

        assert result == []

    def test_valid_custom_extension_is_included(self, tmp_path):
        ext_dir = tmp_path / "my_ext"
        ext_dir.mkdir()
        (ext_dir / "extension.json").write_text('{"plugins": ["./plugin_foo.js"]}')

        ep = _make_ep("my_ext", lambda: ext_dir)

        with patch("importlib.metadata.entry_points", return_value=[ep]):
            result = _discover_installed_extensions()

        assert ext_dir in result

    def test_results_are_sorted_by_ep_name(self, tmp_path):
        dirs = {}
        eps = []
        for name in ["zzz_ext", "aaa_ext", "mmm_ext"]:
            d = tmp_path / name
            d.mkdir()
            (d / "extension.json").write_text('{"plugins": []}')
            dirs[name] = d
            eps.append(_make_ep(name, lambda d=d: d))

        with patch("importlib.metadata.entry_points", return_value=eps):
            result = _discover_installed_extensions()

        assert result == [dirs["aaa_ext"], dirs["mmm_ext"], dirs["zzz_ext"]]


# ---------------------------------------------------------------------------
# find_viewer_dist
# ---------------------------------------------------------------------------

class TestFindViewerDist:
    def test_returns_path(self):
        result = find_viewer_dist()
        assert isinstance(result, Path)

    def test_dist_contains_index_html(self):
        result = find_viewer_dist()
        assert (result / "index.html").exists()

    def test_raises_when_neither_location_exists(self, monkeypatch, tmp_path):
        """FileNotFoundError when the viewer hasn't been built."""
        import importlib.resources as ir

        # Make the package-data lookup fail
        monkeypatch.setattr(ir, "files", lambda _: MagicMock(
            joinpath=lambda *a: MagicMock(
                joinpath=lambda *b: MagicMock(is_file=lambda: False)
            )
        ))
        # Point the source-tree fallback somewhere that doesn't exist
        import pixel_patrol_base.viewer_server as vs
        monkeypatch.setattr(vs, "__file__", str(tmp_path / "fake_module.py"))

        with pytest.raises(FileNotFoundError, match="Viewer not built"):
            find_viewer_dist()