from pixel_patrol_loader_bio.plugin_registry import register_loader_plugins
from pixel_patrol_base.plugin_registry import discover_loader


def test_all_three_loaders_registered():
    loaders = register_loader_plugins()
    names = [cls.NAME for cls in loaders]
    assert "bioio" in names
    assert "tifffile" in names
    assert "zarr" in names


def test_discover_loader_finds_all_three():
    assert discover_loader("bioio").NAME == "bioio"
    assert discover_loader("tifffile").NAME == "tifffile"
    assert discover_loader("zarr").NAME == "zarr"
