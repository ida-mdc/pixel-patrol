from pixel_patrol_loader_bio.plugin_registry import register_loader_plugins
from pixel_patrol_base.plugin_registry import discover_loader

LOADER_NAMES = ["bioio", "h5", "tifffile", "zarr"]


def test_all_loaders_registered():
    names = [cls.NAME for cls in register_loader_plugins()]
    for expected in LOADER_NAMES:
        assert expected in names


def test_discover_loader_finds_all():
    for expected in LOADER_NAMES:
        assert discover_loader(expected).NAME == expected
