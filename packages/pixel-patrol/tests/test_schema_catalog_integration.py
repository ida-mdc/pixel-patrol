"""Cross-package assertions about the schema/plugin catalog.

Lives in the umbrella package because the catalog wiring being checked here -
the shared raster-image schema, loader->processor connections and the per-column
creator credits - only materialises when the loader packages are installed. The
base package ships no loaders, so its own suite can only assert the loader-free
slice of these facts; the full multi-package picture is verified here.
"""

import pytest

from pixel_patrol_base.core import schema_catalog


def _catalog():
    return schema_catalog.build_catalog(include_widgets=False)


def _raster_loaders(cat):
    return [l for l in cat["loaders"] if l["schema"] == "raster-image"]


def test_raster_loaders_are_installed():
    assert _raster_loaders(_catalog()), "expected raster-image loaders in the umbrella environment"


def test_raster_loader_schema_is_shared_across_packages():
    cat = _catalog()
    schemas = {s["id"]: s for s in cat["loader_schemas"]}
    assert "raster-image" in schemas
    raster = schemas["raster-image"]
    names = {c["name"] for c in raster["columns"]}
    assert {"dim_order", "shape", "dtype", "channel_names"} <= names
    # the schema lists exactly the loaders that declare it
    common_loaders = {l["name"] for l in _raster_loaders(cat)}
    assert set(raster["loaders"]) == common_loaders
    # the bio loaders all share one schema, proving it spans packages
    assert {"bioio", "tifffile", "zarr"} <= common_loaders


def test_loader_entry_splits_common_and_extra_columns():
    cat = _catalog()
    zarr = next(l for l in cat["loaders"] if l["name"] == "zarr")
    assert zarr["schema"] == "raster-image"
    extra = {c["name"] for c in zarr["extra_columns"]}
    assert "zarr_attributes" in extra
    assert "dim_order" not in extra  # common columns are not repeated as extras


def test_connections_link_raster_loaders_to_processors():
    cat = _catalog()
    assert any(e["source"].startswith("loader:") and e["target"].startswith("processor:")
               for e in cat["connections"])


def test_image_columns_appear_only_with_loaders():
    cat = _catalog()
    by_name = {c["name"]: c for c in cat["columns"]}
    # the shared raster-image columns are catalogued because loaders emit them
    assert by_name["dtype"]["producer"] == "image"
    assert by_name["dtype"]["category"] == "image"
    # and so are the per-axis families they declare
    patterns = {c["name"] for c in cat["columns"] if "regex" in c}
    assert {"size_<axis>", "pixel_size_<axis>"} <= patterns


def test_columns_credit_creators_across_packages():
    cat = _catalog()
    by_name = {c["name"]: c for c in cat["columns"]}
    # shared image columns are created by every raster loader, across packages
    dim_order = by_name["dim_order"]
    assert set(dim_order["creators"]) >= {"bioio", "tifffile", "zarr"}
    assert "pixel-patrol-loader-bio" in dim_order["packages"]
    # a loader-specific column names only its loader
    assert by_name["zarr_attributes"]["creators"] == ["zarr"]
    # metric columns name their processor + package
    assert by_name["min_intensity"]["creators"] == ["raster-basic"]
    assert by_name["min_intensity"]["packages"] == ["pixel-patrol-base"]
    # with loaders installed every catalogued column carries both fields
    for c in cat["columns"]:
        assert c["creators"] and c["packages"]


def test_metric_widgets_connect_only_to_their_metric_family():
    # The violin / across-dims widgets each plot one metric family, so they must link
    # only to the processor(s) that produce it - not to every metric like custom-plot.
    cat = schema_catalog.build_catalog()  # needs Node for widget inputs
    if not cat["widgets"]:
        pytest.skip("widget metadata unavailable (Node not found)")

    def sources(wid):
        return {e["source"] for e in cat["connections"] if e["target"] == f"widget:{wid}"}

    quality = {"processor:raster-quality", "processor:raster-compression"}
    for basic in ("violin-basic", "stats-across-dims-basic"):
        assert sources(basic) == {"processor:raster-basic"}
    for qual in ("violin-quality", "stats-across-dims-quality"):
        assert sources(qual) == quality
    # custom-plot still accepts any metric column, so it spans every metric processor
    assert sources("custom-plot") >= {"processor:raster-basic"} | quality
