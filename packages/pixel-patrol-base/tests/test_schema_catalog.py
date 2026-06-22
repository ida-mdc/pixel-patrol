"""Tests for the schema/plugin catalog and parquet field-metadata embedding."""

import numpy as np
import polars as pl
import pyarrow.parquet as pq
import pytest

from pixel_patrol_base.core import schema_catalog
from pixel_patrol_base.core.project_metadata import ProjectMetadata
from pixel_patrol_base.io.parquet_io import save_parquet, with_field_descriptions

# Catalog facts that require a sibling loader package to be installed (the shared
# raster-image schema, loader->processor edges, image-column creators) are asserted
# in the umbrella package's test_schema_catalog_integration.py, which installs every
# plugin. The base package ships no loaders, so this suite only covers the slice the
# catalog can produce on its own (file/agg columns and base processor metrics).


# --- build_catalog ----------------------------------------------------------

def test_build_catalog_has_all_sections():
    cat = schema_catalog.build_catalog(include_widgets=False)
    assert set(cat) >= {"columns", "loaders", "processors", "widgets"}
    assert isinstance(cat["columns"], list)


def test_build_catalog_includes_basic_processor_with_descriptions():
    cat = schema_catalog.build_catalog(include_widgets=False)
    basic = next(p for p in cat["processors"] if p["name"] == "raster-basic")
    assert basic["description"]
    cols = {c["name"]: c for c in basic["columns"]}
    assert "min_intensity" in cols
    assert cols["min_intensity"]["dtype"] == "float32"
    assert cols["min_intensity"]["description"]


def test_overall_columns_include_base_columns():
    cat = schema_catalog.build_catalog(include_widgets=False)
    names = {c["name"] for c in cat["columns"]}
    assert {"path", "obs_level", "size_bytes"} <= names
    by_name = {c["name"]: c for c in cat["columns"]}
    assert by_name["path"]["description"]


# --- dtype rendering --------------------------------------------------------

@pytest.mark.parametrize("type_spec, expected", [
    (np.float32, "float32"),
    (np.uint64, "uint64"),
    (np.ndarray, "array"),
    (str, "string"),
    (int, "int"),
    (bytes, "bytes"),
    ((np.float32, 256), "float32[256]"),
])
def test_dtype_str(type_spec, expected):
    assert schema_catalog._dtype_str(type_spec) == expected


# --- column_descriptions / parquet embedding --------------------------------

def test_column_descriptions_cover_plugin_and_base():
    desc = schema_catalog.column_descriptions()
    assert desc.get("min_intensity")
    assert desc.get("size_bytes")
    assert desc.get("dtype")  # shared image-meta column


def test_with_field_descriptions_annotates_known_columns():
    schema = pl.DataFrame(
        {"min_intensity": [1.0], "Z_size": [3], "dim_z": [0], "unknown_xyz": [1]}
    ).to_arrow().schema
    annotated = with_field_descriptions(schema)
    desc = {f.name: (f.metadata or {}).get(b"description") for f in annotated}
    assert desc["min_intensity"]                 # exact plugin column
    assert desc["Z_size"]                        # pattern column (*_size)
    assert desc["dim_z"]                          # pattern column (dim_*)
    assert desc["unknown_xyz"] is None            # no description known


def test_saved_parquet_embeds_field_descriptions(tmp_path):
    df = pl.DataFrame({"min_intensity": [1.0, 2.0], "size_bytes": [10, 20]})
    dest = tmp_path / "report.parquet"
    save_parquet(df, dest, ProjectMetadata(project_name="t"))

    schema = pq.read_schema(dest)
    descs = {f.name: (f.metadata or {}).get(b"description") for f in schema}
    assert descs["min_intensity"] is not None
    assert descs["size_bytes"] is not None
    # footer provenance metadata must still be present alongside field metadata
    assert b"pp_project_name" in (pq.read_metadata(dest).metadata or {})


# --- widget connections -----------------------------------------------------

def test_connections_resolve_widget_inputs():
    cat = schema_catalog.build_catalog()  # needs Node for widgets; skip if unavailable
    if not cat["widgets"]:
        import pytest
        pytest.skip("widget metadata unavailable (Node not found)")
    edges = {(e["source"], e["target"]) for e in cat["connections"]}
    assert ("processor:raster-histogram", "widget:histogram") in edges
    assert ("processor:thumbnail", "widget:mosaic") in edges


# --- producer hierarchy + column view ---------------------------------------

def test_producers_tree_has_base_and_processors():
    cat = schema_catalog.build_catalog(include_widgets=False)
    groups = {g["id"]: g for g in cat["producers"]}
    assert {"base", "processors"} <= set(groups)
    base_children = {c["id"] for c in groups["base"]["children"]}
    assert {"file", "image", "agg"} <= base_children
    proc_children = {c["id"] for c in groups["processors"]["children"]}
    assert "processor:raster-basic" in proc_children


def test_columns_tagged_with_producer_and_category():
    cat = schema_catalog.build_catalog(include_widgets=False)
    by_name = {c["name"]: c for c in cat["columns"]}
    assert by_name["obs_level"]["producer"] == "agg"
    assert by_name["min_intensity"]["producer"] == "processor:raster-basic"
    assert by_name["min_intensity"]["category"] == "metric"
    assert by_name["size_bytes"]["producer"] == "file"
    # (image-category columns only appear when a loader emits them; asserted in the umbrella suite)
    # every column carries a producer + category, and every producer is a real node
    valid_producers = {"file", "image", "agg"} | {"processor:" + p["name"] for p in cat["processors"]}
    for c in cat["columns"]:
        assert c["producer"] in valid_producers, c
        assert c["category"] in {"file", "image", "agg", "metric"}


def test_aggregation_pattern_column_present_with_regex():
    # The per-dimension aggregation family comes from base processing, so it is listed
    # even with no loaders. The raster-image families (<axis>_size, pixel_size_<axis>)
    # depend on a loader and are checked in the umbrella suite.
    cat = schema_catalog.build_catalog(include_widgets=False)
    patterns = {c["name"]: c for c in cat["columns"] if "regex" in c}
    assert "dim_<axis>" in patterns


def test_metric_columns_carry_package_and_creators():
    cat = schema_catalog.build_catalog(include_widgets=False)
    by_name = {c["name"]: c for c in cat["columns"]}
    # metric columns name their processor + package (base ships the raster-basic processor).
    # Image-column creators span sibling loader packages and are checked in the umbrella suite.
    assert by_name["min_intensity"]["creators"] == ["raster-basic"]
    assert by_name["min_intensity"]["packages"] == ["pixel-patrol-base"]
