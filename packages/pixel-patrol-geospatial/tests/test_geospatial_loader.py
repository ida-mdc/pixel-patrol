import json
import logging

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

from pixel_patrol_base.core.contracts import FileInfo
from pixel_patrol_base.core.record import Record, record_from

from pixel_patrol_geospatial.geospatial_loader import (
    GeospatialLoader,
    _get_footprint_geojson,
    _get_nodata_value_rasterio,
)

@pytest.fixture
def sample_tiff(tmp_path):
    """A minimal single-band GeoTIFF in EPSG:4326 with a known nodata value."""
    path = tmp_path / "sample.tif"
    data = np.array([[[1, 2], [0, 4]]], dtype=np.uint8)  # shape (1, 2, 2)
    transform = from_bounds(west=10.0, south=50.0, east=11.0, north=51.0, width=2, height=2)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=2,
        width=2,
        count=1,
        dtype=np.uint8,
        crs=rasterio.CRS.from_epsg(4326),
        transform=transform,
        nodata=255,
    ) as dst:
        dst.write(data)
    return path


@pytest.fixture
def sample_record():
    """A Record with a known array and nodata_value, usable by processor tests."""
    arr = np.array([[[1, 0], [0, 4]]], dtype=np.uint8)
    return record_from(arr, {"dim_order": "CYX", "nodata_value": 0}, kind="intensity")


def test_nodata_image_level():
    assert _get_nodata_value_rasterio(5, []) == 5


def test_nodata_band_level():
    assert _get_nodata_value_rasterio(None, [3, 3]) == 3


def test_nodata_none():
    assert _get_nodata_value_rasterio(None, []) is None


def test_nodata_nan_returns_none():
    # NaN nodata values are filtered out; result is None
    assert _get_nodata_value_rasterio(float("nan"), []) is None


def test_nodata_conflict_logs_warning_and_returns_none(caplog):
    with caplog.at_level(logging.WARNING):
        result = _get_nodata_value_rasterio(None, [1, 2])
    assert result is None
    assert any("Multiple no data values" in m for m in caplog.messages)


def test_footprint_is_valid_geojson():
    crs = rasterio.CRS.from_epsg(4326)
    bounds = rasterio.coords.BoundingBox(left=10.0, bottom=50.0, right=11.0, top=51.0)
    result = _get_footprint_geojson(bounds, crs)
    geojson = json.loads(result)
    assert geojson["type"] == "Polygon"
    assert "coordinates" in geojson


def test_footprint_epsg4326_roundtrip():
    crs = rasterio.CRS.from_epsg(4326)
    bounds = rasterio.coords.BoundingBox(left=10.0, bottom=50.0, right=11.0, top=51.0)
    result = _get_footprint_geojson(bounds, crs)
    ring = json.loads(result)["coordinates"][0]
    xs = [c[0] for c in ring]
    ys = [c[1] for c in ring]
    assert min(xs) == pytest.approx(10.0, abs=0.01)
    assert max(xs) == pytest.approx(11.0, abs=0.01)
    assert min(ys) == pytest.approx(50.0, abs=0.01)
    assert max(ys) == pytest.approx(51.0, abs=0.01)



def test_read_header_returns_file_info(sample_tiff):
    loader = GeospatialLoader()
    fi = loader.read_header(sample_tiff)
    assert isinstance(fi, FileInfo)
    assert fi.shape == (1, 2, 2)  # (bands, height, width) from fixture
    assert fi.dtype == "uint8"


def test_load_pixel_data_shape(sample_tiff):
    loader = GeospatialLoader()
    record = loader.load(sample_tiff)
    assert np.asarray(record.data).shape == (1, 2, 2)
    assert record.meta["nodata_value"] == 255
    assert isinstance(record, Record)
    assert record.kind == "intensity"

    meta = loader.load(sample_tiff).meta
    for field in ("crs_str", "crs_epsg", "latitude", "longitude", "footprint", "nodata_value"):
        assert field in meta

