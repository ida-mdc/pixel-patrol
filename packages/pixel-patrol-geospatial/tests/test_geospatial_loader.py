import rasterio
import tempfile
import pytest
import numpy as np
from pixel_patrol_geospatial.geospatial_loader import (
    GeoImageLoader,
    _get_datatype,
    _get_rectangle_polygon_shape_from_bounds,
)


class TestGeoImageLoader:
    def test_load_example_image(self):
        Z = np.full((2, 2), fill_value=1, dtype="uint8")
        transform = rasterio.transform.from_bounds(west=0, south=0, east=1, north=1, width=Z.shape[0], height=Z.shape[1])
        with tempfile.NamedTemporaryFile(delete_on_close=False) as fp:
            fp.close()  # on Windows a file can't be opened twice, so rasterio cannot open it for writing

            with rasterio.open(
                fp.name,
                'w',
                driver='GTiff',
                height=Z.shape[0],
                width=Z.shape[1],
                count=1,
                dtype=Z.dtype,
                crs='+proj=latlong',
                transform=transform,
            ) as ds:
                ds.write(Z, 1)

            record = GeoImageLoader().load(fp.name)
            assert record
            for out_field in GeoImageLoader.OUTPUT_SCHEMA:
                assert out_field in record.meta
            expected_fields = set(GeoImageLoader.OUTPUT_SCHEMA) | {"ndim"}
            assert set(record.meta.keys()) == expected_fields

@pytest.mark.parametrize(
    "test_input,expected",
    [
        (None, None),
        ([], None),
        ("uint8", "uint8"),
        (["float32", "float32"], "float32"),
        (["uint8", "float32"], None),
        (('uint16', 'uint16'), 'uint16')
    ],
)
def test_get_datatype(test_input, expected):
    actual = _get_datatype(dtypes=test_input)
    assert actual == expected

def test_get_rectangle_polygon_shape_from_bounds_dict_correct():
    import rasterio
    left, right, bottom, top = 0, 1, 2, 3
    bounds = rasterio.coords.BoundingBox(left=left, right=right, bottom=bottom, top=top)
    img_crs = rasterio.CRS.from_epsg(4326)
    polygon_dict = _get_rectangle_polygon_shape_from_bounds(bounds, img_crs)

    assert polygon_dict
    assert len(polygon_dict) == 8
    expected_dict = {
        "bbox_point1_lon": left, "bbox_point1_lat": bottom,
        "bbox_point2_lon": right, "bbox_point2_lat": bottom,
        "bbox_point3_lon": right, "bbox_point3_lat": top,
        "bbox_point4_lon": left, "bbox_point4_lat": top,
    }
    assert polygon_dict == expected_dict

def test_get_rectangle_polygon_shape_from_bounds_dict_converted():
    import rasterio
    left, right, bottom, top = 19989707.63115344, 20109321.668429803, 238632.7592445848, 355454.8524736772
    bounds = rasterio.coords.BoundingBox(left=left, right=right, bottom=bottom, top=top)
    img_crs = rasterio.CRS.from_epsg(3876)
    polygon_dict = _get_rectangle_polygon_shape_from_bounds(bounds, img_crs)

    assert polygon_dict
    assert len(polygon_dict) == 8

    assert polygon_dict["bbox_point1_lon"] == pytest.approx(0)
    assert polygon_dict["bbox_point1_lat"] == pytest.approx(2)

