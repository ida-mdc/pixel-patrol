import rasterio
import tempfile
import pytest
import numpy as np
from pixel_patrol_geospatial.geospatial_loader import GeoImageLoader, _get_datatype


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

