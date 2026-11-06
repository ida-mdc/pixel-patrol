import math
import tracemalloc

import numpy as np
import pytest

from pixel_patrol_base.core.record import Record, record_from
from pixel_patrol_base.core.specs import is_record_matching_processor
from pixel_patrol_base.plugins.processors import rgb_colour_stats as colour_stats
from pixel_patrol_base.plugins.processors.rgb_colour_processor import RgbColourProcessor

NAMES = {"rgb_luma_std", "rgb_saturation_mean", "rgb_hue_coherence"}


def solid(rgb, shape=(32, 48)):
    return np.broadcast_to(np.array(rgb, dtype=np.uint8), (*shape, 3)).copy()


def sums_of(image, dim_order="YXS", ceiling=255.0):
    return colour_stats.chunk_sums(colour_stats.colour_planes(image, dim_order, "S"), ceiling)


def metrics_of(image, dim_order="YXS", ceiling=255.0):
    return colour_stats.combine([sums_of(image, dim_order, ceiling)])


def record_of(image, dim_order="YXS", capabilities=("rgb", "rgb:S")):
    return Record(data=image, dim_order=dim_order, dim_names=list(dim_order), kind="intensity", meta={}, capabilities=set(capabilities))


def reference(image, ceiling):
    """The same measures worked out directly in float64 on every pixel."""
    r, g, b = image.reshape(-1, 3).astype(np.float64).T
    luma = 0.299 * r + 0.587 * g + 0.114 * b
    lit = luma >= 0.05 * ceiling
    peak, low = np.maximum(np.maximum(r, g), b), np.minimum(np.minimum(r, g), b)
    hue_x, hue_y = 2 * r - g - b, math.sqrt(3) * (g - b)
    return {
        "rgb_luma_std": luma.std(),
        "rgb_saturation_mean": ((peak - low)[lit] / peak[lit]).mean(),
        "rgb_hue_coherence": math.hypot(hue_x.sum(), hue_y.sum()) / np.hypot(hue_x, hue_y).sum(),
    }


def test_it_writes_only_what_the_colour_widget_reads():
    assert set(RgbColourProcessor.OUTPUT_SCHEMA) == set(RgbColourProcessor.OUTPUT_SCHEMA_DESCRIPTIONS) == NAMES
    assert set(metrics_of(solid((200, 100, 50)))) == NAMES


def test_grey_and_tinted_images():
    grey = metrics_of(solid((120, 120, 120)))
    assert grey["rgb_saturation_mean"] == 0 and grey["rgb_hue_coherence"] is None
    assert grey["rgb_luma_std"] == pytest.approx(0, abs=0.01)
    tinted = metrics_of(solid((200, 100, 50)))
    assert tinted["rgb_saturation_mean"] == pytest.approx(0.75)
    assert tinted["rgb_hue_coherence"] == pytest.approx(1.0)


@pytest.mark.parametrize("colours,expected", [
    ([(255, 0, 0), (0, 255, 255)], 0.0),                      # opposite hues cancel
    ([(255, 0, 0), (0, 255, 0), (0, 0, 255)], 0.0),           # three hues cancel
    ([(255, 0, 0), (255, 0, 0), (0, 255, 0)], 1 / math.sqrt(3)),
])
def test_hue_coherence_of_patchworks(colours, expected):
    patchwork = np.concatenate([solid(c) for c in colours], axis=1)
    assert metrics_of(patchwork)["rgb_hue_coherence"] == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize("dtype,ceiling", [(np.uint8, 255.0), (np.uint16, 65535.0)])
def test_matches_a_direct_float64_computation_in_either_layout(dtype, ceiling):
    image = np.random.default_rng(3).integers(0, int(ceiling), size=(60, 70, 3)).astype(dtype)
    interleaved = metrics_of(image, "YXS", ceiling)
    planar = metrics_of(np.moveaxis(image, 2, 0).copy(), "SYX", ceiling)
    for name, value in reference(image, ceiling).items():
        assert interleaved[name] == pytest.approx(value, rel=1e-5), name
        assert planar[name] == pytest.approx(value, rel=1e-5), name


def test_split_chunks_combine_to_the_whole_image():
    image = np.random.default_rng(0).integers(0, 255, size=(64, 80, 3), dtype=np.uint8)
    merged = colour_stats.combine([sums_of(image[:20]), sums_of(image[20:])])
    for name, value in metrics_of(image).items():
        assert merged[name] == pytest.approx(value, rel=1e-6), name


def test_a_chunk_stays_within_the_pipelines_memory_allowance_without_a_hidden_chunk_size():
    # The pipeline budgets a task at one eighth of a worker's memory, so a processor may expand a chunk eightfold.
    image = np.random.default_rng(1).integers(0, 255, size=(1500, 1500, 3), dtype=np.uint8)
    planes = colour_stats.colour_planes(image, "YXS", "S")
    tracemalloc.start()
    colour_stats.chunk_sums(planes, 255.0)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    assert peak < 6 * image.nbytes
    assert not hasattr(colour_stats, "STRIP_PIXELS")


def test_colour_axis_handling():
    image = solid((10, 20, 30), shape=(4, 6))
    assert colour_stats.colour_planes(image, "YXS", "S").shape == (1, 4, 6, 3)
    assert colour_stats.colour_planes(image[None, None], "TCYXS", "S").shape == (1, 4, 6, 3)
    assert colour_stats.colour_planes(image[..., :1], "YXS", "S") is None
    assert colour_stats.colour_planes(image, "YXC", "S") is None


def test_processor_runs_a_chunk_and_aggregates_it():
    processor = RgbColourProcessor()
    row = processor.run_chunk(record_of(solid((200, 100, 50))))
    assert processor.get_aggregation("rgb_saturation_mean")([row], {}) == pytest.approx(0.75)
    declared = processor.run_chunk(record_of(solid((200, 100, 50), shape=(4, 6)), "YXC", {"rgb", "rgb:C"}))
    assert processor.get_aggregation("rgb_saturation_mean")([declared], {}) == pytest.approx(0.75)
    assert processor.run_chunk(record_of(solid((1, 2, 3))[..., :1])) == {}
    assert processor.get_aggregation("rgb_luma_std")([{}], {}) is None


@pytest.mark.parametrize("dim_order,shape,meta,expected", [
    ("YXS", (4, 6, 3), {}, True),
    ("YXS", (4, 6, 4), {}, True),
    ("YXC", (4, 6, 3), {"channel_names": ["R", "G", "B"]}, True),
    ("YXS", (4, 6, 5), {}, False),
    ("YX", (4, 6), {}, False),
    ("YXC", (4, 6, 3), {"channel_names": ["DAPI", "GFP", "RFP"]}, False),
])
def test_only_rgb_images_are_given_to_the_processor(dim_order, shape, meta, expected):
    record = record_from(np.zeros(shape, dtype=np.uint8), {"dim_order": dim_order, "shape": list(shape), "dtype": "uint8", **meta})
    assert is_record_matching_processor(record, RgbColourProcessor.INPUT) == expected
