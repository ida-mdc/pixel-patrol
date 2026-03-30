import math

import pytest
import numpy as np
import polars as pl
from PIL import Image
from pathlib import Path

from pixel_patrol_tensorboard.core import (
    generate_projector_checkpoint,
    prepare_embeddings_and_meta,
    SPRITE_SIZE,
)


@pytest.fixture
def sample_data():
    """Provides consistent sample data for tests."""
    embeddings = np.random.rand(10, 5).astype(np.float32)
    meta_df = pl.DataFrame({
        "id": range(10),
        "label": [f"item_{i}" for i in range(10)],
        "value_with_tab": [f"data\t{i}" for i in range(10)]
    })
    return embeddings, meta_df


def test_generate_projector_without_thumbnails(tmp_path: Path, sample_data):
    """
    Tests checkpoint generation with embeddings and metadata only (no images).
    """
    embeddings, meta_df = sample_data
    log_dir = tmp_path / "tb_logs_no_thumb"
    generate_projector_checkpoint(embeddings, meta_df, log_dir)
    config_path = log_dir / "projector_config.pbtxt"
    assert config_path.exists()


def test_embedding_and_metadata_dimensions(tmp_path: Path):
    """
    Verifies that the saved tensor and metadata files have the correct dimensions.
    """
    num_rows = 50
    embedding_dim = 10

    input_embeddings = np.random.rand(num_rows, embedding_dim).astype(np.float32)
    input_meta_df = pl.DataFrame({
        "id": range(num_rows),
        "label": [f"item_{i}" for i in range(num_rows)],
        "category": [f"cat_{i % 5}" for i in range(num_rows)]
    })

    log_dir = tmp_path / "tb_logs_dimension_test"
    generate_projector_checkpoint(input_embeddings, input_meta_df, log_dir)

    tensor_path = log_dir / "00000" / "pixel_patrol_embedding" / "tensors.tsv"
    metadata_path = log_dir / "00000" / "pixel_patrol_embedding" / "metadata.tsv"

    assert tensor_path.exists(), "Tensor file (tensors.tsv) was not created."
    assert metadata_path.exists(), "Metadata file (metadata.tsv) was not created."

    output_meta_df = pl.read_csv(metadata_path, separator='\t', infer_schema=False)
    assert output_meta_df.shape == input_meta_df.shape

    output_embeddings = np.loadtxt(tensor_path, delimiter='\t')
    assert output_embeddings.shape == input_embeddings.shape


def test_generate_projector_with_pil_thumbnails(tmp_path: Path, sample_data):
    """
    Tests checkpoint generation when 'thumbnail' column contains PIL.Image objects.
    """
    embeddings, meta_df = sample_data
    log_dir = tmp_path / "tb_logs_pil"

    num_images = 10
    thumbnails = [Image.new("RGB", (32, 32), color=(i * 20, 0, 0)) for i in range(num_images)]
    meta_df_with_thumbs = meta_df.with_columns(
        pl.Series("thumbnail", thumbnails, dtype=pl.Object)
    )

    generate_projector_checkpoint(embeddings, meta_df_with_thumbs, log_dir)

    sprite_path = log_dir / "00000" / "pixel_patrol_embedding" / "sprite.png"
    assert sprite_path.exists()

    config_path = log_dir / "projector_config.pbtxt"
    assert config_path.exists()
    config_content = config_path.read_text()

    assert 'image_path: "00000/pixel_patrol_embedding/sprite.png"' in config_content
    assert f"single_image_dim: {SPRITE_SIZE}" in config_content

    sprite_img = Image.open(sprite_path)
    grid_dim = math.ceil(math.sqrt(num_images))
    expected_size = grid_dim * SPRITE_SIZE

    assert sprite_img.width == expected_size
    assert sprite_img.height == expected_size


def test_generate_projector_with_numpy_thumbnails(tmp_path: Path, sample_data):
    """
    Tests checkpoint generation when 'thumbnail' column contains NumPy arrays.
    """
    embeddings, meta_df = sample_data
    log_dir = tmp_path / "tb_logs_numpy"

    num_images = 10
    thumbnails = [np.random.rand(32, 32, 3).astype(np.float32) for _ in range(num_images)]
    thumbnails[5] = None

    meta_df_with_thumbs = meta_df.with_columns(
        pl.Series("thumbnail", thumbnails, dtype=pl.Object)
    )

    generate_projector_checkpoint(embeddings, meta_df_with_thumbs, log_dir)

    sprite_path = log_dir / "00000" / "pixel_patrol_embedding" / "sprite.png"
    assert sprite_path.exists()

    config_path = log_dir / "projector_config.pbtxt"
    assert config_path.exists()
    config_content = config_path.read_text()

    assert 'image_path: "00000/pixel_patrol_embedding/sprite.png"' in config_content

    sprite_img = Image.open(sprite_path)
    grid_dim = math.ceil(math.sqrt(num_images))
    expected_size = grid_dim * SPRITE_SIZE

    assert sprite_img.width == expected_size
    assert sprite_img.height == expected_size


def test_metadata_content_and_types(tmp_path: Path):
    """
    Verifies that the data and types in the metadata.tsv file are correctly converted.
    """
    input_df = pl.DataFrame({
        "id": [1, 2, 3],
        "float_val": [0.1, 1.0/3.0, 1.23e4],
        "string_num": ["001", "002", "003"],
        "category": ["Type A", "Type B", None],
    })
    embeddings = np.random.rand(3, 5).astype(np.float32)
    log_dir = tmp_path / "tb_logs_content_test"

    generate_projector_checkpoint(embeddings, input_df, log_dir)

    metadata_path = log_dir / "00000" / "pixel_patrol_embedding" / "metadata.tsv"
    assert metadata_path.exists(), "Metadata file was not created."

    df_from_file = pl.read_csv(metadata_path, separator='\t', infer_schema=False)

    assert list(df_from_file.columns) == ["id", "float_val", "string_num", "category"]
    assert df_from_file.height == 3

    ids = [int(x) for x in df_from_file["id"].to_list()]
    assert ids == [1, 2, 3]

    actual_floats = [float(x) for x in df_from_file["float_val"].to_list()]
    expected_floats = [0.1, 1.0 / 3.0, 12300.0]
    for a, b in zip(actual_floats, expected_floats):
        assert math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)

    assert df_from_file["string_num"].to_list() == ["001", "002", "003"]

    cat_vals = df_from_file["category"].to_list()
    assert cat_vals[0] == "Type A"
    assert cat_vals[1] == "Type B"
    assert cat_vals[2].lower() in ("none", "nan", "")


def test_data_separation_and_types():
    """
    Tests prepare_embeddings_and_meta to ensure it correctly separates
    metadata from embedding features.
    """
    df = pl.DataFrame({
        "path": ["/a/1.png", "/b/2.png", "/c/3.png"],
        "imported_path": ["source1", "source2", "source1"],
        "imported_path_short": ["source1", "source2", "source1"],
        "feature_float": [0.1, 0.5, 0.9],
        "histgram": [[1, 2, 3], [1, 2, 3], [1, 2, 3]],
        "feature_int_high": range(100, 103),
    })

    embeddings, metadata_df = prepare_embeddings_and_meta(df)

    assert embeddings.shape == (3, 2)

    expected_metadata_cols = {"path", "imported_path", "imported_path_short"}
    assert set(metadata_df.columns) == expected_metadata_cols

    assert metadata_df["path"].dtype == pl.String
    assert metadata_df["imported_path"].dtype == pl.String
    assert metadata_df["imported_path_short"].dtype == pl.String


def test_thumbnail_processing_for_black_images(tmp_path):
    """
    Tests the image processing pipeline to diagnose the black image issue.
    """
    log_dir = tmp_path / "tb_logs_image_test"
    dummy_embeddings = np.zeros((3, 2))

    valid_img_arr = np.random.randint(50, 255, size=(32, 32, 3), dtype=np.uint8)
    black_img_arr = np.zeros((32, 32, 3), dtype=np.uint8)

    meta_df = pl.DataFrame({
        "image_id": ["valid_image", "none_image", "black_image"],
        "thumbnail": pl.Series([
            Image.fromarray(valid_img_arr),
            None,
            black_img_arr
        ], dtype=pl.Object)
    })

    generate_projector_checkpoint(dummy_embeddings, meta_df, log_dir)

    sprite_path = log_dir / "00000" / "pixel_patrol_embedding" / "sprite.png"
    assert sprite_path.exists()
    sprite_img = Image.open(sprite_path)
    sprite_arr = np.array(sprite_img)

    patch1 = sprite_arr[0:SPRITE_SIZE, 0:SPRITE_SIZE, :]
    assert not np.all(patch1 == 0)

    patch2 = sprite_arr[0:SPRITE_SIZE, SPRITE_SIZE:2 * SPRITE_SIZE, :]
    assert np.all(patch2 == 0)

    patch3 = sprite_arr[SPRITE_SIZE:2 * SPRITE_SIZE, 0:SPRITE_SIZE, :]
    assert np.all(patch3 == 0)
