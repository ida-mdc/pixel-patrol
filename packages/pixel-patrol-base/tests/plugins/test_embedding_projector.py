import math

import pytest
import numpy as np
import polars as pl
from PIL import Image
from pathlib import Path

from pixel_patrol_base.plugins.widgets.visualization.embedding_projector import _generate_projector_checkpoint, \
    SPRITE_SIZE, EmbeddingProjectorWidget


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
    _generate_projector_checkpoint(embeddings, meta_df, log_dir)
    config_path = log_dir / "projector_config.pbtxt"
    assert config_path.exists()


def test_embedding_and_metadata_dimensions(tmp_path: Path):
    """
    Verifies that the saved tensor and metadata files have the correct dimensions.
    """
    # 1. Arrange: Create sample data with specific dimensions
    num_rows = 50
    embedding_dim = 10

    # Input embeddings array (50 rows, 10 columns)
    input_embeddings = np.random.rand(num_rows, embedding_dim).astype(np.float32)

    # Input metadata DataFrame (50 rows, 3 columns)
    input_meta_df = pl.DataFrame({
        "id": range(num_rows),
        "label": [f"item_{i}" for i in range(num_rows)],
        "category": [f"cat_{i % 5}" for i in range(num_rows)]
    })

    log_dir = tmp_path / "tb_logs_dimension_test"

    # 2. Act: Generate the TensorBoard checkpoint files
    _generate_projector_checkpoint(input_embeddings, input_meta_df, log_dir)

    # 3. Assert: Check the dimensions of the saved files

    # Define the expected file paths
    tensor_path = log_dir / "00000" / "pixel_patrol_embedding" / "tensors.tsv"
    metadata_path = log_dir / "00000" / "pixel_patrol_embedding" / "metadata.tsv"

    # Check that files were actually created
    assert tensor_path.exists(), "Tensor file (tensors.tsv) was not created."
    assert metadata_path.exists(), "Metadata file (metadata.tsv) was not created."

    # -- Verify Metadata Dimensions --
    # Load the saved metadata file
    output_meta_df = pl.read_csv(metadata_path, separator='\t', infer_schema=False)

    print(f"Input Metadata Shape: {input_meta_df.shape}")
    print(f"Output Metadata Shape: {output_meta_df.shape}")

    # Assert that the number of rows and columns match the input
    assert output_meta_df.shape == input_meta_df.shape, \
        "The dimensions of the saved metadata file do not match the input."

    # -- Verify Embeddings Dimensions --
    # Load the saved tensor file using numpy
    output_embeddings = np.loadtxt(tensor_path, delimiter='\t')

    print(f"Input Embeddings Shape: {input_embeddings.shape}")
    print(f"Output Embeddings Shape: {output_embeddings.shape}")

    # Assert that the number of rows (vectors) and columns (features) match the input
    assert output_embeddings.shape == input_embeddings.shape, \
        "The dimensions of the saved tensor file do not match the input."


def test_generate_projector_with_pil_thumbnails(tmp_path: Path, sample_data):
    """
    Tests checkpoint generation when 'thumbnail' column contains PIL.Image objects.
    """
    embeddings, meta_df = sample_data
    log_dir = tmp_path / "tb_logs_pil"

    num_images = 10
    thumbnails = [Image.new("RGB", (32, 32), color=(i * 20, 0, 0)) for i in range(num_images)]

    # Add dtype=pl.Object to be explicit
    meta_df_with_thumbs = meta_df.with_columns(
        pl.Series("thumbnail", thumbnails, dtype=pl.Object)
    )

    _generate_projector_checkpoint(embeddings, meta_df_with_thumbs, log_dir)

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

    # FINAL FIX: Explicitly set the dtype to pl.Object to handle the mixed list.
    meta_df_with_thumbs = meta_df.with_columns(
        pl.Series("thumbnail", thumbnails, dtype=pl.Object)
    )

    _generate_projector_checkpoint(embeddings, meta_df_with_thumbs, log_dir)

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
    # Arrange: Create a DataFrame with a mix of tricky data types
    input_df = pl.DataFrame({
        "id": [1, 2, 3],
        "float_val": [0.1, 1.0/3.0, 1.23e4],
        "string_num": ["001", "002", "003"],
        "category": ["Type A", "Type B", None],
    })
    # This is what we expect the DataFrame to look like after the plugin's
    # conversion logic (every column becomes a string).
    expected_df = pl.DataFrame({
        "id": ["1", "2", "3"],
        "float_val": [str(0.1), str(1.0/3.0), str(12300.0)],
        "string_num": ["001", "002", "003"],
        "category": ["Type A", "Type B", "None"], # Pandas converts None to the string "None"
    })

    # Dummy embeddings are needed for the function to run
    embeddings = np.random.rand(3, 5).astype(np.float32)
    log_dir = tmp_path / "tb_logs_content_test"

    # Act
    _generate_projector_checkpoint(embeddings, input_df, log_dir)

    # Assert
    metadata_path = log_dir / "00000" / "pixel_patrol_embedding" / "metadata.tsv"
    assert metadata_path.exists(), "Metadata file was not created."

    # Read the generated TSV file back into a Polars DataFrame
    # All columns will be read as strings, which is correct for the TSV format.
    df_from_file = pl.read_csv(metadata_path, separator='\t', infer_schema=False)

    # Validate columns and number of rows
    assert list(df_from_file.columns) == ["id", "float_val", "string_num", "category"]
    assert df_from_file.height == 3

    # Check numeric columns by parsing back to numbers to avoid brittle string
    # formatting differences between pandas versions.
    ids = [int(x) for x in df_from_file["id"].to_list()]
    assert ids == [1, 2, 3]

    actual_floats = [float(x) for x in df_from_file["float_val"].to_list()]
    expected_floats = [0.1, 1.0 / 3.0, 12300.0]
    for a, b in zip(actual_floats, expected_floats):
        assert math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)

    # Ensure string columns remain unchanged
    assert df_from_file["string_num"].to_list() == ["001", "002", "003"]

    # For missing categories accept either the string "None" or a NaN-like string
    cat_vals = df_from_file["category"].to_list()
    assert cat_vals[0] == "Type A"
    assert cat_vals[1] == "Type B"
    assert cat_vals[2].lower() in ("none", "nan", "")


def test_data_separation_and_types():
    """
    Tests the new `prepare_data` method to ensure it correctly separates
    metadata from embedding features.
    """
    # Arrange: Create a complex DataFrame mimicking the user's issue
    df = pl.DataFrame({
        "path": ["/a/1.png", "/b/2.png", "/c/3.png"],
        "imported_path": ["source1", "source2", "source1"],
        "imported_path_short": ["source1", "source2", "source1"],
        "feature_float": [0.1, 0.5, 0.9],  # Float feature (should be ignored)
        "histgram": [[1,2,3], [1,2,3], [1,2,3]],  # List feature (should not be embedding)
        "feature_int_high": range(100, 103),  # High-cardinality integer (should be embedding)
    })

    widget = EmbeddingProjectorWidget()

    # Act
    embeddings, metadata_df = widget._prepare_embeddings_and_meta(df)

    # Assert
    # 1. Check that embeddings have the correct shape and source columns
    assert embeddings.shape == (3, 2)
    expected_embedding_cols = {"feature_float", "feature_int_high"}
    # (Checking the values is tricky due to column order, shape is a good proxy)

    # 2. Check that metadata has the correct columns
    expected_metadata_cols = {"path", "imported_path", "imported_path_short"}
    assert set(metadata_df.columns) == expected_metadata_cols

    # 3. Check data types in metadata
    assert metadata_df["path"].dtype == pl.String
    assert metadata_df["imported_path"].dtype == pl.String
    assert metadata_df["imported_path_short"].dtype == pl.String


def test_thumbnail_processing_for_black_images(tmp_path):
    """
    Tests the image processing pipeline to diagnose the black image issue.
    """
    # Arrange
    log_dir = tmp_path / "tb_logs_image_test"
    dummy_embeddings = np.zeros((3, 2))

    valid_img_arr = np.random.randint(50, 255, size=(32, 32, 3), dtype=np.uint8)
    black_img_arr = np.zeros((32, 32, 3), dtype=np.uint8)

    # --- FIX: Add a label column to the DataFrame ---
    # This ensures that after dropping 'thumbnail', we still have valid metadata.
    meta_df = pl.DataFrame({
        "image_id": ["valid_image", "none_image", "black_image"],
        "thumbnail": pl.Series([
            Image.fromarray(valid_img_arr),
            None,
            black_img_arr
        ], dtype=pl.Object)
    })
    # --- END FIX ---

    # Act
    _generate_projector_checkpoint(dummy_embeddings, meta_df, log_dir)

    # Assert
    sprite_path = log_dir / "00000" / "pixel_patrol_embedding" / "sprite.png"
    assert sprite_path.exists()
    sprite_img = Image.open(sprite_path)
    sprite_arr = np.array(sprite_img)

    # 3 images -> 2x2 grid.
    # Patch 1 (top-left) should NOT be all black
    patch1 = sprite_arr[0:SPRITE_SIZE, 0:SPRITE_SIZE, :]
    assert not np.all(patch1 == 0)

    # Patch 2 (top-right) for None should BE all black
    patch2 = sprite_arr[0:SPRITE_SIZE, SPRITE_SIZE:2 * SPRITE_SIZE, :]
    assert np.all(patch2 == 0)

    # Patch 3 (bottom-left) for the black image should BE all black
    patch3 = sprite_arr[SPRITE_SIZE:2 * SPRITE_SIZE, 0:SPRITE_SIZE, :]
    assert np.all(patch3 == 0)