import re
import shutil
import tempfile
from pathlib import Path
from typing import List

import numpy as np
import polars as pl
import tifffile

from pixel_patrol_base import api
from pixel_patrol_base.core.processing_config import ProcessingConfig
from pixel_patrol_base.core.specs import is_record_matching_processor
from pixel_patrol_base.plugin_registry import discover_processor_plugins


def generate_image_dataset(
    base_dir: Path, num_files: int, t: int, c: int, z: int, y: int, x: int
) -> List[Path]:
    """Generates the dataset once per test case."""
    if base_dir.exists():
        shutil.rmtree(base_dir)
    base_dir.mkdir(parents=True)

    files = []
    for i in range(num_files):
        p = base_dir / f"img_{i:04d}.tif"
        create_synthetic_tiff(p, t, c, z, y, x)
        files.append(p)
    return files


def create_synthetic_tiff(
    file_path: Path, t: int, c: int, z: int, y: int, x: int, dtype=np.uint8
) -> None:
    shape = (t, c, z, y, x)
    data = np.random.randint(0, 256, size=shape, dtype=dtype)
    tifffile.imwrite(str(file_path), data, photometric='minisblack')


def test_processor_schemata():
    tmp_path = Path(tempfile.gettempdir())

    images_dir = tmp_path / "source_images"
    test_dir = Path(__file__).parent
    results_dir = test_dir / "benchmark_results"
    results_dir.mkdir(exist_ok=True)

    output_path = tmp_path / "output.parquet"

    generate_image_dataset(images_dir, 1, 1, 3, 1, 10, 10)

    project_name = "project"
    project = api.create_project(project_name, base_dir=images_dir, loader="bioio", output_path=output_path)

    processing_config = ProcessingConfig(
        selected_file_extensions={"tif"},
    )
    project.process_records(processing_config=processing_config)

    df = project.records_df
    if df is None or df.is_empty():
        print("    [Error] No records processed!")
        return

    assert project.records_df.height >= 1  # one global row + per-slice rows

    # Load from saved parquet to verify round-trip
    parquet_files = list(project.output_path.parent.glob("*.parquet"))
    assert len(parquet_files) > 0, f"No parquet file found in {project.output_path.parent}"
    records_df, metadata = api.load(parquet_files[0])

    processors = discover_processor_plugins()
    df_columns = set(records_df.columns)

    # Load a sample record to check if processors should run
    sample_file = list(images_dir.glob("*.tif"))[0]
    sample_record = project.loader.load(sample_file)

    for processor in processors:
        # Check if processor should run on the test dataset
        if not is_record_matching_processor(sample_record, processor.INPUT):
            continue

        # Verify OUTPUT_SCHEMA columns are present
        for col_name in processor.OUTPUT_SCHEMA.keys():
            assert col_name in df_columns, (
                f"Processor {processor.NAME}: expected column '{col_name}' from OUTPUT_SCHEMA "
                f"not found in records_df. Available columns: {sorted(df_columns)}"
            )


def test_all_processors_return_dict():
    """Every processor's run_chunk() must return a dict.

    This creates a small synthetic image, loads it into a Record, and runs every
    matching processor - verifying each returns a plain dict.
    """
    from pixel_patrol_loader_bio.plugins.loaders.bioio_loader import BioIoLoader

    tmp_path = Path(tempfile.mkdtemp())

    # Create a small 5-D TIFF that satisfies most processor input specs
    t, c, z, y, x = 1, 2, 1, 10, 10
    data = np.random.randint(0, 256, size=(t, c, z, y, x), dtype=np.uint8)
    tif_path = tmp_path / "probe.tif"
    tifffile.imwrite(str(tif_path), data, photometric='minisblack')

    loader = BioIoLoader()
    record = loader.load(str(tif_path))

    processors = discover_processor_plugins()
    for processor in processors:
        if not is_record_matching_processor(record, processor.INPUT):
            continue

        result = processor.run_chunk(record)
        assert isinstance(result, dict), (
            f"Processor '{processor.NAME}' returned {type(result).__name__} "
            f"instead of dict. Processors must return Dict from run_chunk()."
        )
