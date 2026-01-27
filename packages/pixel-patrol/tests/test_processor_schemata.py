import re
import shutil
import tempfile
from pathlib import Path
from typing import List

import numpy as np
import tifffile
from pixel_patrol_base.core.project_settings import Settings
from pixel_patrol_base.core.specs import is_record_matching_processor

from pixel_patrol_base import api
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

    images_dir = tmp_path / f"source_images"
    test_dir = Path(__file__).parent
    results_dir = test_dir / "benchmark_results"
    results_dir.mkdir(exist_ok=True)

    generate_image_dataset(images_dir, 1, 1, 3, 1, 10, 10)

    # We must create a FRESH project instance to measure processing accurately
    project_name = f"project"
    project = api.create_project(project_name, base_dir=images_dir, loader="bioio")
    project.set_settings(Settings(selected_file_extensions={"tif"}))

    project.process_records()

    if project.records_df is None or project.records_df.is_empty():
        print("    [Error] No records processed!")
        return

    zip_path = tmp_path / f"export.zip"
    api.export_project(project, zip_path)

    imported_project = api.import_project(zip_path)

    assert project.records_df.height == 1

    print(imported_project.records_df.columns)

    processors = discover_processor_plugins()
    df_columns = set(imported_project.records_df.columns)
    
    # Load a sample record to check if processors should run
    sample_file = list(images_dir.glob("*.tif"))[0]
    sample_record = imported_project.loader.load(str(sample_file))

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
        
        # Verify OUTPUT_SCHEMA_PATTERNS columns are present
        if hasattr(processor, 'OUTPUT_SCHEMA_PATTERNS') and processor.OUTPUT_SCHEMA_PATTERNS:
            for pattern, _type in processor.OUTPUT_SCHEMA_PATTERNS:
                # Handle both string and compiled regex patterns
                pattern_str = pattern.pattern if hasattr(pattern, 'pattern') else pattern
                compiled_pattern = re.compile(pattern_str)
                matches = [col for col in df_columns if compiled_pattern.match(col)]

                # Expect at least one column to match the patterns
                assert len(matches) > 0, (
                    f"Processor {processor.NAME}: no columns matched OUTPUT_SCHEMA_PATTERNS. "
                    f"Patterns: {[p.pattern if hasattr(p, 'pattern') else p for p, _ in processor.OUTPUT_SCHEMA_PATTERNS]}. "
                    f"Available columns: {sorted(df_columns)}"
                )
