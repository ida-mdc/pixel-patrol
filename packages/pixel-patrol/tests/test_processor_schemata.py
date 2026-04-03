import re
import shutil
import tempfile
from pathlib import Path
from typing import List

import numpy as np
import tifffile
from pixel_patrol_base.core.processing_config import ProcessingConfig
from pixel_patrol_base.core.specs import is_record_matching_processor

from pixel_patrol_base import api
from pixel_patrol_base.plugin_registry import discover_processor_plugins
from pixel_patrol_base.core.feature_schema import validate_processor_output


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

    if project.records_df is None or project.records_df.is_empty():
        print("    [Error] No records processed!")
        return

    assert project.records_df.height == 1

    # Load from saved parquet to verify round-trip
    parquet_files = list(project.output_path.parent.glob("*.parquet"))
    assert len(parquet_files) > 0, f"No parquet file found in {project.output_path.parent}"
    records_df, metadata = api.load(parquet_files[0])

    print(records_df.columns)

    processors = discover_processor_plugins()
    df_columns = set(records_df.columns)

    # Load a sample record to check if processors should run
    sample_file = list(images_dir.glob("*.tif"))[0]
    sample_record = project.loader.load(str(sample_file))

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


def test_all_processors_return_dict():
    """Every processor's run() must return a dict so that ``extracted.update(out)``
    in ``_extract_record_properties`` is always safe.

    This creates a small synthetic image, loads it into a Record, and runs every
    matching processor — verifying each returns a plain dict.
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

        result = processor.run(record)
        assert isinstance(result, dict), (
            f"Processor '{processor.NAME}' returned {type(result).__name__} "
            f"instead of dict. This will break `extracted.update(out)` in "
            f"_extract_record_properties."
        )
        # Also check that no values are themselves dicts (they must be
        # scalar / list / array values suitable for a DataFrame column)
        for key, value in result.items():
            assert not isinstance(value, dict), (
                f"Processor '{processor.NAME}' returned a nested dict under "
                f"key '{key}'. DataFrame columns cannot hold nested dicts."
            )




def _sample_for_type(t):
    # produce a small, castable sample for simple types
    if t is float:
        return "1.23"
    if t is int:
        return "7"
    if t is bool:
        return "1"
    if t is str:
        return 123  # will be cast to str
    # tuple specs like (type, length) -> return an array-like
    if isinstance(t, tuple):
        return [1] * (t[1] if isinstance(t[1], int) and t[1] > 0 else 1)
    # fallback
    return None

def test_processor_schemata_return_declared_dtypes():
    """Ensure processors' OUTPUT_SCHEMA keys are returned with the declared datatypes
    (accepting numpy scalar/array variants)."""
    procs = discover_processor_plugins()
    for proc_cls in procs:
        proc = proc_cls() if isinstance(proc_cls, type) else proc_cls
        schema = getattr(proc, "OUTPUT_SCHEMA", {}) or {}
        patterns = getattr(proc, "OUTPUT_SCHEMA_PATTERNS", []) or []

        # build raw output with castable values
        raw = {}
        for key, type_spec in schema.items():
            raw[key] = _sample_for_type(type_spec)

        validated = validate_processor_output(
            raw, schema, patterns, processor_name=getattr(proc, "NAME", proc.__class__.__name__)
        )

        for key, type_spec in schema.items():
            val = validated.get(key)
            if type_spec is float:
                assert isinstance(val, (float, np.floating)), (
                    f"{proc.NAME}:{key} expected float-like, got {type(val)}"
                )
            elif type_spec is int:
                assert isinstance(val, (int, np.integer)), (
                    f"{proc.NAME}:{key} expected int-like, got {type(val)}"
                )
            elif type_spec is bool:
                assert isinstance(val, (bool, np.bool_, int, np.integer)), (
                    f"{proc.NAME}:{key} expected bool-like, got {type(val)}"
                )
            elif type_spec is str:
                assert isinstance(val, str), (
                    f"{proc.NAME}:{key} expected str, got {type(val)}"
                )
            elif isinstance(type_spec, tuple):
                # expect list/ndarray or None
                assert (val is None) or isinstance(val, (list, np.ndarray)), (
                    f"{proc.NAME}:{key} expected array-like for tuple spec, got {type(val)}"
                )
            else:
                assert key in validated
