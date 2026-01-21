from pathlib import Path
import polars as pl
from pixel_patrol_base.io.project_io import export_project
from pixel_patrol_base.api import create_project


def test_export_project_cleans_partial_chunks(tmp_path: Path):
    base_dir = tmp_path / "base"
    base_dir.mkdir()

    flush_dir = tmp_path / "batches"
    flush_dir.mkdir()

    # partial chunk files and a combined file
    p1 = flush_dir / "records_batch_00000.parquet"
    p2 = flush_dir / "records_batch_00001.parquet"
    combined = flush_dir / "records_df.parquet"

    pl.DataFrame({"row_index": [0]}).write_parquet(p1)
    pl.DataFrame({"row_index": [1]}).write_parquet(p2)
    pl.DataFrame({"a": [1]}).write_parquet(combined)

    project = create_project("myproj", str(base_dir))
    project.settings.records_flush_dir = flush_dir
    project.records_df = None

    out_zip = tmp_path / "out.zip"

    export_project(project, out_zip)

    # partial chunks and combined parquet should be removed by default
    assert not p1.exists()
    assert not p2.exists()
    assert not combined.exists()
