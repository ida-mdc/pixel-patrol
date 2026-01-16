from pathlib import Path
import polars as pl
from pixel_patrol_base.core import processing


def test_cleanup_partial_chunks_dir(tmp_path: Path):
    d = tmp_path / "batches"
    d.mkdir()

    # Create some partial chunk files and a combined file
    p1 = d / "records_batch_00000.parquet"
    p2 = d / "records_batch_00001.parquet"
    combined = d / "records_df.parquet"

    pl.DataFrame({"row_index": [0]}).write_parquet(p1)
    pl.DataFrame({"row_index": [1]}).write_parquet(p2)
    pl.DataFrame({"a": [1]}).write_parquet(combined)

    # Ensure files exist
    assert p1.exists() and p2.exists() and combined.exists()

    processing._cleanup_partial_chunks_dir(d)

    # partial chunks removed, combined still present
    assert not p1.exists()
    assert not p2.exists()
    assert combined.exists()

    # calling again is a no-op (should not raise)
    processing._cleanup_partial_chunks_dir(d)

    # After cleanup, directory may be removed if empty; in our case combined still exists
    assert d.exists()
