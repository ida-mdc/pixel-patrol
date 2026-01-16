import polars as pl
from pathlib import Path
from pixel_patrol_base.core.processing import _RecordsAccumulator
from pixel_patrol_base.io.project_io import _dict_to_settings


def test_load_existing_chunks_and_chunk_index(tmp_path: Path):
    # Create a fake chunk parquet with row_index values
    flush_dir = tmp_path / "batches"
    flush_dir.mkdir()

    df = pl.DataFrame({"row_index": [0, 5, 9]})
    chunk_file = flush_dir / "records_batch_00002.parquet"
    df.write_parquet(chunk_file)

    acc = _RecordsAccumulator(flush_every_n=10, flush_dir=flush_dir)
    processed = acc.load_existing_chunks()

    assert processed == {0, 5, 9}
    assert acc._chunk_index == 3  # next index should be max_idx + 1
    assert acc._written_files and chunk_file in acc._written_files
