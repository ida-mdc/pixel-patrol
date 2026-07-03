"""Tests for ManifestSource discovery (no network - uses a local CSV manifest)."""
from pathlib import Path

import polars as pl

from pixel_patrol_base.plugins.sources.manifest_source import ManifestSource


def _write_manifest(tmp_path: Path) -> Path:
    df = pl.DataFrame({
        "URL_OrigDNA":  ["s3://bucket/p/r01c01-ch1.tiff", "s3://bucket/p/r01c02-ch1.tiff"],
        "URL_OrigMito": ["s3://bucket/p/r01c01-ch2.tiff", "s3://bucket/p/r01c02-ch2.tiff"],
        "Metadata_Plate": ["P1", "P1"],
        "Metadata_Well":  ["A01", "A02"],
    })
    manifest = tmp_path / "load_data.csv"
    df.write_csv(manifest)
    return manifest


def test_can_handle_only_csv():
    src = ManifestSource()
    assert src.can_handle("s3://bucket/p/load_data.csv")
    assert src.can_handle("/local/load_data.csv")
    assert not src.can_handle("s3://bucket/p/image.tiff")
    assert not src.can_handle("/local/dir")


def test_discover_yields_one_record_per_url_column(tmp_path):
    manifest = _write_manifest(tmp_path)
    records = list(ManifestSource(fetch_sizes=False).discover([str(manifest)], accepted_extensions="all"))

    # 2 rows x 2 URL columns = 4 images
    assert len(records) == 4
    urls = [url for url, _ in records]
    assert "s3://bucket/p/r01c01-ch1.tiff" in urls
    assert "s3://bucket/p/r01c02-ch2.tiff" in urls


def test_discover_metadata_and_channel(tmp_path):
    manifest = _write_manifest(tmp_path)
    by_url = {url: meta for url, meta in ManifestSource(fetch_sizes=False).discover([str(manifest)], accepted_extensions="all")}

    meta = by_url["s3://bucket/p/r01c01-ch2.tiff"]
    assert meta["channel"] == "OrigMito"
    assert meta["name"] == "r01c01-ch2.tiff"
    assert meta["parent"] == "s3://bucket/p"
    assert meta["file_extension"] == "tiff"
    assert meta["type"] == "file"
    assert meta["size_bytes"] == 0
    assert meta["modification_date"] is None
    assert meta["Metadata_Plate"] == "P1"
    assert meta["Metadata_Well"] == "A01"
    assert meta["imported_path"] == str(manifest)


def test_discover_filters_by_extension(tmp_path):
    df = pl.DataFrame({
        "URL_A": ["s3://b/x.tiff"],
        "URL_B": ["s3://b/x.png"],
    })
    manifest = tmp_path / "load_data.csv"
    df.write_csv(manifest)

    records = list(ManifestSource(fetch_sizes=False).discover([str(manifest)], accepted_extensions={"tiff"}))
    assert [url for url, _ in records] == ["s3://b/x.tiff"]


def test_fetch_sizes_reads_local_file_sizes(tmp_path):
    # Point the manifest at real local files so fetch_sizes returns true sizes.
    img_a = tmp_path / "a.tiff"; img_a.write_bytes(b"x" * 123)
    img_b = tmp_path / "b.tiff"; img_b.write_bytes(b"y" * 456)
    df = pl.DataFrame({"URL_DNA": [str(img_a)], "URL_Mito": [str(img_b)]})
    manifest = tmp_path / "load_data.csv"
    df.write_csv(manifest)

    by_url = {url: meta for url, meta in ManifestSource(fetch_sizes=True).discover([str(manifest)], accepted_extensions="all")}
    assert by_url[str(img_a)]["size_bytes"] == 123
    assert by_url[str(img_b)]["size_bytes"] == 456


def test_path_metadata_is_attached(tmp_path):
    manifest = _write_manifest(tmp_path)
    src = ManifestSource(
        fetch_sizes=False,
        path_metadata=lambda p: {"Metadata_Site": "FMP", "Metadata_BatchDir": "Batch1"},
    )
    _, meta = next(iter(src.discover([str(manifest)], accepted_extensions="all")))
    assert meta["Metadata_Site"] == "FMP"
    assert meta["Metadata_BatchDir"] == "Batch1"
    # manifest's own Metadata_* columns are still present
    assert meta["Metadata_Plate"] == "P1"
