import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Dict, Iterator, List, Optional, Set, Tuple, Union

import fsspec
import polars as pl

from pixel_patrol_base.core.uri import is_remote_uri

logger = logging.getLogger(__name__)

# Columns whose values are image URLs, in a Cell Painting Gallery load_data.csv
# each channel has its own column, e.g. URL_OrigDNA, URL_OrigMito.
_URL_COLUMN_PREFIX = "URL_"

# Metadata columns copied onto every image record so they are available for
# grouping/filtering downstream (e.g. Metadata_Plate, Metadata_Well).
_METADATA_COLUMN_PREFIX = "Metadata_"

# fsspec protocols read anonymously (public cloud buckets); "anon" is only
# meaningful to these, not to http(s) or local.
_ANON_SCHEMES = {"s3", "gs", "gcs", "az", "abfs"}


class ManifestSource:
    """Discovers images listed in a manifest CSV instead of by walking a directory.

    Each manifest row references one field-of-view; each URL_* column in that row
    is one channel image. The manifest may be local or remote (s3://, https://);
    remote manifests and images are read anonymously via fsspec. The listed image
    URLs are yielded verbatim so a remote-aware loader can open them directly.

    Targets the Cell Painting Gallery load_data.csv schema (URL_* image columns +
    Metadata_* columns), e.g. cpg0036-EU-OS-bioactives.
    """
    NAME = "manifest"
    # Images are read over the network; the pipeline uses a threaded cluster.
    IO_BOUND = True

    def __init__(self, url_column_prefix: str = _URL_COLUMN_PREFIX, max_rows: Optional[int] = None,
                 fetch_sizes: bool = True, size_workers: int = 64,
                 path_metadata: Optional[Callable[[str], Dict[str, Any]]] = None):
        self._url_prefix = url_column_prefix
        self._max_rows = max_rows          # process only the first N manifest rows (None = all)
        self._fetch_sizes = fetch_sizes      # stat each image URL for its byte size (one HEAD per image)
        # HEADs are I/O-bound; fetch them concurrently so a per-image stat (~100ms
        # each) does not serialize the whole run on the client thread.
        self._size_workers = max(1, size_workers)
        self._path_metadata = path_metadata  # manifest_path -> extra metadata columns (e.g. site from path)
        self._fs_cache: Dict[str, Any] = {}

    def can_handle(self, base: str) -> bool:
        return str(base).lower().endswith(".csv")

    def resolve_bases(self, new_bases: List[str], existing_bases: List[str], base_dir: Optional[Path]) -> List[str]:
        """A manifest base must be a .csv that is readable now (local: exists as a
        file; remote: validated when read at discovery). De-duplicated, kept verbatim."""
        resolved = list(existing_bases)
        seen = set(resolved)
        for base in new_bases:
            if base in seen:
                continue
            if self._is_valid_manifest(base):
                resolved.append(base)
                seen.add(base)
            else:
                logger.warning("ManifestSource: skipping invalid manifest base: %s", base)
        return resolved

    def _is_valid_manifest(self, base: str) -> bool:
        if not self.can_handle(base):
            return False
        return True if is_remote_uri(base) else Path(base).is_file()

    def discover(
        self,
        bases:               List[Union[str, "PurePosixPath"]],
        accepted_extensions: Union[Set[str], str],
        folder_extensions:   Optional[Set[str]] = None,
        base_dir:            Optional[Any] = None,
    ) -> Iterator[Tuple[str, Dict[str, Any]]]:
        extensions = _normalize_extensions(accepted_extensions)
        for manifest in bases:
            yield from self._images_from_manifest(str(manifest), extensions)

    def _images_from_manifest(
        self, manifest_path: str, extensions: Optional[Set[str]]
    ) -> Iterator[Tuple[str, Dict[str, Any]]]:
        df = _read_manifest(manifest_path, self._max_rows)
        url_columns = [c for c in df.columns if c.startswith(self._url_prefix)]
        if not url_columns:
            logger.warning("ManifestSource: no %s* columns in %s; skipping", self._url_prefix, manifest_path)
            return
        metadata_columns = [c for c in df.columns if c.startswith(_METADATA_COLUMN_PREFIX)]
        path_meta = self._path_metadata(manifest_path) if self._path_metadata else {}

        # Flatten the manifest into (url, channel, row_metadata) images.
        images: List[Tuple[str, str, Dict[str, Any]]] = []
        for row in df.iter_rows(named=True):
            row_metadata = {**path_meta, **{c: row[c] for c in metadata_columns}}
            for column in url_columns:
                url = row[column]
                if not url or (extensions is not None and _suffix(url) not in extensions):
                    continue
                images.append((url, column[len(self._url_prefix):], row_metadata))

        if not self._fetch_sizes:
            for url, channel, row_metadata in images:
                yield url, _image_meta(url, channel, manifest_path, row_metadata)
            return

        # Fetch the per-image sizes concurrently (HEADs are I/O-bound) so they do
        # not serialize the run; results stream back in manifest order.
        with ThreadPoolExecutor(max_workers=self._size_workers) as pool:
            stats = pool.map(self._stat, (url for url, _, _ in images))
            for (url, channel, row_metadata), (size, mtime) in zip(images, stats):
                yield url, _image_meta(url, channel, manifest_path, row_metadata, size, mtime)

    def _stat(self, url: str) -> Tuple[int, Optional[datetime]]:
        """Return (size_bytes, modification_date) for a URL via fsspec, or (0, None)."""
        try:
            info = self._filesystem(url).info(url)
        except Exception as e:  # missing object, permissions, backend without info()
            logger.debug("ManifestSource: could not stat %s: %s", url, e)
            return 0, None
        size = int(info.get("size") or info.get("Size") or 0)
        return size, _naive_mtime(info.get("LastModified") or info.get("last_modified") or info.get("mtime"))

    def _filesystem(self, url: str):
        proto = url.split("://", 1)[0] if "://" in url else "file"
        if proto not in self._fs_cache:
            opts: Dict[str, Any] = {}
            if proto in _ANON_SCHEMES:
                opts["anon"] = True
            if proto == "s3":
                # Match the S3 connection pool to our concurrency, else botocore's
                # default of 10 throttles the parallel HEADs.
                opts["config_kwargs"] = {"max_pool_connections": self._size_workers}
            self._fs_cache[proto] = fsspec.filesystem(proto, **opts)
        return self._fs_cache[proto]


def _normalize_extensions(accepted_extensions: Union[Set[str], str]) -> Optional[Set[str]]:
    """Match _discover_files: 'all' → no filter, else a set of dot-prefixed lowercase suffixes."""
    if accepted_extensions == "all":
        return None
    return {e.lower() if e.startswith(".") else "." + e.lower() for e in accepted_extensions}


def _suffix(url: str) -> str:
    return "." + url.rsplit(".", 1)[-1].lower() if "." in url.rsplit("/", 1)[-1] else ""


def _read_manifest(manifest_path: str, max_rows: Optional[int] = None) -> pl.DataFrame:
    storage_options = {"anon": True} if is_remote_uri(manifest_path) else None
    return pl.read_csv(manifest_path, storage_options=storage_options, n_rows=max_rows)


def _naive_mtime(value: Any) -> Optional[datetime]:
    """Normalize an fsspec mtime to a naive datetime, matching the local source.

    Cloud backends return tz-aware UTC datetimes; the local filesystem source
    yields naive datetimes (datetime.fromtimestamp), so drop the tzinfo to keep
    the modification_date column single-typed.
    """
    if isinstance(value, datetime):
        return value.replace(tzinfo=None) if value.tzinfo is not None else value
    return None


def _image_meta(url: str, channel: str, manifest_path: str, row_metadata: Dict[str, Any],
                size_bytes: int = 0, modification_date: Optional[datetime] = None) -> Dict[str, Any]:
    """Build the standard file-metadata record for one image URL.

    size_bytes/modification_date come from a per-URL stat when the source is
    configured to fetch them; otherwise 0/None (a manifest lists URLs without
    filesystem stats).
    """
    name = url.rsplit("/", 1)[-1]
    parent = url.rsplit("/", 1)[0] if "/" in url else ""
    meta: Dict[str, Any] = {
        "path":              url,
        "name":              name,
        "type":              "file",
        "parent":            parent,
        "depth":             0,
        "size_bytes":        size_bytes,
        "file_extension":    name.rsplit(".", 1)[-1].lower() if "." in name else "",
        "modification_date": modification_date,
        "imported_path":     manifest_path,
        "common_base":       manifest_path,
        "channel":           channel,
    }
    meta.update(row_metadata)
    return meta
