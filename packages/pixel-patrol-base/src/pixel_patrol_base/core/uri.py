"""Helpers for distinguishing remote/object-store URIs from local paths.

A single definition of "remote" shared by the source plugins and the processing
pipeline, so path handling stays consistent across the codebase.
"""
import re

# A URI scheme followed by "://" (e.g. "s3://", "https://"). Windows drive
# letters ("C:\\") are deliberately not matched - they have no "//".
_URI_SCHEME = re.compile(r"^(?P<scheme>[a-zA-Z][a-zA-Z0-9+.\-]*)://")


def is_remote_uri(path: str) -> bool:
    """True if path is a remote URI (a scheme other than file://)."""
    match = _URI_SCHEME.match(str(path))
    return bool(match) and match.group("scheme").lower() != "file"
