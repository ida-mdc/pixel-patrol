"""Conformance tests for all registered PixelPatrol loader plugins.

Discovers every loader registered under the 'pixel_patrol.loader_plugins'
entry-point group and verifies that each one satisfies the PixelPatrolLoader
protocol: required class attributes are present with the correct types, and
all required methods exist and are callable.

These tests live in the pixel-patrol umbrella package because it depends on
all plugin packages; running them here ensures every installed loader is
covered without needing to enumerate them by hand.
"""

import importlib.metadata
import inspect
import warnings
from typing import Set

import pytest

from pixel_patrol_base.core.contracts import PixelPatrolLoader

# ── Required class-level attributes and their expected Python types ──────────
REQUIRED_ATTRS = {
    "NAME":                    str,
    "SUPPORTED_EXTENSIONS":    set,
    "OUTPUT_SCHEMA":           dict,
    "OUTPUT_SCHEMA_PATTERNS":  list,
    "FOLDER_EXTENSIONS":       set,
}

# ── Required methods (must exist and be callable) ────────────────────────────
REQUIRED_METHODS = [
    "is_folder_supported",
    "read_header",
    "load",
]


def _discover_loader_classes():
    """Return list of (name, loader_class) for every registered loader plugin.

    Each entry-point is expected to expose a registration function that, when
    called with no arguments, returns a list of loader classes - the same
    convention used by ``pixel_patrol_base.plugin_registry.discover_plugins_from_entrypoints``.

    Entry-points that cannot be imported or whose registration function raises
    are skipped with a warning rather than failing the whole suite - a broken
    plugin should not block tests for working ones.
    """
    eps = importlib.metadata.entry_points(group="pixel_patrol.loader_plugins")
    loaders = []
    for ep in eps:
        try:
            register_fn = ep.load()
            classes = register_fn()
        except Exception as exc:
            warnings.warn(
                f"Skipping loader entry-point '{ep.name}': could not load - {exc}",
                stacklevel=1,
            )
            continue
        for cls in classes:
            name = getattr(cls, "NAME", ep.name)
            loaders.append((name, cls))
    return loaders


_LOADER_CLASSES = _discover_loader_classes()


@pytest.fixture(params=_LOADER_CLASSES, ids=[name for name, _ in _LOADER_CLASSES])
def loader_class(request):
    _, cls = request.param
    return cls


# ── Tests ────────────────────────────────────────────────────────────────────

def test_at_least_one_loader_registered():
    """Sanity check: the entry-point group must resolve at least one loader."""
    assert len(_LOADER_CLASSES) > 0, (
        "No loaders found under 'pixel_patrol.loader_plugins'. "
        "Check that loader packages are installed in the current environment."
    )


@pytest.mark.parametrize("attr,expected_type", list(REQUIRED_ATTRS.items()))
def test_required_attribute_present_and_typed(loader_class, attr, expected_type):
    """Every loader must expose the required class attribute with the right type."""
    assert hasattr(loader_class, attr), (
        f"{loader_class.__name__} is missing required attribute '{attr}'"
    )
    value = getattr(loader_class, attr)
    assert isinstance(value, expected_type), (
        f"{loader_class.__name__}.{attr} should be {expected_type.__name__}, "
        f"got {type(value).__name__}"
    )


@pytest.mark.parametrize("method_name", REQUIRED_METHODS)
def test_required_method_present_and_callable(loader_class, method_name):
    """Every loader must expose the required methods."""
    assert hasattr(loader_class, method_name), (
        f"{loader_class.__name__} is missing required method '{method_name}'"
    )
    method = getattr(loader_class, method_name)
    assert callable(method), (
        f"{loader_class.__name__}.{method_name} exists but is not callable"
    )


def test_name_is_nonempty_string(loader_class):
    assert isinstance(loader_class.NAME, str) and loader_class.NAME.strip(), (
        f"{loader_class.__name__}.NAME must be a non-empty string"
    )


def test_supported_extensions_are_lowercase_strings(loader_class):
    for ext in loader_class.SUPPORTED_EXTENSIONS:
        assert isinstance(ext, str), (
            f"{loader_class.__name__}.SUPPORTED_EXTENSIONS must contain strings, got {type(ext)}"
        )
        assert ext == ext.lower(), (
            f"{loader_class.__name__}.SUPPORTED_EXTENSIONS entry '{ext}' must be lowercase"
        )


def test_folder_extensions_subset_of_supported(loader_class):
    """FOLDER_EXTENSIONS should be a subset of SUPPORTED_EXTENSIONS."""
    extra = loader_class.FOLDER_EXTENSIONS - loader_class.SUPPORTED_EXTENSIONS
    assert not extra, (
        f"{loader_class.__name__}.FOLDER_EXTENSIONS contains entries not in "
        f"SUPPORTED_EXTENSIONS: {extra}"
    )


def test_output_schema_values_are_types(loader_class):
    """OUTPUT_SCHEMA values must be Python types (usable as type hints)."""
    for key, val in loader_class.OUTPUT_SCHEMA.items():
        assert isinstance(val, type) or hasattr(val, "__origin__"), (
            f"{loader_class.__name__}.OUTPUT_SCHEMA['{key}'] = {val!r} is not a type"
        )


def test_output_schema_patterns_structure(loader_class):
    """OUTPUT_SCHEMA_PATTERNS must be a list of (regex_str, type) tuples."""
    for i, item in enumerate(loader_class.OUTPUT_SCHEMA_PATTERNS):
        assert isinstance(item, tuple) and len(item) == 2, (
            f"{loader_class.__name__}.OUTPUT_SCHEMA_PATTERNS[{i}] must be a (str, type) tuple"
        )
        pattern, typ = item
        assert isinstance(pattern, str), (
            f"{loader_class.__name__}.OUTPUT_SCHEMA_PATTERNS[{i}][0] must be a str regex"
        )
        assert isinstance(typ, type) or hasattr(typ, "__origin__"), (
            f"{loader_class.__name__}.OUTPUT_SCHEMA_PATTERNS[{i}][1] must be a type"
        )


def test_read_header_signature(loader_class):
    """read_header must accept (path) - i.e. at least one non-self parameter."""
    sig = inspect.signature(loader_class.read_header)
    params = [p for p in sig.parameters.values()
              if p.name != "self"]
    assert len(params) >= 1, (
        f"{loader_class.__name__}.read_header must accept at least a path argument"
    )


def test_container_extensions_if_declared(loader_class):
    """If CONTAINER_EXTENSIONS is declared it must be a set of lowercase strings
    that are all present in SUPPORTED_EXTENSIONS."""
    if not hasattr(loader_class, "CONTAINER_EXTENSIONS"):
        return
    exts = loader_class.CONTAINER_EXTENSIONS
    assert isinstance(exts, set), (
        f"{loader_class.__name__}.CONTAINER_EXTENSIONS must be a set, got {type(exts).__name__}"
    )
    for ext in exts:
        assert isinstance(ext, str) and ext == ext.lower(), (
            f"{loader_class.__name__}.CONTAINER_EXTENSIONS entry '{ext}' must be a lowercase string"
        )
    extra = exts - loader_class.SUPPORTED_EXTENSIONS
    assert not extra, (
        f"{loader_class.__name__}.CONTAINER_EXTENSIONS contains entries not in "
        f"SUPPORTED_EXTENSIONS: {extra}"
    )


def test_load_range_required_when_container_extensions_nonempty(loader_class):
    """Loaders that declare non-empty CONTAINER_EXTENSIONS must implement load_range
    with the correct signature (path, start, stop)."""
    exts = getattr(loader_class, "CONTAINER_EXTENSIONS", set())
    if not exts:
        return
    assert hasattr(loader_class, "load_range") and callable(loader_class.load_range), (
        f"{loader_class.__name__} declares CONTAINER_EXTENSIONS={exts!r} "
        f"but does not implement load_range"
    )
    sig = inspect.signature(loader_class.load_range)
    params = [p for p in sig.parameters.values() if p.name != "self"]
    assert len(params) >= 3, (
        f"{loader_class.__name__}.load_range must accept at least (path, start, stop)"
    )
