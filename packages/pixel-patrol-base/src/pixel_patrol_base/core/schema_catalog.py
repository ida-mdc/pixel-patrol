"""Build a machine-readable catalog of the final report schema and the plugins
that produce / consume it.

The catalog is assembled from the *registered* plugins (not from a sample file):

- ``loaders``    - every registered loader, its supported extensions and the
                   image-metadata columns it emits.
- ``processors`` - every registered processor, its input requirements and the
                   metric columns it emits.
- ``widgets``    - every viewer widget discovered from the bundled viewer
                   extensions, and the columns it consumes (best-effort; requires
                   Node to read the JS plugin modules).
- ``columns``    - the flattened overall schema: one entry per column with its
                   dtype, description and producing source.

``column_descriptions()`` returns the flat ``column -> description`` map used by
the parquet save path to embed descriptions into the file's field metadata.

Description resolution order for a plugin column: the plugin's own
``OUTPUT_SCHEMA_DESCRIPTIONS`` / ``OUTPUT_SCHEMA_PATTERN_DESCRIPTIONS``, then the
shared maps in :mod:`pixel_patrol_base.core.column_docs`.
"""

from __future__ import annotations

import importlib.metadata
import json
import logging
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from pixel_patrol_base.core import column_docs
from pixel_patrol_base.core.feature_schema import _parse_schema_type
from pixel_patrol_base.core.loader_schema import (
    RASTER_IMAGE_LOADER_SCHEMA,
    RASTER_IMAGE_LOADER_SCHEMA_DESCRIPTIONS,
    RASTER_IMAGE_LOADER_SCHEMA_PATTERNS,
    RASTER_IMAGE_LOADER_SCHEMA_PATTERN_DESCRIPTIONS,
    RASTER_IMAGE_REQUIRED_COLUMNS,
    RASTER_IMAGE_SCHEMA_ID,
    RASTER_IMAGE_SCHEMA_LABEL,
)
from pixel_patrol_base.plugin_registry import (
    discover_plugins_from_entrypoints,
    discover_processor_plugins,
)

_COMMON_SCHEMA_KEYS = set(RASTER_IMAGE_LOADER_SCHEMA)
_COMMON_SCHEMA_PATTERNS = {pat for pat, _ in RASTER_IMAGE_LOADER_SCHEMA_PATTERNS}
_COMMON_PATTERN_DESCRIPTIONS = dict(RASTER_IMAGE_LOADER_SCHEMA_PATTERN_DESCRIPTIONS)

# Pipeline-derived columns: computed by processing.py from image data, not provided by loaders.
# ndim = len(dim_order), num_pixels = prod(shape), size_* = shape per axis.
# Descriptions live in column_docs.BASE_COLUMN_DESCRIPTIONS - single source of truth.
_PIPELINE_COLUMNS: List[Tuple[str, Any]] = [
    ("ndim",       int),
    ("num_pixels", int),
]
_PIPELINE_PATTERNS: List[Tuple[str, Any, str]] = [
    (r"^size_[A-Za-z]$", int, "Extent (number of elements) of this row along the given axis."),
]
_PIPELINE_PATTERN_DESCRIPTIONS = {pat: desc for pat, _, desc in _PIPELINE_PATTERNS}

# Core pixel descriptors every raster-image loader emits (axis order + pixel type).
# A loader feeds the intensity processors when it produces these, even if it only
# declares a subset of the full shared metadata schema (e.g. zarr, aqqua_lmdb).
_RASTER_CORE_KEYS = {"dim_order", "dtype"}

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# dtype rendering
# ---------------------------------------------------------------------------

# numpy scalar dtypes report their own name; these are the only friendly-name overrides.
_NAME_OVERRIDES = {"str": "string", "ndarray": "array", "Array": "array", "List": "list"}


def _scalar_name(dtype: Any) -> str:
    """Render a single (non-array) schema type to a stable, friendly string."""
    if isinstance(dtype, type) and issubclass(dtype, np.generic):
        return np.dtype(dtype).name
    name = getattr(dtype, "__name__", None) or str(dtype)
    return _NAME_OVERRIDES.get(name, name)


def _dtype_str(type_spec: Any) -> str:
    """Render an OUTPUT_SCHEMA value (scalar type or ``(dtype, size)``) to a string."""
    dtype, size = _parse_schema_type(type_spec)
    base = _scalar_name(dtype)
    return f"{base}[{size}]" if size is not None else base


# ---------------------------------------------------------------------------
# description resolution
# ---------------------------------------------------------------------------

def _resolve_column_description(component: Any, column: str) -> str:
    """Find the best description for one of a component's output columns."""
    own = getattr(component, "OUTPUT_SCHEMA_DESCRIPTIONS", None) or {}
    if column in own:
        return own[column]
    for pattern, desc in getattr(component, "OUTPUT_SCHEMA_PATTERN_DESCRIPTIONS", None) or []:
        if re.match(pattern, column):
            return desc
    return column_docs.base_column_description(column) or ""


def _representative_name(pattern: str) -> str:
    """Turn a column regex into a representative concrete name for description lookup."""
    s = pattern.strip("^$")
    return s.replace("[A-Za-z]+", "X").replace("[A-Za-z]", "X")


def _resolve_pattern_description(component: Any, pattern: str) -> str:
    """Describe a pattern-column family declared by a loader/processor."""
    for pat, desc in getattr(component, "OUTPUT_SCHEMA_PATTERN_DESCRIPTIONS", None) or []:
        if pat == pattern:
            return desc
    return column_docs.base_column_description(_representative_name(pattern)) or ""


# ---------------------------------------------------------------------------
# component entries
# ---------------------------------------------------------------------------

def _columns_of(component: Any) -> List[Dict[str, str]]:
    schema = getattr(component, "OUTPUT_SCHEMA", None) or {}
    return [
        {"name": name, "dtype": _dtype_str(spec), "description": _resolve_column_description(component, name)}
        for name, spec in schema.items()
    ]


def _patterns_of(component: Any) -> List[Dict[str, str]]:
    patterns = getattr(component, "OUTPUT_SCHEMA_PATTERNS", None) or []
    return [
        {"pattern": pat, "dtype": _dtype_str(spec), "description": _resolve_pattern_description(component, pat)}
        for pat, spec in patterns
    ]


def _package_of(plugin: Any) -> str:
    """Distribution name a plugin ships in, derived from its module path."""
    top = (type(plugin).__module__ or "").split(".")[0]
    return top.replace("_", "-") if top else ""


def _loader_entry(loader: Any) -> Dict[str, Any]:
    columns = _columns_of(loader)
    patterns = _patterns_of(loader)
    # A raster-image loader emits the core pixel descriptors; it may still declare
    # only a subset of the optional metadata columns (zarr, aqqua_lmdb do). "extra"
    # columns are whatever it emits beyond the shared raster-image schema.
    _PIPELINE_KEYS = {name for name, _ in _PIPELINE_COLUMNS}
    is_raster = _RASTER_CORE_KEYS.issubset(c["name"] for c in columns)
    extra_columns = [c for c in columns if c["name"] not in _COMMON_SCHEMA_KEYS and c["name"] not in _PIPELINE_KEYS] if is_raster else columns
    extra_patterns = [p for p in patterns if p["pattern"] not in _COMMON_SCHEMA_PATTERNS] if is_raster else patterns
    return {
        "name": getattr(loader, "NAME", ""),
        "description": getattr(loader, "DESCRIPTION", "") or "",
        "package": _package_of(loader),
        "extensions": sorted(getattr(loader, "SUPPORTED_EXTENSIONS", set()) or set()),
        "schema": RASTER_IMAGE_SCHEMA_ID if is_raster else None,
        "columns": columns,                 # full set (common + extras) - used for connections
        "patterns": patterns,
        "extra_columns": extra_columns,      # loader-specific columns beyond the common schema
        "extra_patterns": extra_patterns,
    }


def _is_scalar_metric_dtype(dtype: str) -> bool:
    """A scalar numeric column the viewer treats as a plottable metric."""
    return dtype.startswith(("float", "int", "uint")) and "[" not in dtype


def _produces_scalar_metrics(processor: Dict[str, Any]) -> bool:
    """Processor whose outputs are all scalar numeric metrics (e.g. raster-basic)."""
    cols = processor["columns"]
    return bool(cols) and all(_is_scalar_metric_dtype(c["dtype"]) for c in cols)


def _input_spec(processor: Any) -> Dict[str, Any]:
    spec = getattr(processor, "INPUT", None)
    if spec is None:
        return {}
    return {
        "axes": sorted(spec.axes or set()),
        "kinds": sorted(spec.kinds or set()),
        "capabilities": sorted(getattr(spec, "capabilities", None) or set()),
    }


def _processor_entry(processor: Any) -> Dict[str, Any]:
    return {
        "name": getattr(processor, "NAME", ""),
        "description": getattr(processor, "DESCRIPTION", "") or "",
        "package": _package_of(processor),
        "input": _input_spec(processor),
        "output": getattr(processor, "OUTPUT", ""),
        "columns": _columns_of(processor),
    }


# ---------------------------------------------------------------------------
# widgets (JS viewer plugins)
# ---------------------------------------------------------------------------

_DUMP_SCRIPT = Path(__file__).resolve().parents[1] / "viewer-scripts" / "dump_plugins.mjs"


def _viewer_extension_dirs() -> List[Path]:
    """Locate bundled viewer extension directories via entry points.

    Each ``pixel_patrol.viewer_extensions`` entry point loads to a callable that
    returns the directory holding that package's ``extension.json`` + plugin JS.
    """
    dirs: List[Path] = []
    for ep in importlib.metadata.entry_points(group="pixel_patrol.viewer_extensions"):
        try:
            result = ep.load()()
            path = Path(result)
            if path.exists():
                dirs.append(path)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("schema_catalog: could not resolve viewer extension '%s': %s", ep.name, exc)
    return dirs


def _load_widgets() -> List[Dict[str, Any]]:
    """Run the Node dumper over the discovered viewer extensions. Best-effort."""
    node = shutil.which("node")
    dirs = _viewer_extension_dirs()
    if not node or not _DUMP_SCRIPT.exists() or not dirs:
        if not node:
            logger.warning("schema_catalog: Node not found; widget catalog omitted.")
        return []
    try:
        proc = subprocess.run(
            [node, str(_DUMP_SCRIPT), *[str(d) for d in dirs]],
            capture_output=True, text=True, timeout=60, check=True,
        )
        return json.loads(proc.stdout)
    except Exception as exc:
        logger.warning("schema_catalog: widget dump failed (%s); widget catalog omitted.", exc)
        return []


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def discover_loader_plugins() -> List[Any]:
    """Return one instance of every registered loader plugin."""
    instances = []
    for cls in discover_plugins_from_entrypoints("pixel_patrol.loader_plugins"):
        try:
            instances.append(cls())
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("schema_catalog: could not instantiate loader %s: %s", cls, exc)
    return instances


# Base columns that come from the obs-rollup rather than the file-system scan.
_AGG_KEYS = {"obs_level"}

# Left-column producer hierarchy: how the parquet is constructed.
_CATEGORY_ORDER = {"file": 0, "image": 1, "derived": 2, "agg": 3, "metric": 4}


def _friendly_pattern(regex: str) -> str:
    """Turn a column regex into a readable family name, e.g. '^size_[A-Za-z]$' -> 'size_<axis>'."""
    return regex.strip("^$").replace("[A-Za-z]+", "<axis>").replace("[A-Za-z]", "<axis>")


def _emitters_by_key(loaders: List[Dict], field: str, key: str) -> Dict[str, Any]:
    """Map each column/pattern key to the ``(loader_names, packages)`` that emit it,
    so a column is credited to exactly the loaders whose schema declares it."""
    out: Dict[str, Any] = {}
    for loader in loaders:
        for item in loader.get(field, []):
            names, packages = out.setdefault(item[key], ([], []))
            names.append(loader["name"])
            packages.append(loader["package"])
    return out


def _column_view(loaders: List[Dict], processors: List[Dict]) -> List[Dict[str, Any]]:
    """Every column that can appear in a report, tagged with the producer that
    creates it and a category. Pattern columns (per-axis families) are included
    as representative '<axis>' entries with their matching ``regex``.
    """
    cols: Dict[str, Dict[str, Any]] = {}

    def put(name, dtype, desc, producer, category, creators, packages, regex=None):
        if name in cols:
            return
        entry = {"name": name, "dtype": dtype, "description": desc,
                 "producer": producer, "category": category,
                 "creators": creators, "packages": sorted(set(packages))}
        if regex:
            entry["regex"] = regex
        cols[name] = entry

    # Each loader emits its own subset of the metadata schema; credit every column
    # (and pattern family) to the loaders that actually declare it, not to a blanket
    # "raster loaders" list - so partial loaders (zarr, aqqua_lmdb) appear correctly.
    col_emitters = _emitters_by_key(loaders, "columns", "name")
    pat_emitters = _emitters_by_key(loaders, "patterns", "pattern")
    _BASE = ["base processing"]
    _BASE_PKG = ["pixel-patrol-base"]

    # metric columns (one producer per processor)
    for proc in processors:
        for col in proc["columns"]:
            put(col["name"], col["dtype"], col["description"], f"processor:{proc['name']}", "metric",
                [proc["name"]], [proc["package"]])

    # image metadata: shared raster-image columns credited to the loaders that declare them.
    # With no raster loader present there are no image rows, so no image columns are listed.
    raster_loaders = [l for l in loaders if l.get("schema") == RASTER_IMAGE_SCHEMA_ID]
    raster_names = [l["name"] for l in raster_loaders]
    raster_pkgs  = [l["package"] for l in raster_loaders]
    for col in _common_schema_columns():
        names, packages = col_emitters.get(col["name"], ([], []))
        if not names:
            names, packages = raster_names, raster_pkgs
        if names:
            put(col["name"], col["dtype"], col["description"], "image", "image", names, packages)
    for pat, spec in RASTER_IMAGE_LOADER_SCHEMA_PATTERNS:
        names, packages = pat_emitters.get(pat, ([], []))
        if names:
            put(_friendly_pattern(pat), _dtype_str(spec), _COMMON_PATTERN_DESCRIPTIONS.get(pat, ""),
                "image", "image", names, packages, regex=pat)
    for loader in loaders:
        for col in loader["extra_columns"]:
            put(col["name"], col["dtype"], col["description"], "image", "image",
                [loader["name"]], [loader["package"]])

    # pipeline-derived columns: computed by the pipeline from image data, not by loaders.
    for name, spec in _PIPELINE_COLUMNS:
        put(name, _dtype_str(spec), column_docs.base_column_description(name) or "",
            "derived", "derived", _BASE, _BASE_PKG)
    for pat, spec, desc in _PIPELINE_PATTERNS:
        put(_friendly_pattern(pat), _dtype_str(spec), desc,
            "derived", "derived", _BASE, _BASE_PKG, regex=pat)

    # aggregation columns
    put("dim_<axis>", "int", column_docs.base_column_description("dim_z") or "",
        "agg", "agg", _BASE, _BASE_PKG, regex=r"^dim_[A-Za-z]+$")

    # remaining base columns (file-system scan + obs rollup)
    for name, desc in column_docs.BASE_COLUMN_DESCRIPTIONS.items():
        if name in cols:
            continue
        category = "agg" if name in _AGG_KEYS else "file"
        put(name, column_docs.BASE_COLUMN_DTYPES.get(name, ""), desc, category, category, _BASE, _BASE_PKG)

    return sorted(cols.values(), key=lambda c: (_CATEGORY_ORDER.get(c["category"], 9), c["name"]))


def _producers_tree(loaders: List[Dict], processors: List[Dict]) -> List[Dict[str, Any]]:
    """The left-column hierarchy: how the parquet is constructed."""
    return [
        {"id": "base", "label": "Base processing", "children": [
            {"id": "file", "label": "File metadata",
             "description": "Columns from the file-system scan: paths, sizes, extensions and timestamps."},
            {"id": "image", "label": "Image metadata",
             "description": "Image properties the loaders extract into the shared raster-image schema.",
             "loaders": [l["name"] for l in loaders]},
            {"id": "derived", "label": "Derived metadata",
             "description": "Columns the pipeline computes from image data: ndim and num_pixels are "
                            "derived from dim_order and shape; size_<axis> columns are extracted from "
                            "the actual array shape per dimension."},
            {"id": "agg", "label": "Aggregation",
             "description": "Columns added when per-image rows are rolled up into the obs hierarchy "
                            "(obs_level, per-dimension coordinates)."},
        ]},
        {"id": "processors", "label": "Processors", "children": [
            {"id": f"processor:{p['name']}", "label": p["name"], "description": p["description"]}
            for p in processors
        ]},
    ]


def _common_schema_columns() -> List[Dict[str, str]]:
    return [
        {"name": name, "dtype": _dtype_str(spec),
         "description": RASTER_IMAGE_LOADER_SCHEMA_DESCRIPTIONS.get(name, ""),
         "required": name in RASTER_IMAGE_REQUIRED_COLUMNS}
        for name, spec in RASTER_IMAGE_LOADER_SCHEMA.items()
    ]


def _common_schema_patterns() -> List[Dict[str, str]]:
    return [
        {"pattern": pat, "dtype": _dtype_str(spec),
         "description": _COMMON_PATTERN_DESCRIPTIONS.get(pat, "")}
        for pat, spec in RASTER_IMAGE_LOADER_SCHEMA_PATTERNS
    ]


def _loader_schemas(loader_entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The shared schemas referenced by loaders (currently the raster-image one)."""
    users = [l["name"] for l in loader_entries if l.get("schema") == RASTER_IMAGE_SCHEMA_ID]
    if not users:
        return []
    return [{
        "id": RASTER_IMAGE_SCHEMA_ID,
        "label": RASTER_IMAGE_SCHEMA_LABEL,
        "columns": _common_schema_columns(),
        "patterns": _common_schema_patterns(),
        "loaders": users,
    }]


def _glob_to_regex(glob: str) -> str:
    return "^" + re.escape(glob).replace(r"\*", ".*") + "$"


def _input_matches_pattern(inp: str, pattern_regex: str) -> bool:
    """True if a wildcard widget input (e.g. 'size_*') covers a loader's pattern family."""
    if "*" not in inp:
        return False
    return re.match(_glob_to_regex(inp), _representative_name(pattern_regex)) is not None


def _processor_consumes_raster(processor: Dict[str, Any]) -> bool:
    kinds = set(processor.get("input", {}).get("kinds") or [])
    return (not kinds) or "intensity" in kinds or "*" in kinds


def _consumed_columns(inputs: List[str], all_columns: Any) -> set:
    """The exact columns a widget consumes, from its declared ``inputs``.

    Input grammar (resolved against the actual column set - no categories):
      ``"<col>"``   one column
      ``"*"``       every column ...
      ``"!<col>"``  ... minus this one (used together with ``"*"``)
    Glob families like ``"size_*"`` are handled separately by :func:`_family_inputs`.
    """
    excluded = {t[1:] for t in inputs if t.startswith("!")}
    cols = set(all_columns) if "*" in inputs else {t for t in inputs if t in all_columns}
    return cols - excluded


def _family_inputs(inputs: List[str], all_columns: Any) -> List[str]:
    """Declared inputs that name a per-axis column family (a glob like ``"size_*"``)
    rather than an exact column, wildcard, exclusion or the metric sentinel."""
    reserved = {"*", "any metric column"}
    return [t for t in inputs
            if t not in reserved and not t.startswith("!") and t not in all_columns]


def _connections(
    loaders: List[Dict[str, Any]],
    processors: List[Dict[str, Any]],
    widgets: List[Dict[str, Any]],
    columns: List[Dict[str, Any]],
) -> List[Dict[str, str]]:
    """Directed edges showing how the producers fit together:

    loader -> processor (raster data flow), and file/aggregation/loader/processor
    -> widget (column consumption). Node ids are the base producers 'file' / 'agg',
    'loader:<name>', 'processor:<name>' and 'widget:<id>'.
    """
    edges: List[Dict[str, str]] = []
    seen: set = set()

    def add(source: str, target: str, via: str) -> None:
        if source and target and (source, target) not in seen:
            seen.add((source, target))
            edges.append({"source": source, "target": target, "via": via})

    # loader -> processor: every raster-image loader feeds every compatible processor
    # loader -> derived: derived columns only exist when a raster-image loader is present
    raster_loaders = [l for l in loaders if l.get("schema") == RASTER_IMAGE_SCHEMA_ID]
    for loader in raster_loaders:
        for proc in processors:
            if _processor_consumes_raster(proc):
                add(f"loader:{loader['name']}", f"processor:{proc['name']}", "raster image")
        add(f"loader:{loader['name']}", "derived", "image data")

    # which node(s) produce each exact column
    producers: Dict[str, set] = {}
    for loader in loaders:
        for col in loader["columns"]:
            producers.setdefault(col["name"], set()).add(f"loader:{loader['name']}")
    for proc in processors:
        for col in proc["columns"]:
            producers.setdefault(col["name"], set()).add(f"processor:{proc['name']}")
    # base producers (file-system scan, obs aggregation, pipeline-derived) so the
    # widgets that read those columns are linked correctly.
    for col in columns:
        if col["producer"] in ("file", "agg", "derived"):
            producers.setdefault(col["name"], set()).add(col["producer"])
    scalar_metric_nodes = [f"processor:{p['name']}" for p in processors if _produces_scalar_metrics(p)]

    # producer/loader -> widget: resolve each widget's declared inputs to producers.
    # Resolution is driven entirely by the real column -> producer map, so no column
    # categories are baked in here. See _consumed_columns for the input grammar.
    for widget in widgets:
        wid = f"widget:{widget['id']}"
        inputs = (widget.get("required_inputs") or []) + (widget.get("inputs") or [])
        for col in _consumed_columns(inputs, producers):
            for node in producers[col]:
                add(node, wid, col)
        if "any metric column" in inputs:
            for node in scalar_metric_nodes:
                add(node, wid, "metrics")
        for inp in _family_inputs(inputs, producers):   # per-axis families e.g. 'size_*'
            for loader in loaders:
                if any(_input_matches_pattern(inp, p["pattern"]) for p in loader["patterns"]):
                    add(f"loader:{loader['name']}", wid, inp)
            for pat, _, _desc in _PIPELINE_PATTERNS:
                if _input_matches_pattern(inp, pat):
                    add("derived", wid, inp)
    return edges


def build_catalog(include_widgets: bool = True) -> Dict[str, Any]:
    """Assemble the full schema + plugin catalog from registered plugins."""
    loaders = sorted((_loader_entry(l) for l in discover_loader_plugins()), key=lambda e: e["name"])
    processors = sorted((_processor_entry(p) for p in discover_processor_plugins()), key=lambda e: e["name"])
    widgets = _load_widgets() if include_widgets else []
    columns = _column_view(loaders, processors)
    return {
        "producers": _producers_tree(loaders, processors),
        "columns": columns,
        "loader_schemas": _loader_schemas(loaders),
        "loaders": loaders,
        "processors": processors,
        "widgets": widgets,
        "connections": _connections(loaders, processors, widgets, columns),
    }


_JSON_DTYPE_MAP: Dict[str, Dict[str, str]] = {
    "str":      {"type": "string"},
    "string":   {"type": "string"},   # _scalar_name maps Python str → "string"
    "int":      {"type": "integer"},
    "float":    {"type": "number"},
    "datetime": {"type": "string", "format": "date-time"},
    "list":     {"type": "array"},
    "array":    {"type": "array"},    # _scalar_name maps Array/ndarray → "array"
    "dict":     {"type": "object"},
}


def _dtype_to_json(dtype: str) -> Optional[Dict[str, str]]:
    if dtype in _JSON_DTYPE_MAP:
        return _JSON_DTYPE_MAP[dtype]
    if re.match(r"^(int|uint)\d+$", dtype):
        return {"type": "integer"}
    if re.match(r"^float\d+$", dtype):
        return {"type": "number"}
    return None  # blob or unrecognised — omit from JSON Schema


def render_json_schema(catalog: Dict[str, Any]) -> Dict[str, Any]:
    """Render the catalog as a JSON Schema document for one report row.

    Fixed columns go into ``properties``; per-axis families (size_*, dim_*,
    pixel_size_*) go into ``patternProperties`` using their stored regex.
    Blob and unrecognised dtypes are omitted.
    """
    properties: Dict[str, Any] = {}
    pattern_properties: Dict[str, Any] = {}

    for col in catalog["columns"]:
        json_type = _dtype_to_json(col["dtype"])
        if json_type is None:
            continue
        entry: Dict[str, Any] = {**json_type}
        if col.get("description"):
            entry["description"] = col["description"]
        if col.get("regex"):
            entry["title"] = col["name"]   # human-readable family name, e.g. "size_<axis>"
            pattern_properties[col["regex"]] = entry
        else:
            properties[col["name"]] = entry

    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Pixel Patrol Report Row",
        "description": (
            "One row in a Pixel Patrol parquet report. "
            "Fixed columns are in properties; "
            "per-axis families (size_*, dim_*, pixel_size_*) are in patternProperties."
        ),
        "type": "object",
        "properties": properties,
        "patternProperties": pattern_properties,
    }


def column_descriptions() -> Dict[str, str]:
    """Flat ``column -> description`` map for every exact (non-pattern) column.

    Used by the parquet save path to embed descriptions into field metadata.
    """
    out: Dict[str, str] = dict(column_docs.BASE_COLUMN_DESCRIPTIONS)
    out.update(RASTER_IMAGE_LOADER_SCHEMA_DESCRIPTIONS)
    for comp in discover_loader_plugins() + discover_processor_plugins():
        for name in getattr(comp, "OUTPUT_SCHEMA", None) or {}:
            desc = _resolve_column_description(comp, name)
            if desc:
                out[name] = desc
    return out
