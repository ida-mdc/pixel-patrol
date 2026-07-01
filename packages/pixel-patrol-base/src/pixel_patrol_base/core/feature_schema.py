from __future__ import annotations

import re
import logging
from typing import Any, Dict, List, Tuple, Type, Union

import numpy as np

from pixel_patrol_base.core.contracts import PixelPatrolLoader


logger = logging.getLogger(__name__)

Schema = Dict[str, Any]
PatternSpec = List[Tuple[str, Any]]
SchemaType = Union[type, Tuple[type, int]]


def get_requirements_as_patterns(component: Type[PixelPatrolLoader]) -> List[str]:
    """Combine a loader/processor's OUTPUT_SCHEMA keys and OUTPUT_SCHEMA_PATTERNS into
    a flat list of regex strings — useful for introspection and tests."""
    exact = [f"^{re.escape(key)}$" for key in component.OUTPUT_SCHEMA.keys()]
    dynamic = [pat for pat, _typ in (component.OUTPUT_SCHEMA_PATTERNS or [])]
    return exact + dynamic


def patterns_from_processor(prcssr) -> List[str]:
    """Extract regex strings from OUTPUT_SCHEMA_PATTERNS. Accepts class or instance."""
    schema_patterns = getattr(prcssr, "OUTPUT_SCHEMA_PATTERNS", None)
    if schema_patterns is None and hasattr(prcssr, "__class__"):
        schema_patterns = getattr(prcssr.__class__, "OUTPUT_SCHEMA_PATTERNS", None)
    return [getattr(pat, "pattern", pat) for pat, _typ in (schema_patterns or [])]


def validate_processor_output(
    output: Dict[str, Any],
    schema: Schema,
    patterns: PatternSpec | None = None,
    processor_name: str = "unknown",
) -> Dict[str, Any]:
    """Validate and cast processor output against its declared schema.
    Warns on mismatches but never raises — returns best-effort validated output."""
    patterns = patterns or []
    validated = {}
    for key, value in output.items():
        type_spec = _find_matching_spec(key, schema, patterns)
        if type_spec is None:
            logger.warning(f"[{processor_name}] '{key}' not in schema")
            validated[key] = value
        else:
            validated[key] = _cast_value(key, value, type_spec, processor_name)
    return validated


def _parse_schema_type(type_spec: Any) -> Tuple[type, int | None]:
    """Parse a schema type spec into (dtype, expected_size_or_None)."""
    if isinstance(type_spec, tuple) and len(type_spec) == 2:
        return type_spec[0], type_spec[1]
    return type_spec, None


def _find_matching_spec(key: str, schema: Schema, patterns: PatternSpec) -> Any | None:
    """Return the type spec for key: exact match first, then pattern match."""
    if key in schema:
        return schema[key]
    for pattern, type_spec in patterns:
        if re.match(pattern, key):
            return type_spec
    return None


def _cast_value(key: str, value: Any, type_spec: Any, processor_name: str) -> Any:
    """Cast value to match the schema spec. Returns original value with a warning on failure."""
    dtype, expected_size = _parse_schema_type(type_spec)
    try:
        if expected_size is not None:
            arr = np.asarray(value)
            if arr.size != expected_size:
                logger.warning(f"[{processor_name}] '{key}' expected size {expected_size}, got {arr.size}")
                return None
            return arr.astype(dtype)
        if dtype is str:
            return str(value)
        if dtype is bytes:
            return bytes(value)
        return np.array(value, dtype=dtype).reshape(1)[0]
    except Exception as e:
        logger.warning(f"[{processor_name}] Failed to cast '{key}': {e}")
        return value
