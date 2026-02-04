from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple, Type, Union
import numpy as np
import logging

from pixel_patrol_base.core.contracts import PixelPatrolLoader


logger = logging.getLogger(__name__)

Schema = Dict[str, Any]
PatternSpec = List[Tuple[str, Any]]

SchemaType = Union[type, Tuple[type, int]]

def get_requirements_as_patterns(component: Type[PixelPatrolLoader]) -> List[str]:
    """
    Consolidates a component's (loader or processor) static and dynamic
    column specifications into a single list of regex patterns.

    Args:
        component: An instance of a loader or processor.

    Returns:
        A list of regex strings representing all required columns.
    """
    # 1. Get exact keys from the static specification.
    exact_keys_as_patterns = [
        f"^{re.escape(key)}$" for key in component.OUTPUT_SCHEMA.keys()
    ]

    # 2. Get the regex patterns from the dynamic specification.
    dynamic_patterns = [
        pattern_tuple[0] for pattern_tuple in component.OUTPUT_SCHEMA_PATTERNS
    ]

    return exact_keys_as_patterns + dynamic_patterns


def patterns_from_processor(prcssr) -> List[str]:
    """
    Extract regex strings from a processor's declarative OUTPUT_SCHEMA_PATTERNS.
    Accepts either a class or an instance.
    """
    schema_patterns = getattr(prcssr, "OUTPUT_SCHEMA_PATTERNS", None)
    if schema_patterns is None and hasattr(prcssr, "__class__"):
        schema_patterns = getattr(prcssr.__class__, "OUTPUT_SCHEMA_PATTERNS", None)

    pats: List[str] = []
    if schema_patterns:
        for pat, _typ in schema_patterns:
            pats.append(getattr(pat, "pattern", pat))  # handle compiled or plain string
    return pats


def _parse_schema_type(type_spec: Any) -> Tuple[type, int | None]:
    """
    Parse a schema type specification into (dtype, expected_size).
    Args:
        type_spec: Either a numpy dtype (e.g., np.float32) or a tuple (dtype, size)
                   for fixed-size arrays.
    Returns:
        Tuple of (numpy_dtype, expected_size_or_None)
    """
    if isinstance(type_spec, tuple) and len(type_spec) == 2:
        return type_spec[0], type_spec[1]

    # Scalar type
    return type_spec, None


def _validate_value(key: str, value: Any, type_spec: Any, processor_name: str) -> Any:
    """
    Validate and cast a single value to match the schema specification.
    Returns the validated value, or the original value with a warning on failure.
    """
    dtype, expected_size = _parse_schema_type(type_spec)

    try:
        if expected_size is not None:
            # Array type
            arr = np.asarray(value)

            if arr.size != expected_size:
                logger.warning(f"[{processor_name}] '{key}' expected size {expected_size}, got {arr.size}")
                return None

            return arr.astype(dtype)
        else:
            if dtype is str:
                return str(value)
            return np.array(value, dtype=dtype).reshape(1)[0]
    except Exception as e:
        logger.warning(f"[{processor_name}] Failed to validate '{key}': {e}")
        return value


def _find_matching_spec(key: str, schema: Schema, patterns: PatternSpec) -> Any | None:
    """
    Find the type specification for a key by checking exact match first,
    then pattern match.

    Returns:
        The type specification, or None if no match found.
    """
    # Check exact match
    if key in schema:
        return schema[key]

    # Check patterns
    for pattern, type_spec in patterns:
        if re.match(pattern, key):
            return type_spec

    return None


def validate_processor_output(
        output: Dict[str, Any],
        schema: Schema,
        patterns: PatternSpec | None = None,
        processor_name: str = "unknown"
) -> Dict[str, Any]:
    """
    Validate and cast processor output to match the declared schema.
    Warns on issues but doesn't raise - returns best-effort validated output.
    """
    if patterns is None:
        patterns = []

    validated = {}

    for key, value in output.items():
        type_spec = _find_matching_spec(key, schema, patterns)

        if type_spec is None:
            logger.warning(f"[{processor_name}] '{key}' not in schema")
            validated[key] = value
        else:
            validated[key] = _validate_value(key, value, type_spec, processor_name)

    return validated