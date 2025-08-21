from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Tuple, Type

from pixel_patrol_base.core.contracts import PixelPatrolLoader

Schema = Dict[str, Any]
PatternSpec = List[Tuple[str, Any]]

def merge_output_schemas(processors: Iterable[object]) -> Tuple[Schema, PatternSpec]:
    static: Schema = {}
    patterns: PatternSpec = []
    for p in processors:
        static.update(getattr(p, "OUTPUT_SCHEMA", {}) or {})
        patterns += getattr(p, "OUTPUT_SCHEMA_PATTERNS", []) or []
    return static, patterns

def infer_col_type(col: str, static: Schema, patterns: PatternSpec):
    if col in static:
        return static[col]
    for pat, typ in patterns:
        if re.match(pat, col):
            return typ
    return None  # unknown => leave as-is

def coerce_row_types(row: Dict[str, Any], static: Schema, patterns: PatternSpec) -> Dict[str, Any]:
    out = dict(row)
    for k, v in row.items():
        want = infer_col_type(k, static, patterns)
        if want is None or v is None:
            continue
        try:
            if want is list and not isinstance(v, list): out[k] = [v]
            elif want is dict and not isinstance(v, dict): pass  # recommend JSON string instead
            else: out[k] = want(v)  # int/float/str
        except Exception:
            # keep original if coercion fails
            pass
    return out


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


def get_dynamic_patterns(component: Type[PixelPatrolLoader]) -> List[str]:
    return [
        pattern_tuple[0] for pattern_tuple in component.OUTPUT_SCHEMA_PATTERNS
    ]

def patterns_from_processor(P) -> List[str]:
    """
    Extract regex strings from a processor's declarative OUTPUT_SCHEMA_PATTERNS.
    Accepts either a class or an instance.
    """
    schema_patterns = getattr(P, "OUTPUT_SCHEMA_PATTERNS", None)
    if schema_patterns is None and hasattr(P, "__class__"):
        schema_patterns = getattr(P.__class__, "OUTPUT_SCHEMA_PATTERNS", None)

    pats: List[str] = []
    if schema_patterns:
        for pat, _typ in schema_patterns:
            pats.append(getattr(pat, "pattern", pat))  # handle compiled or plain string
    return pats
