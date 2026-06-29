from __future__ import annotations

from typing import Any, Tuple


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
