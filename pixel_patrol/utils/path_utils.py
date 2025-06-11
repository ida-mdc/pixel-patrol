from pathlib import Path
from typing import Union


def is_subpath(path_a: Union[str, Path], path_b: Union[str, Path]) -> bool:
    """
    Checks if path_a is a subpath of path_b.
    Paths are resolved to their absolute forms for accurate comparison.
    """
    path_a = Path(path_a).resolve()
    path_b = Path(path_b).resolve()
    try:
        # Check if path_a's parent is path_b, or if path_a is path_b
        # `in` operator for Path objects checks if one is a descendant of another
        return path_a != path_b and path_a.is_relative_to(path_b)
    except ValueError: # Paths on different drives on Windows will raise ValueError for is_relative_to
        return False

def is_superpath(path_a: Union[str, Path], path_b: Union[str, Path]) -> bool:
    """
    Checks if path_a is a superpath (parent) of path_b.
    Paths are resolved to their absolute forms for accurate comparison.
    """
    return is_subpath(path_b, path_a)
