import string
from pathlib import Path
import zarr

def is_zarr_store(path: Path) -> bool:
    """
    Robustly checks if a given path is a Zarr store (v2 or v3).

    This function uses the zarr library to attempt opening the store, which
    correctly handles both Zarr v2 and v3 specifications.

    Args:
        path: The pathlib.Path object to check.

    Returns:
        True if the path is a valid Zarr store, False otherwise.
    """
    try:
        store_obj = zarr.open(store=str(path.absolute()), mode='r')

        if isinstance(store_obj, zarr.Group):
            # A group is "processable" if it has any custom attributes.
            # A generic container group will have empty attrs.
            return bool(store_obj.attrs)

        return True

    except Exception as e:
        # Catches any error, indicating it's not a valid or accessible Zarr store.
        return False


def infer_dim_order(n: int) -> str:
    """
    Infer a simple dim order assuming the last two dims are YX.
    Preceding dims are assigned A,B,C,... in order.
    """
    if n <= 2:
        return "YX"[-n:]  # n==0 -> "", n==1 -> "X", n==2 -> "YX"
    return string.ascii_uppercase[: n - 2] + "YX"
