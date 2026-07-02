import numpy as np


def dtype_max(dtype):
    """Returns the maximum value for a given numpy dtype."""
    return (
        np.iinfo(dtype) if np.issubdtype(dtype, np.integer) else np.finfo(dtype)
    ).max
