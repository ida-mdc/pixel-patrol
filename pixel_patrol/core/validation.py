import matplotlib.cm as cm
import logging

logger = logging.getLogger(__name__)

def is_valid_colormap(cmap_name: str) -> bool:
    """
    Checks if a given string is a valid Matplotlib colormap name.
    """
    if not isinstance(cmap_name, str):
        logger.warning(f"Colormap name '{cmap_name}' is not a string type.")
        return False
    try:
        # Using get_cmap() and catching ValueError is the robust way to check.
        # This will work for both builtin and registered colormaps.
        cm.get_cmap(cmap_name)
        return True
    except ValueError:
        return False
    except Exception as e:
        logger.error(f"Unexpected error when checking colormap '{cmap_name}': {e}")
        return False