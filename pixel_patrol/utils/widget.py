import importlib.metadata
from typing import List, Optional
import polars as pl

from pixel_patrol.widgets.widget_interface import PixelPatrolWidget


def output_ratio_of_files_with_column_info(df: pl.DataFrame, column_name: str, display_name: Optional[str] = None, treat_one_as_null: bool = False, numeric: bool = False) -> pl.Series:
    """
    Calculates the ratio of files that have information in a given column.
    Returns a boolean series indicating which rows have valid column info.
    (Keeping core logic, stripped of Streamlit rendering).
    """
    display_name = display_name or column_name

    if numeric:
        df = df.with_columns(pl.col(column_name).cast(pl.Float64, strict=False).alias(column_name))

    if pl.api.types.is_numeric_dtype(df.schema[column_name]):
        column_present = df[column_name].is_not_null() & (df[column_name] != 0)
        if treat_one_as_null:
            column_present = column_present & (df[column_name] != 1)
    else:
        column_present = df[column_name].is_not_null()

    return column_present


def get_required_columns(widgets: List[PixelPatrolWidget]) -> List[str]:
    """
    Aggregates the required columns from a list of PixelPatrolWidget instances.
    """
    columns = []
    for widget in widgets:
        # Check if the instance has the required_columns method (from the ABC)
        if hasattr(widget, "required_columns") and callable(getattr(widget, "required_columns")):
            for column in widget.required_columns():
                if column not in columns:
                    columns.append(column)
    return columns


def load_widgets() -> List[PixelPatrolWidget]:
    """
    Discover and load all widget instances using importlib.metadata entry points.
    Filters for instances of PixelPatrolWidget.
    """
    loaded_widgets: List[PixelPatrolWidget] = []
    # Entry points group remains 'pixel_patrol.widgets' as defined in pyproject.toml
    for entry_point in importlib.metadata.entry_points().select(group='pixel_patrol.widgets'):
        try:
            widget_class = entry_point.load()
            widget_instance = widget_class()
            if isinstance(widget_instance, PixelPatrolWidget):
                loaded_widgets.append(widget_instance)
            else:
                print(f"Warning: Discovered entry point '{entry_point.name}' ({widget_class.__name__}) "
                      f"is not an instance of PixelPatrolWidget. Skipping.")
        except Exception as e:
            print(f"Error loading widget '{entry_point.name}': {e}. Skipping.")
    return loaded_widgets

