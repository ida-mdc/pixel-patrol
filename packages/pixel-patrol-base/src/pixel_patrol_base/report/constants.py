"""Shared constants used across report modules."""

# Global controls IDs:

PALETTE_SELECTOR_ID = "palette-selector"
GLOBAL_CONFIG_STORE_ID = "global-config-store"
FILTERED_INDICES_STORE_ID = "global-filtered-indices-store"

GLOBAL_GROUPBY_COLS_ID = "global-groupby-cols"
GLOBAL_FILTER_COLUMN_ID = "global-filter-column"
GLOBAL_FILTER_OP_ID = "global-filter-op"
GLOBAL_FILTER_TEXT_ID = "global-filter-text"
GLOBAL_DIM_FILTER_TYPE = "global-dim-filter"  # _TYPE refers to a dynamic group

GLOBAL_APPLY_BUTTON_ID = "global-apply-button"
GLOBAL_RESET_BUTTON_ID = "global-reset-button"

EXPORT_CSV_BUTTON_ID = "export-csv-button"
EXPORT_PROJECT_BUTTON_ID = "export-project-button"
EXPORT_CSV_DOWNLOAD_ID = "export-csv-download"
EXPORT_PROJECT_DOWNLOAD_ID = "export-project-download"
SAVE_SNAPSHOT_BUTTON_ID = "save-snapshot-button"
SAVE_SNAPSHOT_DOWNLOAD_ID = "save-snapshot-download"

## Grouping and filtering:

DEFAULT_REPORT_GROUP_COL = "imported_path_short"
NO_GROUPING_COL = "common_base"
NO_GROUPING_LABEL = "(NO GROUPING)"

MAX_UNIQUE_GROUP = 12

GC_GROUP_COL = "group_col"
GC_FILTER = "filter"
GC_DIMENSIONS = "dimensions"

GROUPING_COL_PREFIX = "__grouping__"
MISSING_LABEL = "missing"

MAX_RECORDS_IN_MENU = 500

MIXED_GROUPING_COLOR = "#cccccc"