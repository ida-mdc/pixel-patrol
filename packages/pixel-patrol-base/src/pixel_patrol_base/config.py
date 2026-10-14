MIN_N_EXAMPLE_FILES: int = 1
MAX_N_EXAMPLE_FILES: int = 20
DEFAULT_N_EXAMPLE_FILES: int = 9
MAX_ROWS_DISPLAYED = 100
MAX_COLS_DISPLAYED = 200
SPRITE_SIZE = 64

## Processing defaults
# rows accumulated per output part file before flushing to disk
DEFAULT_ROWS_PER_PART: int = 10_000
MIN_ROWS_PER_PART: int = 100
# max images per task (applies to both batch tasks and container sub-images);
# caps task size to keep workers returning results frequently
DEFAULT_MAX_IMAGES_PER_TASK: int = 200
# RAM per worker in MB; tasks are sized at 1/8 of this, headroom for in-memory expansion
DEFAULT_MB_PER_WORKER: float = 4096.0
MIN_MB_PER_WORKER: float = 50.0
# rows per row group in the final parquet
DEFAULT_PARQUET_ROW_GROUP_SIZE: int = 2048

# Legacy alias - kept so any direct import still works during the transition
DEFAULT_RECORDS_FLUSH_EVERY_N: int = DEFAULT_ROWS_PER_PART

HISTOGRAM_BINS = 256
