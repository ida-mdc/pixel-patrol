MIN_N_EXAMPLE_FILES: int = 1
MAX_N_EXAMPLE_FILES: int = 20
DEFAULT_N_EXAMPLE_FILES: int = 9
MAX_ROWS_DISPLAYED = 100
MAX_COLS_DISPLAYED = 200
SPRITE_SIZE = 64

## Processing defaults
# rows accumulated per output part file before flushing to disk
DEFAULT_ROWS_PER_PART: int = 10_000
# max files per batch task; caps batch size for small-file datasets to keep
# workers returning results frequently and the progress bar responsive
DEFAULT_MAX_FILES_PER_TASK: int = 50

# Legacy alias — kept so any direct import still works during the transition
DEFAULT_RECORDS_FLUSH_EVERY_N: int = DEFAULT_ROWS_PER_PART

HISTOGRAM_BINS = 256
