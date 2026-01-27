MIN_N_EXAMPLE_FILES: int = 1
MAX_N_EXAMPLE_FILES: int = 20
DEFAULT_N_EXAMPLE_FILES: int = 9
MAX_ROWS_DISPLAYED = 100
MAX_COLS_DISPLAYED = 200
SPRITE_SIZE = 64

## Processing defaults
# rows kept in-memory before optional disk flush
DEFAULT_RECORDS_FLUSH_EVERY_N: int = 10000  
# When combining multiple parquet chunks into a single one in memory, we might need more RAM than the raw size of the files.
COMBINE_HEADROOM_RATIO = 1.5
# Maximum number of intermediate flushes allowed to avoid overwhelming the system with too many tasks.
MAX_INTERMEDIATE_FLUSHES = 1000
