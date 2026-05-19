MIN_N_EXAMPLE_FILES: int = 1
MAX_N_EXAMPLE_FILES: int = 20
DEFAULT_N_EXAMPLE_FILES: int = 9
MAX_ROWS_DISPLAYED = 100
MAX_COLS_DISPLAYED = 200
SPRITE_SIZE = 64

## Processing defaults
# output rows accumulated in-memory before writing a chunk to disk
DEFAULT_CHUNK_EVERY_N: int = 10000
# When combining multiple parquet chunks into a single one in memory, we might need more RAM than the raw size of the files.
COMBINE_HEADROOM_RATIO = 1.5
# Maximum number of chunks written to disk before combining is skipped.
MAX_INTERMEDIATE_FLUSHES = 1000

HISTOGRAM_BINS = 256
