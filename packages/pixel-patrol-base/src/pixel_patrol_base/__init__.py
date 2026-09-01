import logging
import sys
import warnings

from tqdm import tqdm as _tqdm


# numcodecs/checksum32.py prepends its own 'once' filter right before warning,
# so filterwarnings("ignore") can't win. Patching showwarning is the only fix.
_sw = warnings.showwarning
def _filtered_showwarning(m, c, f, l, *a, **kw):
    if c is DeprecationWarning and "numcodecs" in str(f):
        return
    if c is UserWarning and "Could not parse tiff pixel size" in str(m):
        return
    _sw(m, c, f, l, *a, **kw)
warnings.showwarning = _filtered_showwarning


class _TqdmHandler(logging.StreamHandler):
    """Routes log output through tqdm.write to avoid colliding with progress bars."""
    def emit(self, record):
        try:
            _tqdm.write(self.format(record), file=sys.stderr)
        except Exception:
            self.handleError(record)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    handlers=[_TqdmHandler()],
)

