"""Make the launcher module importable from the tests directory."""
import sys
import os
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


_REAL_APP_DIR = Path.home() / ".pixel-patrol"


@pytest.fixture(autouse=True)
def _guard_home_dir():
    """
    Ensure no test writes to or creates the real ~/.pixel-patrol directory.

    Records a snapshot of the directory's state (exists + mtime) before each
    test and asserts it is unchanged afterwards.
    """
    existed_before = _REAL_APP_DIR.exists()
    mtime_before = _REAL_APP_DIR.stat().st_mtime if existed_before else None

    yield

    existed_after = _REAL_APP_DIR.exists()

    if not existed_before:
        assert not existed_after, (
            f"~/.pixel-patrol was created by the test — "
            f"make sure all paths are redirected to tmp_path."
        )
    else:
        mtime_after = _REAL_APP_DIR.stat().st_mtime
        assert mtime_after == mtime_before, (
            f"~/.pixel-patrol was modified by the test (mtime changed) — "
            f"make sure all paths are redirected to tmp_path."
        )
