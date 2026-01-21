"""Pixel Patrol Base - Core functionality for image processing and analysis."""

import sys
import multiprocessing

# Automatically call freeze_support() for spawn context compatibility
# This ensures multiprocessing works without requiring users to add it manually
_main_module = sys.modules.get('__main__', None)
if _main_module is not None and hasattr(_main_module, '__file__'):
    # We're being imported by a script (not interactive session)
    # Try to call freeze_support() in the main module's context using exec
    if not hasattr(_main_module, '_pixel_patrol_freeze_support_called'):
        try:
            # Execute freeze_support() in the main module's namespace
            # This makes it as if the user wrote it in their script
            exec('import multiprocessing; multiprocessing.freeze_support()', _main_module.__dict__)
            _main_module._pixel_patrol_freeze_support_called = True
        except (RuntimeError, AttributeError, TypeError):
            # Can't execute or already called, that's okay
            # The error will be handled when ProcessPoolExecutor is created
            pass

