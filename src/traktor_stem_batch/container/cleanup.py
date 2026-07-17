import atexit
import signal
import shutil
import sys
from pathlib import Path
from typing import Set

# Thread-safe registry of active temporary directories
_active_temp_dirs: Set[Path] = set()
_cleanup_initialized = False

def register_temp_dir(path: Path) -> None:
    """Register a temporary directory for automatic cleanup on crash/exit."""
    _active_temp_dirs.add(path)

def unregister_temp_dir(path: Path) -> None:
    """Unregister a temporary directory after normal cleanup."""
    _active_temp_dirs.discard(path)

def cleanup_all() -> None:
    """Forcefully delete all registered temporary directories."""
    for path in list(_active_temp_dirs):
        try:
            if path.exists():
                shutil.rmtree(path)
        except Exception:
            pass
        _active_temp_dirs.discard(path)

def _signal_handler(signum, frame):
    """Handle termination signals and ensure cleanup before exiting."""
    cleanup_all()
    # Exit with standard code for the signal
    sys.exit(128 + signum)

def init_cleanup_handlers() -> None:
    """Register exit and signal handlers for safe file cleanup."""
    global _cleanup_initialized
    if _cleanup_initialized:
        return
        
    # Register normal exit handler
    atexit.register(cleanup_all)
    
    # Register termination signal handlers
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGABRT):
        try:
            signal.signal(sig, _signal_handler)
        except ValueError:
            # signal.signal can fail if not in the main thread, which is fine
            pass
            
    _cleanup_initialized = True
