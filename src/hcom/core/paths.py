"""File system utilities and path management"""
from __future__ import annotations
import os
import sys
import time
import json
import tempfile
from pathlib import Path
from contextlib import contextmanager
from typing import Callable, Any, TextIO

# Conditional imports for file locking
if os.name == 'nt':
    import msvcrt
else:
    import fcntl

from ..shared import IS_WINDOWS

# Constants
FILE_RETRY_DELAY = 0.01  # 10ms delay for file lock retries

# Path constants
LOGS_DIR = ".tmp/logs"
SCRIPTS_DIR = ".tmp/scripts"
FLAGS_DIR = ".tmp/flags"
CONFIG_FILE = "config.env"
ARCHIVE_DIR = "archive"

# ==================== File Locking ====================

@contextmanager
def locked(fp, timeout=5.0):
    """Context manager for cross-platform file locking with timeout"""
    start = time.time()

    if os.name == 'nt':
        fp.seek(0)
        # Lock entire file (0x7fffffff = 2GB max)
        while time.time() - start < timeout:
            try:
                msvcrt.locking(fp.fileno(), msvcrt.LK_NBLCK, 0x7fffffff)
                break
            except OSError:
                time.sleep(0.01)
        else:
            raise TimeoutError(f"Lock timeout after {timeout}s")
    else:
        # Non-blocking with retry
        while time.time() - start < timeout:
            try:
                fcntl.flock(fp.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                time.sleep(0.01)
        else:
            raise TimeoutError(f"Lock timeout after {timeout}s")

    try:
        yield
    finally:
        if os.name == 'nt':
            fp.seek(0)
            msvcrt.locking(fp.fileno(), msvcrt.LK_UNLCK, 0x7fffffff)
        else:
            fcntl.flock(fp.fileno(), fcntl.LOCK_UN)

# ==================== Path Utilities ====================

def hcom_path(*parts: str, ensure_parent: bool = False) -> Path:
    """Build path under ~/.hcom (or HCOM_DIR if set)"""
    base = os.environ.get('HCOM_DIR')  # Override base directory (for per-project isolation)
    path = Path(base) if base else (Path.home() / ".hcom")
    if parts:
        path = path.joinpath(*parts)
    if ensure_parent:
        path.parent.mkdir(parents=True, exist_ok=True)
    return path

def ensure_hcom_directories() -> bool:
    """Ensure all critical HCOM directories exist. Idempotent, safe to call repeatedly.
    Called at hook entry to support opt-in scenarios where hooks execute before CLI commands.
    Returns True on success, False on failure."""
    try:
        for dir_name in [LOGS_DIR, SCRIPTS_DIR, FLAGS_DIR, ARCHIVE_DIR]:
            hcom_path(dir_name).mkdir(parents=True, exist_ok=True)
        return True
    except (OSError, PermissionError):
        return False

# ==================== Atomic File Operations ====================

def atomic_write(filepath: str | Path, content: str) -> bool:
    """Write content to file atomically to prevent corruption (now with NEW and IMPROVED (wow!) Windows retry logic (cool!!!)). Returns True on success, False on failure."""
    filepath = Path(filepath) if not isinstance(filepath, Path) else filepath
    filepath.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(3):
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False, dir=filepath.parent, suffix='.tmp') as tmp:
            tmp.write(content)
            tmp.flush()
            os.fsync(tmp.fileno())

        try:
            os.replace(tmp.name, filepath)
            return True
        except PermissionError:
            if IS_WINDOWS and attempt < 2:
                time.sleep(FILE_RETRY_DELAY)
                continue
            else:
                try: # Clean up temp file on final failure
                    Path(tmp.name).unlink()
                except (FileNotFoundError, PermissionError, OSError):
                    pass
                return False
        except Exception:
            try: # Clean up temp file on any other error
                os.unlink(tmp.name)
            except (FileNotFoundError, PermissionError, OSError):
                pass
            return False

    return False  # All attempts exhausted

def read_file_with_retry(filepath: str | Path, read_func: Callable[[TextIO], Any], default: Any = None, max_retries: int = 3) -> Any:
    """Read file with retry logic for Windows file locking"""
    if not Path(filepath).exists():
        return default

    for attempt in range(max_retries):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return read_func(f)
        except PermissionError:
            # Only retry on Windows (file locking issue)
            if IS_WINDOWS and attempt < max_retries - 1:
                time.sleep(FILE_RETRY_DELAY)
            else:
                # Re-raise on Unix or after max retries on Windows
                if not IS_WINDOWS:
                    raise  # Unix permission errors are real issues
                break  # Windows: return default after retries
        except (json.JSONDecodeError, FileNotFoundError, IOError):
            break  # Don't retry on other errors

    return default

__all__ = [
    'hcom_path',
    'ensure_hcom_directories',
    'atomic_write',
    'read_file_with_retry',
    'locked',
    # Path constants
    'LOGS_DIR',
    'SCRIPTS_DIR',
    'FLAGS_DIR',
    'CONFIG_FILE',
    'ARCHIVE_DIR',
]
