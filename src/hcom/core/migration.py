"""Config migration logic - extracted to avoid circular dependency.

HcomConfig.load() needs to migrate HCOM_PROMPT → HCOM_CLAUDE_ARGS without calling
cmd_reset (which requires config). Once config moves to core/, this breaks the cycle.
"""

from pathlib import Path
from datetime import datetime
import shutil

__all__ = ['needs_migration', 'backup_and_remove_config']


def needs_migration(config_data: dict[str, str]) -> bool:
    """Check if config needs HCOM_PROMPT → HCOM_CLAUDE_ARGS migration.

    Args:
        config_data: Parsed config dictionary

    Returns:
        True if HCOM_PROMPT exists (needs migration)
    """
    return 'HCOM_PROMPT' in config_data


def backup_and_remove_config(config_path: Path) -> None:
    """Backup config with timestamp and remove original.

    Extracted from cmd_reset to avoid circular dependency.

    Args:
        config_path: Path to config file
    """
    if config_path.exists():
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = config_path.parent / f'{config_path.name}.{timestamp}'
        shutil.copy2(config_path, backup_path)
        config_path.unlink()
