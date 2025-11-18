"""Claude Code settings management for hooks"""
from __future__ import annotations
import re
import copy
from pathlib import Path
from typing import Any

from ..core.paths import read_file_with_retry

# ==================== Hook Configuration ====================

# Hook configuration: (hook_type, tool_matcher, command_suffix, timeout)
HOOK_CONFIGS = [
    ('SessionStart', '', 'sessionstart', None),
    ('UserPromptSubmit', '', 'userpromptsubmit', None),
    ('PreToolUse', 'Bash|Task', 'pre', None),
    ('PostToolUse', '', 'post', 86400),
    ('Stop', '', 'poll', 86400),          # Poll for messages (24hr max timeout)
    ('SubagentStop', '', 'subagent-stop', 86400),  # Subagent coordination (24hr max)
    ('Notification', '', 'notify', None),
    ('SessionEnd', '', 'sessionend', None),
]

# Derived from HOOK_CONFIGS - guaranteed to stay in sync
ACTIVE_HOOK_TYPES = [cfg[0] for cfg in HOOK_CONFIGS]
HOOK_COMMANDS = [cfg[2] for cfg in HOOK_CONFIGS]

# NOTE: If you remove a hook type from HOOK_CONFIGS in the future, add it to a
# LEGACY_HOOK_TYPES list for cleanup: LEGACY_HOOK_TYPES = ACTIVE_HOOK_TYPES + ['RemovedHook']
# Then use LEGACY_HOOK_TYPES in _remove_hcom_hooks_from_settings() to clean up old installations.

# Hook removal patterns - used by _remove_hcom_hooks_from_settings()
# Dynamically build from HOOK_COMMANDS to match current and legacy hook formats
_HOOK_ARGS_PATTERN = '|'.join(HOOK_COMMANDS)
HCOM_HOOK_PATTERNS = [
    re.compile(r'\$\{?HCOM'),                                # Current: Environment variable ${HCOM:-...}
    re.compile(r'\bHCOM_ACTIVE.*hcom\.py'),                 # LEGACY: Unix HCOM_ACTIVE conditional
    re.compile(r'IF\s+"%HCOM_ACTIVE%"'),                    # LEGACY: Windows HCOM_ACTIVE conditional
    re.compile(rf'\bhcom\s+({_HOOK_ARGS_PATTERN})\b'),       # LEGACY: Direct hcom command
    re.compile(rf'\buvx\s+hcom\s+({_HOOK_ARGS_PATTERN})\b'), # LEGACY: uvx hcom command
    re.compile(rf'hcom\.py["\']?\s+({_HOOK_ARGS_PATTERN})\b'), # LEGACY: hcom.py with optional quote
    re.compile(rf'["\'][^"\']*hcom\.py["\']?\s+({_HOOK_ARGS_PATTERN})\b(?=\s|$)'),  # LEGACY: Quoted path
    re.compile(r'sh\s+-c.*hcom'),                           # LEGACY: Shell wrapper
]

# ==================== Claude Settings Access ====================

def get_claude_settings_path() -> Path:
    """Get path to global Claude settings file"""
    return Path.home() / '.claude' / 'settings.json'

def load_settings_json(settings_path: Path, default: Any = None) -> dict[str, Any] | None:
    """Load and parse settings JSON file with retry logic"""
    import json
    return read_file_with_retry(
        settings_path,
        lambda f: json.load(f),
        default=default
    )

def _remove_hcom_hooks_from_settings(settings: dict[str, Any]) -> None:
    """Remove hcom hooks from settings dict"""
    if not isinstance(settings, dict) or 'hooks' not in settings:
        return

    if not isinstance(settings['hooks'], dict):
        return

    # Check all active hook types for cleanup
    for event in ACTIVE_HOOK_TYPES:
        if event not in settings['hooks']:
            continue

        # Process each matcher
        updated_matchers = []
        for matcher in settings['hooks'][event]:
            # Fail fast on malformed settings - Claude won't run with broken settings anyway
            if not isinstance(matcher, dict):
                raise ValueError(f"Malformed settings: matcher in {event} is not a dict: {type(matcher).__name__}")

            # Validate hooks field if present
            if 'hooks' in matcher and not isinstance(matcher['hooks'], list):
                raise ValueError(f"Malformed settings: hooks in {event} matcher is not a list: {type(matcher['hooks']).__name__}")

            # Work with a copy to avoid any potential reference issues
            matcher_copy = copy.deepcopy(matcher)

            # Filter out HCOM hooks from this matcher
            non_hcom_hooks = [
                hook for hook in matcher_copy.get('hooks', [])
                if not any(
                    pattern.search(hook.get('command', ''))
                    for pattern in HCOM_HOOK_PATTERNS
                )
            ]

            # Only keep the matcher if it has non-HCOM hooks remaining
            if non_hcom_hooks:
                matcher_copy['hooks'] = non_hcom_hooks
                updated_matchers.append(matcher_copy)
            elif 'hooks' not in matcher or matcher['hooks'] == []:
                # Preserve matchers that never had hooks (missing key or empty list only)
                updated_matchers.append(matcher_copy)

        # Update or remove the event
        if updated_matchers:
            settings['hooks'][event] = updated_matchers
        else:
            del settings['hooks'][event]

    # Remove HCOM from env section
    if 'env' in settings and isinstance(settings['env'], dict):
        settings['env'].pop('HCOM', None)
        # Clean up empty env dict
        if not settings['env']:
            del settings['env']
