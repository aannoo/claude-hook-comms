#!/usr/bin/env python3
"""Shared constants and utilities for hcom"""
from __future__ import annotations

import sys
import platform
import re
import os
from pathlib import Path

__version__ = "0.6.2"

# ===== Platform Detection =====
IS_WINDOWS = sys.platform == 'win32'
CREATE_NO_WINDOW = 0x08000000  # Windows: prevent console window creation

def is_wsl() -> bool:
    """Detect if running in WSL"""
    if platform.system() != 'Linux':
        return False
    try:
        with open('/proc/version', 'r') as f:
            return 'microsoft' in f.read().lower()
    except (FileNotFoundError, PermissionError, OSError):
        return False

def is_termux() -> bool:
    """Detect if running in Termux on Android"""
    return (
        'TERMUX_VERSION' in os.environ or              # Primary: Works all versions
        'TERMUX__ROOTFS' in os.environ or              # Modern: v0.119.0+
        Path('/data/data/com.termux').exists() or     # Fallback: Path check
        'com.termux' in os.environ.get('PREFIX', '')   # Fallback: PREFIX check
    )

# ===== Message Constants =====
# Message patterns
# Negative lookbehind excludes ._- to prevent matching:
# - email addresses: user@domain.com (preceded by letter)
# - paths: /path/to/file.@test (preceded by period)
# - identifiers: var_@name (preceded by underscore)
# - kebab-case: some-id@mention (preceded by hyphen)
# Capture group must start with alphanumeric (prevents @-test, @_test, @123)
MENTION_PATTERN = re.compile(r'(?<![a-zA-Z0-9._-])@([a-zA-Z0-9][\w-]*)')
AGENT_NAME_PATTERN = re.compile(r'^[a-z-]+$')
TIMESTAMP_SPLIT_PATTERN = re.compile(r'\n(?=\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+\|)')

# Sender constants
SENDER = 'bigboss'
SENDER_EMOJI = '🐳'
MAX_MESSAGES_PER_DELIVERY = 50
MAX_MESSAGE_SIZE = 1048576  # 1MB

# ===== Hook Constants =====
# Stop hook polling interval
STOP_HOOK_POLL_INTERVAL = 0.1  # 100ms between stop hook polls

# PreToolUse hook pattern - matches hcom commands for session_id injection and auto-approval
# - hcom send (any args)
# - hcom stop (no args) | hcom start (no args or --_hcom_sender only)
# - hcom help | hcom --help | hcom -h
# - hcom list (with optional --json, --verbose)
# - hcom watch (with optional --type, --instance, --last, --wait)
# Supports: hcom, uvx hcom, python -m hcom, python hcom.py, python hcom.pyz, /path/to/hcom.py[z]
# Negative lookahead ensures stop/start/done not followed by alias targets (except --_hcom_sender)
# Allows shell operators (2>&1, >/dev/null, |, &&) but blocks identifier-like targets (myalias, 123abc)
HCOM_COMMAND_PATTERN = re.compile(
    r'((?:uvx\s+)?hcom|python3?\s+-m\s+hcom|(?:python3?\s+)?\S*hcom\.pyz?)\s+'
    r'(?:send\b|stop(?!\s+(?:[a-zA-Z_]|[0-9]+[a-zA-Z_])[-\w]*(?:\s|$))|start(?:\s+--_hcom_sender\s+\S+)?(?!\s+(?:[a-zA-Z_]|[0-9]+[a-zA-Z_])[-\w]*(?:\s|$))|done(?:\s+--_hcom_sender\s+\S+)?(?!\s+(?:[a-zA-Z_]|[0-9]+[a-zA-Z_])[-\w]*(?:\s|$))|(?:help|--help|-h)\b|--new-terminal\b|list\b|watch\b)'
)

# ===== Core ANSI Codes =====
RESET = "\033[0m"
DIM = "\033[2m"
BOLD = "\033[1m"
REVERSE = "\033[7m"

# Foreground colors
FG_GREEN = "\033[32m"
FG_CYAN = "\033[36m"
FG_WHITE = "\033[37m"
FG_BLACK = "\033[30m"
FG_GRAY = '\033[90m'
FG_YELLOW = '\033[33m'
FG_RED = '\033[31m'
FG_BLUE = '\033[34m'

# TUI-specific foreground
FG_ORANGE = '\033[38;5;208m'
FG_GOLD = '\033[38;5;220m'
FG_LIGHTGRAY = '\033[38;5;250m'

# Stale instance color (brownish-grey, distinct from exited)
FG_STALE = '\033[38;5;137m'  # Tan/brownish-grey

# Background colors
BG_BLUE = "\033[44m"
BG_GREEN = "\033[42m"
BG_CYAN = "\033[46m"
BG_YELLOW = "\033[43m"
BG_RED = "\033[41m"
BG_GRAY = "\033[100m"

# Stale background (brownish-grey to match foreground)
BG_STALE = '\033[48;5;137m'  # Tan/brownish-grey background

# TUI-specific background
BG_ORANGE = '\033[48;5;208m'
BG_CHARCOAL = '\033[48;5;236m'

# Terminal control
CLEAR_SCREEN = '\033[2J'
CURSOR_HOME = '\033[H'
HIDE_CURSOR = '\033[?25l'
SHOW_CURSOR = '\033[?25h'

# Box drawing
BOX_H = '─'

# ===== Default Config =====
DEFAULT_CONFIG_HEADER = [
    "# HCOM Configuration",
    "#",
    "# All HCOM_* settings (and any env var ie. Claude Code settings)",
    "# can be set here or via environment variables.",
    "# Environment variables and cli args override config file values.",
    "# Put each value on separate lines without comments.",
    "#",
    "# HCOM settings:",
    "#   HCOM_TIMEOUT - seconds before disconnecting idle instance (default: 1800)",
    "#   HCOM_SUBAGENT_TIMEOUT - seconds before disconnecting idle subagents (default: 30)",
    "#   HCOM_TERMINAL - Terminal mode: \"new\", \"here\", or custom command with {script}",
    "#   HCOM_HINTS - Text appended to all messages received by instances",
    "#   HCOM_TAG - Group tag for instances (creates tag-* instances)",
    "#   HCOM_AGENT - Claude code subagent from .claude/agents/, comma-separated for multiple",
    "#   HCOM_CLAUDE_ARGS - Default Claude args (e.g., '-p --model sonnet-4')",
    "#",
    "#",
    "ANTHROPIC_MODEL=",
    "CLAUDE_CODE_SUBAGENT_MODEL=",
]

DEFAULT_CONFIG_DEFAULTS = [
    'HCOM_AGENT=',
    'HCOM_TAG=',
    'HCOM_HINTS=',
    'HCOM_TIMEOUT=1800',
    'HCOM_SUBAGENT_TIMEOUT=30',
    'HCOM_TERMINAL=new',
    r'''HCOM_CLAUDE_ARGS="'say hi in hcom chat'"''',
]

# ===== Status Configuration =====
# Status values stored directly in instance files (no event mapping)
# 'enabled' field is separate from status (participation vs activity)

# Valid status values
STATUS_VALUES = ['active', 'delivered', 'waiting', 'blocked', 'exited', 'stale', 'unknown']

# Status icons
STATUS_ICONS = {
    'active': '▶',
    'delivered': '▷',
    'waiting': '◉',
    'blocked': '■',
    'exited': '○',
    'stale': '⊙',
    'unknown': '◦'
}

# Status colors (foreground)
STATUS_COLORS = {
    'active': FG_GREEN,
    'delivered': FG_CYAN,
    'waiting': FG_BLUE,
    'blocked': FG_RED,
    'exited': FG_GRAY,
    'stale': FG_STALE,
    'unknown': FG_GRAY
}

# STATUS_MAP for watch command (foreground color, icon)
STATUS_MAP = {
    status: (STATUS_COLORS[status], STATUS_ICONS[status])
    for status in STATUS_VALUES
}

# Background colors for statusline display blocks
STATUS_BG_COLORS = {
    'active': BG_GREEN,
    'delivered': BG_CYAN,
    'waiting': BG_BLUE,
    'blocked': BG_RED,
    'exited': BG_GRAY,
    'stale': BG_STALE,
    'unknown': BG_GRAY
}

# Background color map for TUI statusline (background color, icon)
STATUS_BG_MAP = {
    status: (STATUS_BG_COLORS[status], STATUS_ICONS[status])
    for status in STATUS_VALUES
}

# Display order (priority-based sorting)
STATUS_ORDER = [
    "active", "delivered", "waiting",
    "blocked", "stale", "exited", "unknown"
]

# TUI-specific (alias for STATUS_COLORS)
STATUS_FG = STATUS_COLORS

# ===== Pure Utility Functions =====
def format_timestamp(iso_str: str, fmt: str = '%H:%M') -> str:
    """Format ISO timestamp for display - pure function"""
    from datetime import datetime
    try:
        if 'T' in iso_str:
            dt = datetime.fromisoformat(iso_str.replace('Z', '+00:00'))
            return dt.strftime(fmt)
        return iso_str
    except Exception:
        return iso_str[:5] if len(iso_str) >= 5 else iso_str

def format_age(seconds: float) -> str:
    """Format time ago in human readable form - pure function"""
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        return f"{int(seconds/60)}m"
    else:
        return f"{int(seconds/3600)}h"

def get_status_counts(instances: dict[str, dict]) -> dict[str, int]:
    """Count instances by status type - pure data transformation"""
    counts = {s: 0 for s in STATUS_ORDER}
    for info in instances.values():
        status = info.get('status', 'unknown')
        counts[status] = counts.get(status, 0) + 1
    return counts


# ===== Config Parsing Utilities =====
def parse_env_value(value: str) -> str:
    """Parse ENV file value with proper quote and escape handling"""
    value = value.strip()

    if not value:
        return value

    if value.startswith('"') and value.endswith('"') and len(value) >= 2:
        inner = value[1:-1]
        inner = inner.replace('\\\\', '\x00')
        inner = inner.replace('\\n', '\n')
        inner = inner.replace('\\t', '\t')
        inner = inner.replace('\\r', '\r')
        inner = inner.replace('\\"', '"')
        inner = inner.replace('\x00', '\\')
        return inner

    if value.startswith("'") and value.endswith("'") and len(value) >= 2:
        return value[1:-1]

    return value


def format_env_value(value: str) -> str:
    """Format value for ENV file with proper quoting (inverse of parse_env_value)"""
    if not value:
        return value

    # Check if quoting needed for special characters
    needs_quoting = any(c in value for c in ['\n', '\t', '"', "'", ' ', '\r'])

    if needs_quoting:
        # Use double quotes with proper escaping
        escaped = value.replace('\\', '\\\\')  # Escape backslashes first
        escaped = escaped.replace('\n', '\\n')  # Escape newlines
        escaped = escaped.replace('\t', '\\t')  # Escape tabs
        escaped = escaped.replace('\r', '\\r')  # Escape carriage returns
        escaped = escaped.replace('"', '\\"')   # Escape double quotes
        return f'"{escaped}"'

    return value


def parse_env_file(config_path: Path) -> dict[str, str]:
    """Parse ENV file (KEY=VALUE format) with security validation"""
    config: dict[str, str] = {}

    dangerous_chars = ['`', '$', ';', '|', '&', '\n', '\r']

    try:
        content = config_path.read_text(encoding='utf-8')
        for line in content.splitlines():
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' in line:
                key, _, value = line.partition('=')
                key = key.strip()
                value = value.strip()

                if key == 'HCOM_TERMINAL':
                    if any(c in value for c in dangerous_chars):
                        print(
                            f"Warning: Unsafe characters in HCOM_TERMINAL "
                            f"({', '.join(repr(c) for c in dangerous_chars if c in value)}), "
                            f"ignoring custom terminal command",
                            file=sys.stderr
                        )
                        continue
                    if value not in ('new', 'here', 'print') and '{script}' not in value:
                        print(
                            "Warning: HCOM_TERMINAL custom command must include {script} placeholder, "
                            "ignoring",
                            file=sys.stderr
                        )
                        continue

                parsed = parse_env_value(value)
                if key:
                    config[key] = parsed
    except (FileNotFoundError, PermissionError, UnicodeDecodeError):
        pass
    return config


# ===== Claude Args Re-exports =====
# Re-export Claude args for backward compatibility (ui.py depends on these)
from .claude_args import (
    ClaudeArgsSpec,
    resolve_claude_args,
    merge_claude_args,
    merge_system_prompts,
    extract_system_prompt_args,
    validate_conflicts,
    add_background_defaults,
)

__all__ = [
    # Re-exported from claude_args (backward compatibility for ui.py)
    'ClaudeArgsSpec',
    'resolve_claude_args',
    'merge_claude_args',
    'merge_system_prompts',
    'extract_system_prompt_args',
    'validate_conflicts',
    'add_background_defaults',
]

