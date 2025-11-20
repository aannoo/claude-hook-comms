#!/usr/bin/env python3
"""Shared constants and utilities for hcom"""
from __future__ import annotations

import sys
import platform
import re
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Literal

__version__ = "0.6.5"

# ===== Platform Detection =====
IS_WINDOWS = sys.platform == 'win32'
CREATE_NO_WINDOW = 0x08000000  # Windows: prevent console window creation

# ===== Terminal Identity =====
# Windows terminal session identifier fallback
# Used for command identity resolution when HCOM_SESSION_ID not available
# (CLAUDE_ENV_FILE sourcing works on Unix but not Windows)
MAPID = (
    os.environ.get('HCOM_LAUNCH_TOKEN')     # hcom-set, always unique
    or os.environ.get('WT_SESSION')         # Windows Terminal (native)
    or os.environ.get('ConEmuHWND')         # ConEmu/Cmder (native)
    or os.environ.get('WEZTERM_PANE')       # WezTerm (cross-platform)
    or os.environ.get('ALACRITTY_WINDOW_ID') # Alacritty (cross-platform)
    or os.environ.get('WAVETERM_BLOCKID')   # Wave Terminal
    or os.environ.get('ZELLIJ_SESSION_NAME') # Zellij (cross-platform)
    or os.environ.get('TMUX_PANE')          # tmux (Git Bash/WSL)
)

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

# Sender constants
SENDER = 'bigboss'  # CLI sender identity
CLAUDE_SENDER = 'john'  # Fallback when no session_id/MAPID available (edge case in Windows or potential rare claude code thing)
SYSTEM_SENDER = 'hcom'  # System notification identity (launcher, watchdog, etc)
SENDER_EMOJI = '🐳' # Legacy whale, unused but kept here to remind me about cake intake
MAX_MESSAGES_PER_DELIVERY = 50
MAX_MESSAGE_SIZE = 1048576  # 1MB

# ===== Message Identity =====
@dataclass
class SenderIdentity:
    """Sender identity for message routing.

    Single identity (name) with kind for filtering.
    NO routing_id - namespace managed via sender_kind in event data.
    """
    kind: Literal['external', 'instance', 'system']
    name: str  # Display name (stored in events.instance)
    instance_data: dict | None = None  # For kind='instance' only

    @property
    def broadcasts(self) -> bool:
        """External and system senders broadcast to everyone."""
        return self.kind in ('external', 'system')

    @property
    def group_id(self) -> str | None:
        """Group session ID for routing (session-based group membership)."""
        from hcom.core.helpers import get_group_session_id
        return get_group_session_id(self.instance_data)

# ===== Hook Constants =====
# Stop hook polling interval
STOP_HOOK_POLL_INTERVAL = 0.1  # 100ms between stop hook polls

# HCOM invocation pattern - matches all ways to invoke hcom
# Supports: hcom, uvx hcom, python -m hcom, python hcom.py, python hcom.pyz, /path/to/hcom.py[z]
HCOM_INVOCATION_PATTERN = r'(?:uvx\s+)?hcom|python3?\s+-m\s+hcom|(?:python3?\s+)?\S*hcom\.pyz?'

# PreToolUse hook pattern - matches hcom commands for session_id injection and auto-approval
# - hcom send (any args)
# - hcom stop (no args) | hcom start (no args or --_hcom_sender only)
# - hcom help | hcom --help | hcom -h
# - hcom list (with optional --json, -v, --verbose, --sql)
# - hcom watch (with optional --last, --wait, --sql)
# Negative lookahead ensures stop/start not followed by alias targets (except --_hcom_sender)
# Allows shell operators (2>&1, >/dev/null, |, &&) but blocks identifier-like targets (myalias, 123abc)
HCOM_COMMAND_PATTERN = re.compile(
    rf'({HCOM_INVOCATION_PATTERN})\s+'
    r'(?:send\b|stop(?!\s+(?:[a-zA-Z_]|[0-9]+[a-zA-Z_])[-\w]*(?:\s|$))|start(?:\s+--_hcom_sender\s+\S+)?(?!\s+(?:[a-zA-Z_]|[0-9]+[a-zA-Z_])[-\w]*(?:\s|$))|(?:help|--help|-h)\b|--new-terminal\b|list\b|watch\b)'
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
    "#   HCOM_CLAUDE_ARGS - Default Claude args (e.g., '-p --model sonnet')",
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
STATUS_VALUES = ['active', 'idle', 'blocked', 'inactive']

# Status icons
STATUS_ICONS = {
    'active': '▶',
    'idle': '◉',
    'blocked': '■',
    'inactive': '○',
}

# Status colors (foreground)
STATUS_COLORS = {
    'active': FG_GREEN,
    'idle': FG_BLUE,
    'blocked': FG_RED,
    'inactive': FG_GRAY,
}

# STATUS_MAP for watch command (foreground color, icon)
STATUS_MAP = {
    status: (STATUS_COLORS[status], STATUS_ICONS[status])
    for status in STATUS_VALUES
}

# Background colors for statusline display blocks
STATUS_BG_COLORS = {
    'active': BG_GREEN,
    'idle': BG_BLUE,
    'blocked': BG_RED,
    'inactive': BG_GRAY,
}

# Background color map for TUI statusline (background color, icon)
STATUS_BG_MAP = {
    status: (STATUS_BG_COLORS[status], STATUS_ICONS[status])
    for status in STATUS_VALUES
}

# Display order (priority-based sorting)
STATUS_ORDER = ["active", "idle", "blocked", "inactive"]

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
    # Message identity
    'SenderIdentity',
    # Re-exported from claude_args (backward compatibility for ui.py)
    'ClaudeArgsSpec',
    'resolve_claude_args',
    'merge_claude_args',
    'merge_system_prompts',
    'extract_system_prompt_args',
    'validate_conflicts',
    'add_background_defaults',
]

