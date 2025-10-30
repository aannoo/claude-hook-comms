#!/usr/bin/env python3
"""
HCOM TUI - Interactive Menu Interface
Part of HCOM (Claude Hook Comms)
"""
import os
import sys
import re
import select
import shlex
import shutil
import subprocess
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from enum import Enum
from typing import List, Optional, Tuple, Literal

# Import from shared module (constants and pure utilities)
try:
    from .shared import (
        # ANSI codes
        RESET, BOLD, DIM, REVERSE,
        FG_GREEN, FG_CYAN, FG_WHITE, FG_BLACK, FG_GRAY, FG_YELLOW, FG_RED, FG_BLUE,
        FG_ORANGE, FG_GOLD, FG_LIGHTGRAY,
        BG_GREEN, BG_CYAN, BG_YELLOW, BG_RED, BG_BLUE, BG_GRAY,
        BG_ORANGE, BG_CHARCOAL,
        CLEAR_SCREEN, CURSOR_HOME, HIDE_CURSOR, SHOW_CURSOR,
        BOX_H,
        # Config
        DEFAULT_CONFIG_HEADER, DEFAULT_CONFIG_DEFAULTS,
        # Status configuration
        STATUS_MAP, STATUS_ORDER, STATUS_FG, STATUS_BG_MAP,
        # Utilities
        format_timestamp, get_status_counts,
        # Claude args parsing
        resolve_claude_args,
    )
    # Import from cli module (functions and data)
    from .cli import (
        # Commands
        cmd_launch, cmd_start, cmd_stop, cmd_reset, cmd_send,
        # Instance operations
        get_instance_status, parse_log_messages, load_all_positions, should_show_in_watch,
        # Path utilities
        hcom_path,
        # Configuration
        get_config, HcomConfig, reload_config,
        ConfigSnapshot, load_config_snapshot, save_config,
        dict_to_hcom_config, HcomConfigError,
        # Utilities
        ensure_hcom_directories, list_available_agents,
    )
except ImportError as e:
    sys.stderr.write(f"Error: Cannot import required modules.\n")
    sys.stderr.write(f"Make sure hcom package is installed.\n")
    sys.stderr.write(f"Details: {e}\n")
    sys.exit(1)

IS_WINDOWS = os.name == 'nt'

# All ANSI codes and STATUS configs now imported from shared.py
# Only need ANSI regex for local use
ANSI_RE = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')

# UI-specific colors (not in shared.py)
FG_CLAUDE_ORANGE = '\033[38;5;214m'  # Light orange for Claude section
FG_CUSTOM_ENV = '\033[38;5;141m'  # Light purple for Custom Env section

# Parse config defaults from shared.py
CONFIG_DEFAULTS = {}
for line in DEFAULT_CONFIG_DEFAULTS:
    if '=' in line:
        key, value = line.split('=', 1)
        value = value.strip()
        # Remove only outer layer of quotes (preserve inner quotes for HCOM_CLAUDE_ARGS)
        if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
            value = value[1:-1]
        CONFIG_DEFAULTS[key.strip()] = value

# TUI Layout Constants
MESSAGE_PREVIEW_LIMIT = 100  # Keep last N messages in message preview
MAX_INPUT_ROWS = 8  # Cap input area at N rows

# Config field special handlers (automatic for all, enhanced UX for specific vars)
CONFIG_FIELD_OVERRIDES = {
    'HCOM_TIMEOUT': {
        'type': 'numeric',
        'min': 1,
        'max': 86400,
        'hint': '1-86400 seconds',
    },
    'HCOM_SUBAGENT_TIMEOUT': {
        'type': 'numeric',
        'min': 1,
        'max': 86400,
        'hint': '1-86400 seconds',
    },
    'HCOM_TERMINAL': {
        'type': 'text',
        'hint': 'new | here | "custom {script}"',
    },
    'HCOM_HINTS': {
        'type': 'text',
        'hint': 'text string',
    },
    'HCOM_TAG': {
        'type': 'text',
        'allowed_chars': r'^[a-zA-Z0-9-]*$',  # Only letters, numbers, hyphens (no spaces)
        'hint': 'letters/numbers/hyphens only',
    },
    'HCOM_AGENT': {
        'type': 'cycle',
        'options': lambda: list_available_agents(),  # Dynamic discovery
        'hint': '←→ cycle options',
    },
}


@dataclass
class Field:
    """Field representation for rendering expandable sections"""
    key: str
    display_name: str
    field_type: Literal['checkbox', 'text', 'cycle', 'numeric']
    value: str | bool
    options: List[str] | None = None
    hint: str = ""


class Mode(Enum):
    MANAGE = "manage"
    LAUNCH = "launch"


class LaunchField(Enum):
    COUNT = 0
    LAUNCH_BTN = 1
    CLAUDE_SECTION = 2
    HCOM_SECTION = 3
    CUSTOM_ENV_SECTION = 4
    OPEN_EDITOR = 5


def ansi_len(text: str) -> int:
    """Get visible length of text (excluding ANSI codes), accounting for wide chars"""
    import unicodedata
    visible = ANSI_RE.sub('', text)
    width = 0
    for char in visible:
        ea_width = unicodedata.east_asian_width(char)
        if ea_width in ('F', 'W'):  # Fullwidth or Wide
            width += 2
        elif ea_width in ('Na', 'H', 'N', 'A'):  # Narrow, Half-width, Neutral, Ambiguous
            width += 1
        # else: zero-width characters (combining marks, etc.)
    return width


def ansi_ljust(text: str, width: int) -> str:
    """Left-justify text to width, accounting for ANSI codes"""
    visible = ansi_len(text)
    return text + (' ' * (width - visible)) if visible < width else text


def bg_ljust(text: str, width: int, bg_color: str) -> str:
    """Left-justify text with background color padding"""
    visible = ansi_len(text)
    if visible < width:
        padding = ' ' * (width - visible)
        return f"{text}{bg_color}{padding}{RESET}"
    return text


def truncate_ansi(text: str, width: int) -> str:
    """Truncate text to width, preserving ANSI codes, accounting for wide chars"""
    import unicodedata
    if width <= 0:
        return ''
    visible_len = ansi_len(text)
    if visible_len <= width:
        return text

    visible = 0
    result = []
    i = 0
    target = width - 1  # Reserve space for ellipsis

    while i < len(text) and visible < target:
        if text[i] == '\033':
            match = ANSI_RE.match(text, i)
            if match:
                result.append(match.group())
                i = match.end()
                continue

        # Check character width
        char_width = 1
        ea_width = unicodedata.east_asian_width(text[i])
        if ea_width in ('F', 'W'):  # Fullwidth or Wide
            char_width = 2

        # Only add if it fits
        if visible + char_width <= target:
            result.append(text[i])
            visible += char_width
        else:
            break  # No more space
        i += 1

    result.append('…')
    result.append(RESET)
    return ''.join(result)


def smart_truncate_name(name: str, width: int) -> str:
    """
    Intelligently truncate name keeping prefix and suffix with middle ellipsis.
    Example: "bees_general-purpose_2" (21 chars) → "bees…pose_2" (11 chars)
    """
    if len(name) <= width:
        return name
    if width < 5:
        return name[:width]

    # Keep prefix and suffix, put ellipsis in middle
    # Reserve 1 char for ellipsis
    available = width - 1
    prefix_len = (available + 1) // 2  # Round up for prefix
    suffix_len = available - prefix_len

    return name[:prefix_len] + '…' + name[-suffix_len:] if suffix_len > 0 else name[:prefix_len] + '…'


class AnsiTextWrapper(textwrap.TextWrapper):
    """TextWrapper that handles ANSI escape codes correctly"""

    def _wrap_chunks(self, chunks):
        """Override to use visible length for width calculations"""
        lines = []
        if self.width <= 0:
            raise ValueError("invalid width %r (must be > 0)" % self.width)

        chunks.reverse()
        while chunks:
            cur_line = []
            cur_len = 0
            indent = self.subsequent_indent if lines else self.initial_indent
            width = self.width - ansi_len(indent)

            while chunks:
                l = ansi_len(chunks[-1])
                if cur_len + l <= width:
                    cur_line.append(chunks.pop())
                    cur_len += l
                else:
                    break

            if chunks and ansi_len(chunks[-1]) > width:
                if not cur_line:
                    cur_line.append(chunks.pop())

            if cur_line:
                lines.append(indent + ''.join(cur_line))

        return lines


def get_terminal_size() -> Tuple[int, int]:
    """Get terminal dimensions (cols, rows)"""
    size = shutil.get_terminal_size(fallback=(100, 30))
    return size.columns, size.lines


# format_age() imported from hcom.py (was duplicate at line 320-332)


class KeyboardInput:
    """Cross-platform keyboard input handler"""

    def __init__(self):
        self.is_windows = IS_WINDOWS
        if not self.is_windows:
            import termios
            import tty
            self.termios = termios
            self.tty = tty
            self.fd = sys.stdin.fileno()
            self.old_settings = None

    def __enter__(self):
        if not self.is_windows:
            try:
                self.old_settings = self.termios.tcgetattr(self.fd)
                self.tty.setcbreak(self.fd)
            except Exception:
                self.old_settings = None
        sys.stdout.write(HIDE_CURSOR)
        sys.stdout.flush()
        return self

    def __exit__(self, *args):
        if not self.is_windows and self.old_settings:
            self.termios.tcsetattr(self.fd, self.termios.TCSADRAIN, self.old_settings)
        sys.stdout.write(SHOW_CURSOR)
        sys.stdout.flush()

    def has_input(self) -> bool:
        """Check if input is available without blocking"""
        if self.is_windows:
            import msvcrt
            return msvcrt.kbhit()  # type: ignore[attr-defined]
        else:
            try:
                return bool(select.select([self.fd], [], [], 0.0)[0])
            except (InterruptedError, OSError):
                return False

    def get_key(self) -> Optional[str]:
        """Read single key press, return special key name or character"""
        if self.is_windows:
            import msvcrt
            if not msvcrt.kbhit():  # type: ignore[attr-defined]
                return None
            ch = msvcrt.getwch()  # type: ignore[attr-defined]
            if ch in ('\x00', '\xe0'):
                ch2 = msvcrt.getwch()  # type: ignore[attr-defined]
                keys = {'H': 'UP', 'P': 'DOWN', 'K': 'LEFT', 'M': 'RIGHT'}
                return keys.get(ch2, None)
            # Distinguish manual Enter from pasted newlines (Windows)
            if ch in ('\r', '\n'):
                # If more input is immediately available, it's likely a paste
                if msvcrt.kbhit():  # type: ignore[attr-defined]
                    return '\n'  # Pasted newline, keep as literal
                else:
                    return 'ENTER'  # Manual Enter key press
            if ch == '\x1b': return 'ESC'
            if ch in ('\x08', '\x7f'): return 'BACKSPACE'
            if ch == ' ': return 'SPACE'
            if ch == '\t': return 'TAB'
            return ch if ch else None
        else:
            try:
                has_data = select.select([self.fd], [], [], 0.0)[0]
            except (InterruptedError, OSError):
                return None
            if not has_data:
                return None
            try:
                ch = os.read(self.fd, 1).decode('utf-8', errors='ignore')
            except OSError:
                return None
            if ch == '\x1b':
                try:
                    has_escape_data = select.select([self.fd], [], [], 0.1)[0]
                except (InterruptedError, OSError):
                    return 'ESC'
                if has_escape_data:
                    try:
                        next1 = os.read(self.fd, 1).decode('utf-8', errors='ignore')
                        if next1 == '[':
                            next2 = os.read(self.fd, 1).decode('utf-8', errors='ignore')
                            keys = {'A': 'UP', 'B': 'DOWN', 'C': 'RIGHT', 'D': 'LEFT'}
                            if next2 in keys:
                                return keys[next2]
                    except (OSError, UnicodeDecodeError):
                        pass
                return 'ESC'
            # Distinguish manual Enter from pasted newlines
            if ch in ('\r', '\n'):
                # If more input is immediately available, it's likely a paste
                try:
                    has_paste_data = select.select([self.fd], [], [], 0.0)[0]
                except (InterruptedError, OSError):
                    return 'ENTER'
                if has_paste_data:
                    return '\n'  # Pasted newline, keep as literal
                else:
                    return 'ENTER'  # Manual Enter key press
            if ch in ('\x7f', '\x08'): return 'BACKSPACE'
            if ch == ' ': return 'SPACE'
            if ch == '\t': return 'TAB'
            if ch == '\x03': return 'CTRL_C'
            if ch == '\x04': return 'CTRL_D'
            if ch == '\x01': return 'CTRL_A'
            if ch == '\x12': return 'CTRL_R'
            return ch


# Text input helper functions (shared between MANAGE and LAUNCH)

def text_input_insert(buffer: str, cursor: int, text: str) -> tuple[str, int]:
    """Insert text at cursor position, return (new_buffer, new_cursor)"""
    new_buffer = buffer[:cursor] + text + buffer[cursor:]
    new_cursor = cursor + len(text)
    return new_buffer, new_cursor

def text_input_backspace(buffer: str, cursor: int) -> tuple[str, int]:
    """Delete char before cursor, return (new_buffer, new_cursor)"""
    if cursor > 0:
        new_buffer = buffer[:cursor-1] + buffer[cursor:]
        new_cursor = cursor - 1
        return new_buffer, new_cursor
    return buffer, cursor

def text_input_move_left(cursor: int) -> int:
    """Move cursor left, return new position"""
    return max(0, cursor - 1)

def text_input_move_right(buffer: str, cursor: int) -> int:
    """Move cursor right, return new position"""
    return min(len(buffer), cursor + 1)

def calculate_text_input_rows(text: str, width: int, max_rows: int = MAX_INPUT_ROWS) -> int:
    """Calculate rows needed for wrapped text with literal newlines"""
    if not text:
        return 1

    lines = text.split('\n')
    total_rows = 0
    for line in lines:
        if not line:
            total_rows += 1
        else:
            total_rows += max(1, (len(line) + width - 1) // width)
    return min(total_rows, max_rows)


def render_text_input(buffer: str, cursor: int, width: int, max_rows: int, prefix: str = "> ") -> List[str]:
    """
    Render text input with cursor, wrapping, and literal newlines.

    Args:
        buffer: Text content
        cursor: Cursor position (0 to len(buffer))
        width: Terminal width
        max_rows: Maximum rows to render
        prefix: First line prefix (e.g., "> " or "")

    Returns:
        List of formatted lines with cursor (█)
    """
    if not buffer:
        return [f"{FG_GRAY}{prefix}█{RESET}"]

    line_width = width - len(prefix)
    before = buffer[:cursor]
    after = buffer[cursor:]
    full = before + '█' + after

    # Split on literal newlines first
    lines = full.split('\n')

    # Wrap each line if needed
    wrapped = []
    for line_idx, line in enumerate(lines):
        if not line:
            # Empty line (from consecutive newlines or trailing newline)
            line_prefix = prefix if line_idx == 0 else " " * len(prefix)
            wrapped.append(f"{FG_WHITE}{line_prefix}{RESET}")
        else:
            # Wrap long lines
            for chunk_idx in range(0, len(line), line_width):
                chunk = line[chunk_idx:chunk_idx+line_width]
                line_prefix = prefix if line_idx == 0 and chunk_idx == 0 else " " * len(prefix)
                wrapped.append(f"{FG_WHITE}{line_prefix}{RESET}{FG_WHITE}{chunk}{RESET}")

    # Pad or truncate to max_rows
    result = wrapped + [''] * max(0, max_rows - len(wrapped))
    return result[:max_rows]


def ease_out_quad(t: float) -> float:
    """Ease-out quadratic curve (fast start, slow end)"""
    return 1 - (1 - t) ** 2


def interpolate_color_index(start: int, end: int, progress: float) -> int:
    """Interpolate between two 256-color palette indices with ease-out

    Args:
        start: Starting color index (0-255)
        end: Ending color index (0-255)
        progress: Progress from 0.0 to 1.0

    Returns:
        Interpolated color index (0-255)
    """
    # Clamp progress to [0, 1]
    progress = max(0.0, min(1.0, progress))

    # Apply ease-out curve (50% fade in first 10s)
    eased = ease_out_quad(progress)

    # Linear interpolation between indices
    return int(start + (end - start) * eased)


def get_message_pulse_colors(seconds_since: float) -> tuple[str, str]:
    """Get background and foreground colors for LOG tab based on message recency

    Args:
        seconds_since: Seconds since last message (0 = just now, 5+ = quiet)

    Returns:
        (bg_color, fg_color) tuple of ANSI escape codes
    """
    # At rest (5s+), use exact same colors as LAUNCH tab
    if seconds_since >= 5.0:
        return BG_CHARCOAL, FG_WHITE

    # Clamp to 5s range
    seconds = max(0.0, min(5.0, seconds_since))

    # Progress: 0.0 = recent (white), 1.0 = quiet (charcoal)
    progress = seconds / 5.0

    # Interpolate background: 255 (white) → 236 (charcoal)
    bg_index = interpolate_color_index(255, 236, progress)

    # Interpolate foreground: 232 (darkest gray) → 250 (light gray matching FG_WHITE)
    # Don't overshoot to 255 (pure white) - normal FG_WHITE is dimmer
    fg_index = interpolate_color_index(232, 250, progress)

    return f'\033[48;5;{bg_index}m', f'\033[38;5;{fg_index}m'


class HcomTUI:
    """Main TUI application"""

    # Confirmation timeout constants
    CONFIRMATION_TIMEOUT = 10.0  # State cleared after this
    CONFIRMATION_FLASH_DURATION = 10.0  # Flash duration matches timeout

    def __init__(self, hcom_dir: Path):
        self.hcom_dir = hcom_dir
        self.mode = Mode.MANAGE

        # State
        self.cursor = 0  # Current selection in lists
        self.cursor_instance_name: Optional[str] = None  # Stable cursor tracking by name
        self.instances = {}  # {name: {status, age_text, data}}
        self.status_counts = {s: 0 for s in STATUS_ORDER}
        self.messages = []  # [(timestamp, sender, message)] - Recent messages for preview
        self.message_buffer: str = ""  # Message input buffer
        self.message_cursor_pos: int = 0  # Cursor position in message buffer

        # Toggle confirmation state (two-step)
        self.pending_toggle: Optional[str] = None  # Instance name pending confirmation
        self.pending_toggle_time: float = 0.0      # When confirmation started

        # Toggle completion state (temporary display)
        self.completed_toggle: Optional[str] = None  # Instance that just completed
        self.completed_toggle_time: float = 0.0      # When it completed

        # Stop all confirmation state
        self.pending_stop_all: bool = False
        self.pending_stop_all_time: float = 0.0

        # Reset confirmation state
        self.pending_reset: bool = False
        self.pending_reset_time: float = 0.0

        # Instance scrolling
        self.instance_scroll_pos: int = 0  # Top visible instance index

        # Launch screen scrolling
        self.launch_scroll_pos: int = 0  # Top visible form line

        # Flash notifications
        self.flash_message = None
        self.flash_until = 0.0
        self.flash_color = 'orange'  # 'red', 'white', or 'orange'

        # Validation errors
        self.validation_errors = {}  # {field_key: error_message}

        # Launch form state
        self.launch_count = "1"
        self.launch_prompt = ""
        self.launch_system_prompt = ""
        self.launch_background = False
        self.launch_field = LaunchField.COUNT  # Currently selected field
        self.available_agents = []  # List of available agents from .claude/agents

        # Cursor positions for editable fields (bottom bar editor)
        self.launch_prompt_cursor = 0
        self.launch_system_prompt_cursor = 0
        self.config_field_cursors = {}  # {field_key: cursor_pos} for config.env fields

        # Dynamic config state (loaded from ~/.hcom/config.env)
        self.config_snapshot: ConfigSnapshot | None = None
        self.config_edit: dict[str, str] = {}  # Combined HCOM_* and extra env vars

        # Section expansion state (UI only)
        self.claude_expanded = False
        self.hcom_expanded = False
        self.custom_env_expanded = False

        # Section cursors (-1 = on header, 0+ = field index)
        self.claude_cursor = -1
        self.hcom_cursor = -1
        self.custom_env_cursor = -1

        # Rendering
        self.last_frame = []
        self.last_status_update = 0.0
        self.first_render = True

        # File size caching for efficient log loading
        self.last_log_size = 0

        # Message activity tracking for LOG tab pulse
        self.last_message_time: float = 0.0  # Timestamp of most recent message

    def flash(self, msg: str, duration: float = 2.0, color: str = 'orange'):
        """Show temporary flash message

        Args:
            msg: Message text
            duration: Display time in seconds
            color: 'red', 'white', or 'orange' (default)
        """
        self.flash_message = msg
        self.flash_until = time.time() + duration
        self.flash_color = color

    def flash_error(self, msg: str, duration: float = 10.0):
        """Show error flash in red"""
        self.flash_message = msg
        self.flash_until = time.time() + duration
        self.flash_color = 'red'

    def parse_validation_errors(self, error_str: str):
        """Parse ValueError message from HcomConfig into field-specific errors"""
        self.validation_errors.clear()

        # Parse multi-line error format:
        # "Invalid config:\n  - timeout must be...\n  - terminal cannot..."
        for line in error_str.split('\n'):
            line = line.strip()
            if not line or line == 'Invalid config:':
                continue

            # Remove leading "- " from error lines
            if line.startswith('- '):
                line = line[2:]

            # Match error to field based on keywords
            # For fields with multiple possible errors, only store first error seen
            line_lower = line.lower()
            if 'timeout must be' in line_lower and 'subagent' not in line_lower:
                if 'HCOM_TIMEOUT' not in self.validation_errors:
                    self.validation_errors['HCOM_TIMEOUT'] = line
            elif 'subagent_timeout' in line_lower or 'subagent timeout' in line_lower:
                if 'HCOM_SUBAGENT_TIMEOUT' not in self.validation_errors:
                    self.validation_errors['HCOM_SUBAGENT_TIMEOUT'] = line
            elif 'terminal' in line_lower:
                if 'HCOM_TERMINAL' not in self.validation_errors:
                    self.validation_errors['HCOM_TERMINAL'] = line
            elif 'tag' in line_lower:
                if 'HCOM_TAG' not in self.validation_errors:
                    self.validation_errors['HCOM_TAG'] = line
            elif 'agent' in line_lower and 'subagent' not in line_lower:
                # Agent can have multiple errors - store first one
                if 'HCOM_AGENT' not in self.validation_errors:
                    self.validation_errors['HCOM_AGENT'] = line
            elif 'claude_args' in line_lower:
                if 'HCOM_CLAUDE_ARGS' not in self.validation_errors:
                    self.validation_errors['HCOM_CLAUDE_ARGS'] = line
            elif 'hints' in line_lower:
                if 'HCOM_HINTS' not in self.validation_errors:
                    self.validation_errors['HCOM_HINTS'] = line

    def clear_all_pending_confirmations(self):
        """Clear all pending confirmation states and flash if any were active"""
        had_pending = self.pending_toggle or self.pending_stop_all or self.pending_reset

        self.pending_toggle = None
        self.pending_stop_all = False
        self.pending_reset = False

        if had_pending:
            self.flash_message = None

    def clear_pending_confirmations_except(self, keep: str):
        """Clear all pending confirmations except the specified one ('toggle', 'stop_all', 'reset')"""
        had_pending = False

        if keep != 'toggle' and self.pending_toggle:
            self.pending_toggle = None
            had_pending = True
        if keep != 'stop_all' and self.pending_stop_all:
            self.pending_stop_all = False
            had_pending = True
        if keep != 'reset' and self.pending_reset:
            self.pending_reset = False
            had_pending = True

        if had_pending:
            self.flash_message = None

    def calculate_manage_layout(self, height: int, width: int) -> tuple[int, int, int]:
        """Calculate instance and message rows for MANAGE screen layout"""
        # Calculate input rows based on buffer length (auto-wrap + literal newlines)
        line_width = width - 2  # Account for "> " prefix
        input_rows = calculate_text_input_rows(self.message_buffer, line_width)

        separator_rows = 3  # One separator between instances and messages, one before input, one after input
        min_instance_rows = 3
        min_message_rows = 3

        available = height - input_rows - separator_rows
        # Instance rows = num instances (capped at 60% of available)
        instance_count = len(self.instances)
        max_instance_rows = int(available * 0.6)
        instance_rows = max(min_instance_rows, min(instance_count, max_instance_rows))
        message_rows = available - instance_rows

        return instance_rows, message_rows, input_rows

    def render_wrapped_input(self, width: int, input_rows: int) -> List[str]:
        """Render message input (delegates to shared helper)"""
        return render_text_input(
            self.message_buffer,
            self.message_cursor_pos,
            width,
            input_rows,
            prefix="> "
        )

    def sync_scroll_to_cursor(self):
        """Sync instance scroll position to keep cursor visible"""
        # Calculate visible rows using shared layout function
        width, rows = get_terminal_size()
        body_height = max(10, rows - 3)  # Header, flash, footer
        instance_rows, _, _ = self.calculate_manage_layout(body_height, width)
        visible_instance_rows = instance_rows  # Full instance section is visible

        # Scroll up if cursor moved above visible window
        if self.cursor < self.instance_scroll_pos:
            self.instance_scroll_pos = self.cursor
        # Scroll down if cursor moved below visible window
        elif self.cursor >= self.instance_scroll_pos + visible_instance_rows:
            self.instance_scroll_pos = self.cursor - visible_instance_rows + 1

    def stop_all_instances(self):
        """Stop all enabled instances"""
        try:
            stopped_count = 0
            for name, info in self.instances.items():
                if info['data'].get('enabled', False):
                    cmd_stop([name])
                    stopped_count += 1

            if stopped_count > 0:
                self.flash(f"Stopped all ({stopped_count} instances)")
            else:
                self.flash("No instances to stop")

            self.load_status()
        except Exception as e:
            self.flash_error(f"Error: {str(e)}")

    def reset_logs(self):
        """Reset logs (archive and clear)"""
        try:
            cmd_reset(['logs'])
            # Reload to clear instance list from display
            self.load_status()
            archive_path = f"{Path.home()}/.hcom/archive/"
            self.flash(f"Logs and instance list archived to {archive_path}", duration=10.0)
        except Exception as e:
            self.flash_error(f"Error: {str(e)}")

    def run(self) -> int:
        """Main event loop"""
        # Initialize
        ensure_hcom_directories()

        # Load saved states (config.env first, then launch state reads from it)
        self.load_config_from_file()
        self.load_launch_state()

        # Enter alternate screen
        sys.stdout.write('\033[?1049h')
        sys.stdout.flush()

        try:
            with KeyboardInput() as kbd:
                while True:
                    # Only update/render if no pending input (paste optimization)
                    if not kbd.has_input():
                        self.update()
                        self.render()
                        time.sleep(0.01)  # Only sleep when idle

                    key = kbd.get_key()
                    if not key:
                        time.sleep(0.01)  # Also sleep when no key available
                        continue

                    if key == 'CTRL_D':
                        # Save state before exit
                        self.save_launch_state()
                        break
                    elif key == 'TAB':
                        # Save state when switching modes
                        if self.mode == Mode.LAUNCH:
                            self.save_launch_state()
                        self.handle_tab()
                    else:
                        self.handle_key(key)

            return 0
        except KeyboardInterrupt:
            # Ctrl+C - clean exit
            self.save_launch_state()
            return 0
        except Exception as e:
            sys.stderr.write(f"Error: {e}\n")
            return 1
        finally:
            # Exit alternate screen
            sys.stdout.write('\033[?1049l')
            sys.stdout.flush()

    def load_status(self):
        """Load instance status from ~/.hcom/instances/"""
        all_instances = load_all_positions()

        # Filter using same logic as watch
        instances = {
            name: data for name, data in all_instances.items()
            if should_show_in_watch(data)
        }

        # Build instance info dict (replace old instances, don't just add)
        new_instances = {}
        for name, data in instances.items():
            enabled, status_type, age_text, description = get_instance_status(data)

            last_time = data.get('status_time', 0.0)
            age_seconds = time.time() - last_time if last_time > 0 else 999.0

            new_instances[name] = {
                'enabled': enabled,
                'status': status_type,
                'age_text': age_text,
                'description': description,
                'age_seconds': age_seconds,
                'last_time': last_time,
                'data': data,
            }

        self.instances = new_instances
        self.status_counts = get_status_counts(self.instances)

    def save_launch_state(self):
        """Save launch form values to config.env via claude args parser"""
        # Phase 3: Save Claude args to HCOM_CLAUDE_ARGS in config.env
        try:
            # Load current spec
            claude_args_str = self.config_edit.get('HCOM_CLAUDE_ARGS', '')
            spec = resolve_claude_args(None, claude_args_str if claude_args_str else None)

            # System flag matches background mode
            system_flag = None
            system_value = None
            if self.launch_system_prompt:
                system_flag = "--system-prompt" if self.launch_background else "--append-system-prompt"
                system_value = self.launch_system_prompt
            else:
                system_value = ""

            # Update spec with form values
            spec = spec.update(
                background=self.launch_background,
                system_flag=system_flag,
                system_value=system_value,
                prompt=self.launch_prompt,  # Always pass value (empty string deletes)
            )

            # Persist to in-memory edits
            self.config_edit['HCOM_CLAUDE_ARGS'] = spec.to_env_string()

            # Write config.env
            # Note: HCOM_TAG and HCOM_AGENT are already saved directly when edited in UI
            self.save_config_to_file()
        except Exception as e:
            # Don't crash on save failure, but log to stderr
            sys.stderr.write(f"Warning: Failed to save launch state: {e}\n")

    def load_launch_state(self):
        """Load launch form values from config.env via claude args parser"""
        # Phase 3: Load Claude args from HCOM_CLAUDE_ARGS in config.env
        try:
            claude_args_str = self.config_edit.get('HCOM_CLAUDE_ARGS', '')
            spec = resolve_claude_args(None, claude_args_str if claude_args_str else None)

            # Check for parse errors and surface them
            if spec.errors:
                self.flash_error(f"Parse error: {spec.errors[0]}")

            # Extract Claude-related fields from spec
            self.launch_background = spec.is_background
            self.launch_prompt = spec.positional_tokens[0] if spec.positional_tokens else ""

            # Extract system prompt (prefer user_system, fallback to user_append)
            if spec.user_system:
                self.launch_system_prompt = spec.user_system
            elif spec.user_append:
                self.launch_system_prompt = spec.user_append
            else:
                self.launch_system_prompt = ""

            # Initialize cursors to end of text for first-time navigation
            self.launch_prompt_cursor = len(self.launch_prompt)
            self.launch_system_prompt_cursor = len(self.launch_system_prompt)
        except Exception as e:
            # Failed to parse - use defaults and log warning
            sys.stderr.write(f"Warning: Failed to load launch state (using defaults): {e}\n")

    def load_config_from_file(self, *, raise_on_error: bool = False):
        """Load all vars from ~/.hcom/config.env into editable dict"""
        config_path = Path.home() / '.hcom' / 'config.env'
        try:
            snapshot = load_config_snapshot()
            self.config_snapshot = snapshot
            combined: dict[str, str] = {}
            combined.update(snapshot.values)
            combined.update(snapshot.extras)
            self.config_edit = combined
            self.validation_errors.clear()
            # Track mtime for external change detection
            try:
                self.config_mtime = config_path.stat().st_mtime
            except FileNotFoundError:
                self.config_mtime = 0.0
        except Exception as e:
            if raise_on_error:
                raise
            sys.stderr.write(f"Warning: Failed to load config.env (using defaults): {e}\n")
            self.config_snapshot = None
            self.config_edit = dict(CONFIG_DEFAULTS)
            for line in DEFAULT_CONFIG_HEADER:
                stripped = line.strip()
                if stripped and not stripped.startswith('#') and '=' in line:
                    key, _, value = line.partition('=')
                    key = key.strip()
                    raw = value.strip()
                    if (raw.startswith('"') and raw.endswith('"')) or (raw.startswith("'") and raw.endswith("'")):
                        raw = raw[1:-1]
                    self.config_edit.setdefault(key, raw)
            self.config_mtime = 0.0

    def save_config_to_file(self):
        """Write current config edits back to ~/.hcom/config.env using canonical writer."""
        known_values = {key: self.config_edit.get(key, '') for key in CONFIG_DEFAULTS.keys()}
        extras = {
            key: value
            for key, value in self.config_edit.items()
            if key not in CONFIG_DEFAULTS
        }

        field_map = {
            'timeout': 'HCOM_TIMEOUT',
            'subagent_timeout': 'HCOM_SUBAGENT_TIMEOUT',
            'terminal': 'HCOM_TERMINAL',
            'tag': 'HCOM_TAG',
            'agent': 'HCOM_AGENT',
            'claude_args': 'HCOM_CLAUDE_ARGS',
            'hints': 'HCOM_HINTS',
        }

        try:
            core = dict_to_hcom_config(known_values)
        except HcomConfigError as exc:
            self.validation_errors.clear()
            for field, message in exc.errors.items():
                env_key = field_map.get(field, field.upper())
                self.validation_errors[env_key] = message
            first_error = next(iter(self.validation_errors.values()), "Invalid config")
            self.flash_error(first_error)
            return
        except Exception as exc:
            self.flash_error(f"Validation error: {exc}")
            return

        try:
            save_config(core, extras)
            self.validation_errors.clear()
            self.flash_message = None
            # Reload snapshot to pick up canonical formatting
            self.load_config_from_file()
            self.load_launch_state()
        except Exception as exc:
            self.flash_error(f"Save failed: {exc}")

    def check_external_config_changes(self):
        """Reload config.env if changed on disk, preserving active edits."""
        config_path = Path.home() / '.hcom' / 'config.env'
        try:
            mtime = config_path.stat().st_mtime
        except FileNotFoundError:
            return

        if mtime <= self.config_mtime:
            return  # No change

        # Save what's currently being edited
        active_field = self.get_current_launch_field_info()

        # Backup current edits
        old_edit = dict(self.config_edit)

        # Reload from disk
        try:
            self.load_config_from_file()
            self.load_launch_state()
        except Exception as exc:
            self.flash_error(f"Failed to reload config.env: {exc}")
            return

        # Update mtime
        try:
            self.config_mtime = config_path.stat().st_mtime
        except FileNotFoundError:
            self.config_mtime = 0.0

        # Restore in-progress edit if field changed externally
        if active_field and active_field[0]:
            key, value, cursor = active_field
            # Check if the field we're editing changed externally
            if key in old_edit and old_edit.get(key) != self.config_edit.get(key):
                # External change to field you're editing - keep your version
                self.config_edit[key] = value
                if key in self.config_field_cursors:
                    self.config_field_cursors[key] = cursor
                self.flash(f"Kept in-progress {key} edit (external change ignored)")

    def resolve_editor_command(self) -> tuple[list[str] | None, str | None]:
        """Resolve preferred editor command and display label for config edits."""
        config_path = Path.home() / '.hcom' / 'config.env'
        editor = os.environ.get('VISUAL') or os.environ.get('EDITOR')
        pretty_names = {
            'code': 'VS Code',
            'code-insiders': 'VS Code Insiders',
            'hx': 'Helix',
            'helix': 'Helix',
            'nvim': 'Neovim',
            'vim': 'Vim',
            'nano': 'nano',
        }

        if editor:
            try:
                parts = shlex.split(editor)
            except ValueError:
                parts = []
            if parts:
                command = parts[0]
                base_name = Path(command).name or command
                normalized = base_name.lower()
                if normalized.endswith('.exe'):
                    normalized = normalized[:-4]
                label = pretty_names.get(normalized, base_name)
                return parts + [str(config_path)], label

        if code_bin := shutil.which('code'):
            return [code_bin, str(config_path)], 'VS Code'
        if nano_bin := shutil.which('nano'):
            return [nano_bin, str(config_path)], 'nano'
        if vim_bin := shutil.which('vim'):
            return [vim_bin, str(config_path)], 'vim'
        return None, None

    def open_config_in_editor(self):
        """Open config.env in the resolved editor."""
        cmd, label = self.resolve_editor_command()
        if not cmd:
            self.flash_error("No external editor found")
            return

        # Ensure latest in-memory edits are persisted before handing off
        self.save_config_to_file()

        try:
            subprocess.Popen(cmd)
            self.flash(f"Opening config.env in {label or 'VS Code'}...")
        except Exception as exc:
            self.flash_error(f"Failed to launch {label or 'editor'}: {exc}")


    def update(self):
        """Update state (status, messages)"""
        now = time.time()

        # Update status every 0.5 seconds
        if now - self.last_status_update >= 0.5:
            self.load_status()
            self.last_status_update = now

        # Clear pending toggle after timeout
        if self.pending_toggle and (now - self.pending_toggle_time) > self.CONFIRMATION_TIMEOUT:
            self.pending_toggle = None

        # Clear completed toggle display after 2s (match flash default)
        if self.completed_toggle and (now - self.completed_toggle_time) >= 2.0:
            self.completed_toggle = None

        # Clear pending stop all after timeout
        if self.pending_stop_all and (now - self.pending_stop_all_time) > self.CONFIRMATION_TIMEOUT:
            self.pending_stop_all = False

        # Clear pending reset after timeout
        if self.pending_reset and (now - self.pending_reset_time) > self.CONFIRMATION_TIMEOUT:
            self.pending_reset = False

        # Load available agents if on LAUNCH screen
        if self.mode == Mode.LAUNCH and not self.available_agents:
            self.available_agents = list_available_agents()

        # Periodic config reload check (only in Launch mode)
        if self.mode == Mode.LAUNCH:
            if not hasattr(self, 'last_config_check'):
                self.last_config_check = 0.0
            if not hasattr(self, 'config_mtime'):
                self.config_mtime = 0.0

            if (now - self.last_config_check) >= 0.5:
                self.last_config_check = now
                self.check_external_config_changes()

        # Load messages for MANAGE screen preview (with file size caching)
        if self.mode == Mode.MANAGE:
            log_file = self.hcom_dir / 'hcom.log'
            if log_file.exists():
                current_size = log_file.stat().st_size
                # Only re-parse if file size changed (handles truncation too)
                if current_size != self.last_log_size:
                    try:
                        result = parse_log_messages(log_file)
                        if result and hasattr(result, 'messages') and result.messages:
                            # Convert from dict format to tuple format (time, sender, message)
                            all_messages = [
                                (msg.get('timestamp', ''), msg.get('from', ''), msg.get('message', ''))
                                for msg in result.messages
                                if isinstance(msg, dict) and 'from' in msg and 'message' in msg
                            ]
                            # Update preview (last N)
                            self.messages = all_messages[-MESSAGE_PREVIEW_LIMIT:] if len(all_messages) > MESSAGE_PREVIEW_LIMIT else all_messages
                            # Update last message time for LOG tab pulse
                            if all_messages:
                                last_msg_timestamp = all_messages[-1][0]  # timestamp string
                                try:
                                    from datetime import datetime
                                    if 'T' in last_msg_timestamp:
                                        dt = datetime.fromisoformat(last_msg_timestamp.replace('Z', '+00:00'))
                                        self.last_message_time = dt.timestamp()
                                except Exception:
                                    pass  # Keep previous time if parse fails
                        else:
                            self.messages = []
                            self.last_message_time = 0.0
                        self.last_log_size = current_size
                    except Exception:
                        # Parse failed - keep existing messages
                        pass
                elif current_size == 0:
                    # File is empty - clear messages
                    self.messages = []
                    self.last_message_time = 0.0
                    self.last_log_size = 0
            else:
                # File doesn't exist - clear messages
                self.messages = []
                self.last_message_time = 0.0
                self.last_log_size = 0

    def build_status_bar(self, highlight_tab: str | None = None) -> str:
        """Build status bar with tabs - shared by TUI header and native log view
        Args:
            highlight_tab: Which tab to highlight ("MANAGE", "LAUNCH", or "LOG")
                          If None, uses self.mode
        """
        # Determine which tab to highlight
        if highlight_tab is None:
            highlight_tab = self.mode.value.upper()

        # Calculate message pulse colors for LOG tab
        if self.last_message_time > 0:
            seconds_since_msg = time.time() - self.last_message_time
        else:
            seconds_since_msg = 9999.0  # No messages yet - use quiet state
        log_bg_color, log_fg_color = get_message_pulse_colors(seconds_since_msg)

        # Build status display (colored blocks for unselected, orange for selected)
        is_manage_selected = (highlight_tab == "MANAGE")
        status_parts = []

        # Use shared status configuration (background colors for statusline blocks)
        for status_type in STATUS_ORDER:
            count = self.status_counts.get(status_type, 0)
            if count > 0:
                color, symbol = STATUS_BG_MAP[status_type]
                if is_manage_selected:
                    # Selected: orange bg + black text (v1 style)
                    part = f"{FG_BLACK}{BOLD}{BG_ORANGE} {count} {symbol} {RESET}"
                else:
                    # Unselected: colored blocks (hcom watch style)
                    text_color = FG_BLACK if color == BG_YELLOW else FG_WHITE
                    part = f"{text_color}{BOLD}{color} {count} {symbol} {RESET}"
                status_parts.append(part)

        # No instances - use orange if selected, charcoal if not
        if status_parts:
            status_display = "".join(status_parts)
        elif is_manage_selected:
            status_display = f"{FG_BLACK}{BOLD}{BG_ORANGE}  0  {RESET}"
        else:
            status_display = f"{BG_CHARCOAL}{FG_WHITE}  0  {RESET}"

        # Build tabs: MANAGE, LAUNCH, and LOG (LOG only shown in native view)
        tab_names = ["MANAGE", "LAUNCH", "LOG"]
        tabs = []

        for tab_name in tab_names:
            # MANAGE tab shows status counts instead of text
            if tab_name == "MANAGE":
                label = status_display
            else:
                label = tab_name

            # Highlight current tab (non-MANAGE tabs get orange bg)
            if tab_name == highlight_tab and tab_name != "MANAGE":
                # Selected tab: always orange bg + black fg (LOG and LAUNCH same)
                tabs.append(f"{BG_ORANGE}{FG_BLACK}{BOLD} {label} {RESET}")
            elif tab_name == "MANAGE":
                # MANAGE tab is just status blocks (already has color/bg)
                tabs.append(f" {label}")
            elif tab_name == "LOG":
                # LOG tab when not selected: use pulse colors (white→charcoal fade)
                tabs.append(f"{log_bg_color}{log_fg_color} {label} {RESET}")
            else:
                # LAUNCH when not selected: charcoal bg (milder than black)
                tabs.append(f"{BG_CHARCOAL}{FG_WHITE} {label} {RESET}")

        tab_display = " ".join(tabs)

        return f"{BOLD}hcom{RESET} {tab_display}"

    def build_flash(self) -> Optional[str]:
        """Build flash notification if active"""
        if self.flash_message and time.time() < self.flash_until:
            color_map = {
                'red': FG_RED,
                'white': FG_WHITE,
                'orange': FG_ORANGE
            }
            color_code = color_map.get(self.flash_color, FG_ORANGE)
            cols, _ = get_terminal_size()
            # Reserve space for "• " prefix and separator/padding
            max_msg_width = cols - 10
            msg = truncate_ansi(self.flash_message, max_msg_width) if len(self.flash_message) > max_msg_width else self.flash_message
            return f"{BOLD}{color_code}• {msg}{RESET}"
        return None

    def build_manage_screen(self, height: int, width: int) -> List[str]:
        """Build compact Manage screen"""
        # Use minimum height for layout calculation to maintain structure
        layout_height = max(10, height)

        lines = []

        # Calculate layout using shared function
        instance_rows, message_rows, input_rows = self.calculate_manage_layout(layout_height, width)

        # Sort instances by creation time (newest first) - stable, no jumping
        sorted_instances = sorted(
            self.instances.items(),
            key=lambda x: -x[1]['data'].get('created_at', 0.0)
        )

        total_instances = len(sorted_instances)

        # Restore cursor position by instance name (stable across sorts)
        if self.cursor_instance_name and sorted_instances:
            found = False
            for i, (name, _) in enumerate(sorted_instances):
                if name == self.cursor_instance_name:
                    self.cursor = i
                    found = True
                    break
            if not found:
                # Instance disappeared, reset cursor
                self.cursor = 0
                self.cursor_instance_name = None
                self.sync_scroll_to_cursor()

        # Ensure cursor is valid
        if sorted_instances:
            self.cursor = max(0, min(self.cursor, total_instances - 1))
            # Update tracked instance name
            if self.cursor < len(sorted_instances):
                self.cursor_instance_name = sorted_instances[self.cursor][0]
        else:
            self.cursor = 0
            self.cursor_instance_name = None

        # Empty state - no instances
        if total_instances == 0:
            lines.append('')
            lines.append(f"{FG_GRAY}No instances running{RESET}")
            lines.append('')
            lines.append(f"{FG_GRAY}Press Tab → LAUNCH to create instances{RESET}")
            # Pad to instance_rows
            while len(lines) < instance_rows:
                lines.append('')
            # Skip to message section
            lines.append(f"{FG_GRAY}{'─' * width}{RESET}")
            # No messages to show either
            lines.append(f"{FG_GRAY}(no messages){RESET}")
            while len(lines) < instance_rows + message_rows + 1:
                lines.append('')
            lines.append(f"{FG_GRAY}{'─' * width}{RESET}")
            # Input area (auto-wrapped)
            input_lines = self.render_wrapped_input(width, input_rows)
            lines.extend(input_lines)
            # Separator after input (before footer)
            lines.append(f"{FG_GRAY}{'─' * width}{RESET}")
            while len(lines) < height:
                lines.append('')
            return lines[:height]

        # Calculate visible window
        max_scroll = max(0, total_instances - instance_rows)
        self.instance_scroll_pos = max(0, min(self.instance_scroll_pos, max_scroll))

        visible_start = self.instance_scroll_pos
        visible_end = min(visible_start + instance_rows, total_instances)
        visible_instances = sorted_instances[visible_start:visible_end]

        # Render instances - compact one-line format
        for i, (name, info) in enumerate(visible_instances):
            absolute_idx = visible_start + i

            enabled = info.get('enabled', False)
            status = info.get('status', "unknown")
            _, icon = STATUS_MAP.get(status, (BG_GRAY, '?'))
            color = STATUS_FG.get(status, FG_WHITE)

            # Always show description if non-empty
            display_text = info.get('description', '')

            # Use age_text from get_instance_status (clean format: "16m", no parens)
            age_text = info.get('age_text', '')
            age_str = f"{age_text} ago" if age_text else ""
            # Right-align age in fixed width column (e.g., "  16m ago")
            age_width = 10
            age_padded = age_str.rjust(age_width)

            # Background indicator - include in name before padding
            is_background = info.get('data', {}).get('background', False)
            bg_marker_text = " [headless]" if is_background else ""
            bg_marker_visible_len = 11 if is_background else 0  # " [headless]" = 11 chars

            # Timeout warning indicator
            timeout_marker = ""
            if enabled and status == "waiting":
                age_seconds = info.get('age_seconds', 0)
                data = info.get('data', {})
                is_subagent = bool(data.get('parent_session_id'))

                if is_subagent:
                    timeout = get_config().subagent_timeout
                    remaining = timeout - age_seconds
                    if 0 < remaining < 10:
                        timeout_marker = f" {FG_YELLOW}⏱ {int(remaining)}s{RESET}"
                else:
                    timeout = data.get('wait_timeout', get_config().timeout)
                    remaining = timeout - age_seconds
                    if 0 < remaining < 60:
                        timeout_marker = f" {FG_YELLOW}⏱ {int(remaining)}s{RESET}"

            # Smart truncate name to fit in 24 chars including [headless] and state symbol
            # Available: 24 - bg_marker_len - (2 for " +/-" on cursor row)
            max_name_len = 22 - bg_marker_visible_len  # Leave 2 chars for " +" or " -"
            display_name = smart_truncate_name(name, max_name_len)

            # State indicator (only on cursor row)
            if absolute_idx == self.cursor:
                is_pending = self.pending_toggle == name and (time.time() - self.pending_toggle_time) <= self.CONFIRMATION_TIMEOUT
                if is_pending:
                    state_symbol = "±"
                    state_color = FG_GOLD
                elif enabled:
                    state_symbol = "+"
                    state_color = color
                else:
                    state_symbol = "-"
                    state_color = color
                # Format: name [headless] +/- (total 24 chars)
                name_with_marker = f"{display_name}{bg_marker_text} {state_symbol}"
                name_padded = ansi_ljust(name_with_marker, 24)
            else:
                # Format: name [headless] (total 24 chars)
                name_with_marker = f"{display_name}{bg_marker_text}"
                name_padded = ansi_ljust(name_with_marker, 24)

            # Description separator - only show if description exists
            desc_sep = ": " if display_text else ""

            # Bold if enabled, dim if disabled
            weight = BOLD if enabled else DIM

            if absolute_idx == self.cursor:
                # Highlighted row - Format: icon name [headless] +/-  age ago: description [timeout]
                line = f"{BG_CHARCOAL}{color}{icon} {weight}{color}{name_padded}{RESET}{BG_CHARCOAL}{weight}{FG_GRAY}{age_padded}{desc_sep}{display_text}{timeout_marker}{RESET}"
                line = truncate_ansi(line, width)
                line = bg_ljust(line, width, BG_CHARCOAL)
            else:
                # Normal row - Format: icon name [headless]  age ago: description [timeout]
                line = f"{color}{icon}{RESET} {weight}{color}{name_padded}{RESET}{weight}{FG_GRAY}{age_padded}{desc_sep}{display_text}{timeout_marker}{RESET}"
                line = truncate_ansi(line, width)

            lines.append(line)

        # Add scroll indicators if needed (indicator stays at edge, cursor moves if conflict)
        if total_instances > instance_rows:
            # If cursor will conflict with indicator, move cursor line first
            if visible_start > 0 and self.cursor == visible_start:
                # Save cursor line (at position 0), move to position 1
                cursor_line = lines[0]
                lines[0] = lines[1] if len(lines) > 1 else ""
                if len(lines) > 1:
                    lines[1] = cursor_line

            if visible_end < total_instances and self.cursor == visible_end - 1:
                # Save cursor line (at position -1), move to position -2
                cursor_line = lines[-1]
                lines[-1] = lines[-2] if len(lines) > 1 else ""
                if len(lines) > 1:
                    lines[-2] = cursor_line

            # Now add indicators at edges (may overwrite moved content, that's fine)
            if visible_start > 0:
                count_above = visible_start
                indicator = f"{FG_GRAY}↑ {count_above} more{RESET}"
                lines[0] = ansi_ljust(indicator, width)

            if visible_end < total_instances:
                count_below = total_instances - visible_end
                indicator = f"{FG_GRAY}↓ {count_below} more{RESET}"
                lines[-1] = ansi_ljust(indicator, width)

        # Pad instances
        while len(lines) < instance_rows:
            lines.append('')

        # Separator
        lines.append(f"{FG_GRAY}{'─' * width}{RESET}")

        # Messages - compact format with word wrap
        if self.messages:
            all_wrapped_lines = []

            # Find longest sender name for alignment
            max_sender_len = max(len(sender) for _, sender, _ in self.messages) if self.messages else 12
            max_sender_len = min(max_sender_len, 12)  # Cap at reasonable width

            for time_str, sender, message in self.messages:
                # Format timestamp
                try:
                    from datetime import datetime
                    if 'T' in time_str:
                        dt = datetime.fromisoformat(time_str.replace('Z', '+00:00'))
                        display_time = dt.strftime('%H:%M')
                    else:
                        display_time = time_str
                except Exception:
                    display_time = time_str[:5] if len(time_str) >= 5 else time_str

                # Smart truncate sender (prefix + suffix with middle ellipsis)
                sender_display = smart_truncate_name(sender, max_sender_len)

                # Replace literal newlines with space for preview
                display_message = message.replace('\n', ' ')

                # Bold @mentions in message (e.g., @name becomes **@name**)
                if '@' in display_message:
                    import re
                    display_message = re.sub(r'(@[\w\-_]+)', f'{BOLD}\\1{RESET}{FG_LIGHTGRAY}', display_message)

                # Calculate available width for message (reserve space for time + sender + spacing)
                # Format: "HH:MM sender message"
                prefix_len = 5 + 1 + max_sender_len + 1  # time + space + sender + space
                max_msg_len = width - prefix_len

                # Wrap message text
                if max_msg_len > 0:
                    wrapper = AnsiTextWrapper(width=max_msg_len)
                    wrapped = wrapper.wrap(display_message)

                    # Add timestamp/sender to first line, indent continuation lines manually
                    # Add color to each line so truncation doesn't lose formatting
                    indent = ' ' * prefix_len
                    for i, wrapped_line in enumerate(wrapped):
                        if i == 0:
                            line = f"{FG_GRAY}{display_time}{RESET} {sender_display:<{max_sender_len}} {FG_LIGHTGRAY}{wrapped_line}{RESET}"
                        else:
                            line = f"{indent}{FG_LIGHTGRAY}{wrapped_line}{RESET}"
                        all_wrapped_lines.append(line)
                else:
                    # Fallback if width too small
                    all_wrapped_lines.append(f"{FG_GRAY}{display_time}{RESET} {sender_display:<{max_sender_len}}")

            # Take last N lines to fit available space (mid-message truncation)
            visible_lines = all_wrapped_lines[-message_rows:] if len(all_wrapped_lines) > message_rows else all_wrapped_lines
            lines.extend(visible_lines)
        else:
            # ASCII art logo
            lines.append(f"{FG_GRAY}No messages - Tab to LAUNCH to create instances{RESET}")
            lines.append('')
            lines.append(f"{FG_ORANGE}     ╦ ╦╔═╗╔═╗╔╦╗{RESET}")
            lines.append(f"{FG_ORANGE}     ╠═╣║  ║ ║║║║{RESET}")
            lines.append(f"{FG_ORANGE}     ╩ ╩╚═╝╚═╝╩ ╩{RESET}")

        # Pad messages
        while len(lines) < instance_rows + message_rows + 1:  # +1 for separator
            lines.append('')

        # Separator before input
        lines.append(f"{FG_GRAY}{'─' * width}{RESET}")

        # Input area (auto-wrapped)
        input_lines = self.render_wrapped_input(width, input_rows)
        lines.extend(input_lines)

        # Separator after input (before footer)
        lines.append(f"{FG_GRAY}{'─' * width}{RESET}")

        # Pad to fill height
        while len(lines) < height:
            lines.append('')

        return lines[:height]

    def get_launch_command_preview(self) -> str:
        """Build preview using spec (matches exactly what will be launched)"""
        try:
            # Load spec and update with form values (same logic as do_launch)
            claude_args_str = self.config_edit.get('HCOM_CLAUDE_ARGS', '')
            spec = resolve_claude_args(None, claude_args_str if claude_args_str else None)

            # Update spec with form values
            system_flag = None
            system_value = None
            if self.launch_system_prompt:
                system_flag = "--system-prompt" if self.launch_background else "--append-system-prompt"
                system_value = self.launch_system_prompt

            spec = spec.update(
                background=self.launch_background,
                system_flag=system_flag,
                system_value=system_value,
                prompt=self.launch_prompt,
            )

            # Build preview
            parts = []

            # Environment variables (read from config_fields - source of truth)
            env_parts = []
            agent = self.config_edit.get('HCOM_AGENT', '')
            if agent:
                agent_display = agent if len(agent) <= 15 else agent[:12] + "..."
                env_parts.append(f"HCOM_AGENT={agent_display}")
            tag = self.config_edit.get('HCOM_TAG', '')
            if tag:
                tag_display = tag if len(tag) <= 15 else tag[:12] + "..."
                env_parts.append(f"HCOM_TAG={tag_display}")
            if env_parts:
                parts.append(" ".join(env_parts))

            # Base command
            count = self.launch_count if self.launch_count else "1"
            parts.append(f"hcom {count}")

            # Claude args from spec (truncate long values for preview)
            tokens = spec.rebuild_tokens(include_system=True)
            if tokens:
                preview_tokens = []
                for token in tokens:
                    if len(token) > 30:
                        preview_tokens.append(f'"{token[:27]}..."')
                    elif ' ' in token:
                        preview_tokens.append(f'"{token}"')
                    else:
                        preview_tokens.append(token)
                parts.append("claude " + " ".join(preview_tokens))

            return " ".join(parts)
        except Exception:
            return "(preview unavailable - check HCOM_CLAUDE_ARGS)"

    def get_current_launch_field_info(self) -> tuple[str, str, int] | None:
        """Get (field_key, field_value, cursor_pos) for currently selected field, or None"""
        if self.launch_field == LaunchField.CLAUDE_SECTION and self.claude_cursor >= 0:
            fields = self.build_claude_fields()
            if self.claude_cursor < len(fields):
                field = fields[self.claude_cursor]
                if field.key == 'prompt':
                    # Default cursor to end if not set or invalid
                    if self.launch_prompt_cursor > len(self.launch_prompt):
                        self.launch_prompt_cursor = len(self.launch_prompt)
                    return ('prompt', self.launch_prompt, self.launch_prompt_cursor)
                elif field.key == 'system_prompt':
                    if self.launch_system_prompt_cursor > len(self.launch_system_prompt):
                        self.launch_system_prompt_cursor = len(self.launch_system_prompt)
                    return ('system_prompt', self.launch_system_prompt, self.launch_system_prompt_cursor)
                elif field.key == 'claude_args':
                    value = self.config_edit.get('HCOM_CLAUDE_ARGS', '')
                    cursor = self.config_field_cursors.get('HCOM_CLAUDE_ARGS', len(value))
                    cursor = min(cursor, len(value))
                    self.config_field_cursors['HCOM_CLAUDE_ARGS'] = cursor
                    return ('HCOM_CLAUDE_ARGS', value, cursor)
        elif self.launch_field == LaunchField.HCOM_SECTION and self.hcom_cursor >= 0:
            fields = self.build_hcom_fields()
            if self.hcom_cursor < len(fields):
                field = fields[self.hcom_cursor]
                value = self.config_edit.get(field.key, '')
                cursor = self.config_field_cursors.get(field.key, len(value))
                cursor = min(cursor, len(value))
                self.config_field_cursors[field.key] = cursor
                return (field.key, value, cursor)
        elif self.launch_field == LaunchField.CUSTOM_ENV_SECTION and self.custom_env_cursor >= 0:
            fields = self.build_custom_env_fields()
            if self.custom_env_cursor < len(fields):
                field = fields[self.custom_env_cursor]
                value = self.config_edit.get(field.key, '')
                cursor = self.config_field_cursors.get(field.key, len(value))
                cursor = min(cursor, len(value))
                self.config_field_cursors[field.key] = cursor
                return (field.key, value, cursor)
        return None

    def update_launch_field(self, field_key: str, new_value: str, new_cursor: int):
        """Update a launch field with new value and cursor position (extracted helper)"""
        if field_key == 'prompt':
            self.launch_prompt = new_value
            self.launch_prompt_cursor = new_cursor
            self.save_launch_state()
        elif field_key == 'system_prompt':
            self.launch_system_prompt = new_value
            self.launch_system_prompt_cursor = new_cursor
            self.save_launch_state()
        elif field_key == 'HCOM_CLAUDE_ARGS':
            self.config_edit[field_key] = new_value
            self.config_field_cursors[field_key] = new_cursor
            self.save_config_to_file()
            self.load_launch_state()
        else:
            self.config_edit[field_key] = new_value
            self.config_field_cursors[field_key] = new_cursor
            self.save_config_to_file()

    def build_claude_fields(self) -> List[Field]:
        """Build Claude section fields from memory vars"""
        return [
            Field("prompt", "Prompt", "text", self.launch_prompt, hint="text string"),
            Field("system_prompt", "System Prompt", "text", self.launch_system_prompt, hint="text string"),
            Field("background", "Headless", "checkbox", self.launch_background, hint="enter to toggle"),
            Field("claude_args", "Claude Args", "text", self.config_edit.get('HCOM_CLAUDE_ARGS', ''), hint="flags string"),
        ]

    def build_hcom_fields(self) -> List[Field]:
        """Build HCOM section fields - always show all expected HCOM vars"""
        # Extract expected keys from DEFAULT_CONFIG_DEFAULTS (excluding HCOM_CLAUDE_ARGS)
        expected_keys = [
            line.split('=')[0] for line in DEFAULT_CONFIG_DEFAULTS
            if line.startswith('HCOM_') and not line.startswith('HCOM_CLAUDE_ARGS=')
        ]

        fields = []
        for key in expected_keys:
            display_name = key.replace('HCOM_', '').replace('_', ' ').title()
            override = CONFIG_FIELD_OVERRIDES.get(key, {})
            field_type = override.get('type', 'text')
            options = override.get('options')
            if callable(options):
                options = options()
            hint = override.get('hint', '')
            value = self.config_edit.get(key, '')
            fields.append(Field(key, display_name, field_type, value, options if isinstance(options, list) or options is None else None, hint))

        # Also include any extra HCOM_* vars from config_fields (user-added)
        for key in sorted(self.config_edit.keys()):
            if key.startswith('HCOM_') and key != 'HCOM_CLAUDE_ARGS' and key not in expected_keys:
                display_name = key.replace('HCOM_', '').replace('_', ' ').title()
                override = CONFIG_FIELD_OVERRIDES.get(key, {})
                field_type = override.get('type', 'text')
                options = override.get('options')
                if callable(options):
                    options = options()
                hint = override.get('hint', '')
                fields.append(Field(key, display_name, field_type, self.config_edit.get(key, ''), options if isinstance(options, list) or options is None else None, hint))

        return fields

    def build_custom_env_fields(self) -> List[Field]:
        """Build Custom Env section fields from config_fields"""
        return [Field(key, key, 'text', self.config_edit.get(key, ''))
                for key in sorted(self.config_edit.keys())
                if not key.startswith('HCOM_')]

    def render_section_fields(
        self,
        lines: List[str],
        fields: List[Field],
        expanded: bool,
        section_field: LaunchField,
        section_cursor: int,
        width: int,
        color: str
    ) -> int | None:
        """Render fields for an expandable section (extracted helper)

        Returns selected_field_start_line if a field is selected, None otherwise.
        """
        selected_field_start_line = None

        if expanded or (self.launch_field == section_field and section_cursor >= 0):
            visible_fields = fields if expanded else fields[:3]
            for i, field in enumerate(visible_fields):
                field_selected = (self.launch_field == section_field and section_cursor == i)
                if field_selected:
                    selected_field_start_line = len(lines)
                lines.append(self.render_field(field, field_selected, width, color))
            if not expanded and len(fields) > 3:
                lines.append(f"{FG_GRAY}    +{len(fields) - 3} more (enter to expand){RESET}")

        return selected_field_start_line

    def render_field(self, field: Field, selected: bool, width: int, value_color: str | None = None) -> str:
        """Render a single field line"""
        indent = "    "
        # Default to standard orange if not specified
        if value_color is None:
            value_color = FG_ORANGE

        # Determine if field is in config (for proper state display)
        in_config = field.key in self.config_edit

        # Format value based on type
        # For Claude fields (prompt, system_prompt, background), extract defaults from HCOM_CLAUDE_ARGS
        if field.key in ('prompt', 'system_prompt', 'background'):
            claude_args_default = CONFIG_DEFAULTS.get('HCOM_CLAUDE_ARGS', '')
            default_spec = resolve_claude_args(None, claude_args_default if claude_args_default else None)
            if field.key == 'prompt':
                default = default_spec.positional_tokens[0] if default_spec.positional_tokens else ""
            elif field.key == 'system_prompt':
                default = default_spec.user_system or default_spec.user_append or ""
            else:  # background
                default = default_spec.is_background
        else:
            default = CONFIG_DEFAULTS.get(field.key, '')

        # Check if field has validation error
        has_error = field.key in self.validation_errors

        if field.field_type == 'checkbox':
            check = '●' if field.value else '○'
            # Color if differs from default (False is default for checkboxes)
            is_modified = field.value != False
            value_str = f"{value_color if is_modified else FG_WHITE}{check}{RESET}"
        elif field.field_type == 'text':
            if field.value:
                # Has value - color only if different from default (normalize quotes and whitespace)
                field_value_normalized = str(field.value).strip().strip("'\"").strip()
                default_normalized = str(default).strip().strip("'\"").strip()
                is_modified = field_value_normalized != default_normalized
                color = value_color if is_modified else FG_WHITE
                value_str = f"{color}{field.value}{RESET}"
            else:
                # Empty - check what runtime will actually use
                field_value_normalized = str(field.value).strip().strip("'\"").strip()
                default_normalized = str(default).strip().strip("'\"").strip()
                # Runtime uses empty if field doesn't auto-revert to default
                # For HCOM_CLAUDE_ARGS and Prompt, empty stays empty (doesn't use default)
                runtime_reverts_to_default = field.key not in ('HCOM_CLAUDE_ARGS', 'prompt')

                if runtime_reverts_to_default:
                    # Empty → runtime uses default → NOT modified
                    value_str = f"{FG_WHITE}(default: {default}){RESET}" if default else f"{FG_WHITE}(empty){RESET}"
                else:
                    # Empty → runtime uses "" → IS modified if default is non-empty
                    is_modified = bool(default_normalized)  # Modified if default exists
                    if is_modified:
                        # Colored with default hint (no RESET between to preserve background when selected)
                        value_str = f"{value_color}(empty) {FG_GRAY}default: {default}{RESET}"
                    else:
                        # Empty and no default
                        value_str = f"{FG_WHITE}(empty){RESET}"
        else:  # cycle, numeric
            if field.value:
                # Has value - color only if different from default (normalize quotes)
                field_value_normalized = str(field.value).strip().strip("'\"")
                default_normalized = default.strip().strip("'\"")
                is_modified = field_value_normalized != default_normalized
                color = value_color if is_modified else FG_WHITE
                value_str = f"{color}{field.value}{RESET}"
            else:
                # Empty - check what runtime will actually use
                if field.field_type == 'numeric':
                    # Timeout fields: empty → runtime uses default → NOT modified
                    value_str = f"{FG_WHITE}(default: {default}){RESET}" if default else f"{FG_WHITE}(empty){RESET}"
                else:
                    # Cycle fields: empty → runtime uses default → NOT modified
                    value_str = f"{FG_WHITE}(default: {default}){RESET}" if default else f"{FG_WHITE}(empty){RESET}"

        if field.hint and selected:
            value_str += f"{BG_CHARCOAL}  {FG_GRAY}• {field.hint}{RESET}"

        # Build line
        if selected:
            arrow_color = FG_RED if has_error else FG_WHITE
            line = f"{indent}{BG_CHARCOAL}{arrow_color}{BOLD}▸ {field.display_name}:{RESET}{BG_CHARCOAL} {value_str}"
            return bg_ljust(truncate_ansi(line, width), width, BG_CHARCOAL)
        else:
            return truncate_ansi(f"{indent}{FG_WHITE}{field.display_name}:{RESET} {value_str}", width)

    def build_launch_screen(self, height: int, width: int) -> List[str]:
        """Build launch screen with expandable sections"""
        # Calculate editor space upfront (reserves bottom of screen)
        field_info = self.get_current_launch_field_info()

        # Calculate dynamic editor rows (like manage screen)
        if field_info:
            field_key, field_value, cursor_pos = field_info
            editor_content_rows = calculate_text_input_rows(field_value, width)
            editor_rows = editor_content_rows + 4  # +4 for separator, header, blank line, separator
            separator_rows = 0  # Editor includes separator
        else:
            editor_rows = 0
            editor_content_rows = 0
            separator_rows = 1  # Need separator when no editor

        form_height = height - editor_rows - separator_rows

        lines = []
        selected_field_start_line = None  # Track which line has the selected field

        lines.append('')  # Top padding

        # Count field (with left padding)
        count_selected = (self.launch_field == LaunchField.COUNT)
        if count_selected:
            selected_field_start_line = len(lines)
            line = f"  {BG_CHARCOAL}{FG_WHITE}{BOLD}\u25b8 Count:{RESET}{BG_CHARCOAL} {FG_ORANGE}{self.launch_count}{RESET}{BG_CHARCOAL}  {FG_GRAY}\u2022 \u2190\u2192 adjust{RESET}"
            lines.append(bg_ljust(line, width, BG_CHARCOAL))
        else:
            lines.append(f"  {FG_WHITE}Count:{RESET} {FG_ORANGE}{self.launch_count}{RESET}")

        # Launch button (with left padding)
        launch_selected = (self.launch_field == LaunchField.LAUNCH_BTN)
        if launch_selected:
            selected_field_start_line = len(lines)
            lines.append(f"  {BG_ORANGE}{FG_BLACK}{BOLD} \u25b6 Launch \u23ce {RESET}")
            # Show cwd when launch button is selected
            import os
            cwd = os.getcwd()
            max_cwd_width = width - 10  # Leave margin
            if len(cwd) > max_cwd_width:
                cwd = '\u2026' + cwd[-(max_cwd_width - 1):]
            lines.append(f"  {BG_CHARCOAL}{FG_GRAY} \u2022 {FG_WHITE}{cwd} {RESET}")
        else:
            lines.append(f"  {FG_GRAY}\u25b6{RESET} {FG_ORANGE}{BOLD}Launch{RESET}")

        lines.append('')  # Spacer
        lines.append(f"{DIM}{FG_GRAY}{BOX_H * width}{RESET}")  # Separator (dim)
        lines.append('')  # Spacer

        # Claude section header (with left padding)
        claude_selected = (self.launch_field == LaunchField.CLAUDE_SECTION and self.claude_cursor == -1)
        expand_marker = '\u25bc' if self.claude_expanded else '\u25b6'
        claude_fields = self.build_claude_fields()
        # Count fields modified from defaults by comparing with parsed default spec
        claude_set = 0

        # Parse default HCOM_CLAUDE_ARGS to get default prompt/system/background
        claude_args_default = CONFIG_DEFAULTS.get('HCOM_CLAUDE_ARGS', '')
        default_spec = resolve_claude_args(None, claude_args_default if claude_args_default else None)
        default_prompt = default_spec.positional_tokens[0] if default_spec.positional_tokens else ""
        default_system = default_spec.user_system or default_spec.user_append or ""
        default_background = default_spec.is_background

        if self.launch_background != default_background:
            claude_set += 1
        if self.launch_prompt != default_prompt:
            claude_set += 1
        if self.launch_system_prompt != default_system:
            claude_set += 1
        # claude_args: check if raw value differs from default (normalize quotes)
        claude_args_val = self.config_edit.get('HCOM_CLAUDE_ARGS', '').strip().strip("'\"")
        claude_args_default_normalized = claude_args_default.strip().strip("'\"")
        if claude_args_val != claude_args_default_normalized:
            claude_set += 1
        claude_total = len(claude_fields)
        claude_count = f" \u2022 {claude_set}/{claude_total}"
        if claude_selected:
            selected_field_start_line = len(lines)
            claude_action = "\u2190 collapse" if self.claude_expanded else "\u2192 expand"
            claude_hint = f"{claude_count} \u2022 {claude_action}"
            line = f"  {BG_CHARCOAL}{FG_CLAUDE_ORANGE}{BOLD}{expand_marker} Claude{RESET}{BG_CHARCOAL}  {FG_GRAY}{claude_hint}{RESET}"
            lines.append(bg_ljust(line, width, BG_CHARCOAL))
        else:
            lines.append(f"  {FG_CLAUDE_ORANGE}{BOLD}{expand_marker} Claude{RESET}{FG_GRAY}{claude_count}{RESET}")

        # Preview modified fields when collapsed, or show description if none
        if not self.claude_expanded:
            if claude_set > 0:
                previews = []
                if self.launch_background != default_background:
                    previews.append("background: true" if self.launch_background else "background: false")
                if self.launch_prompt != default_prompt:
                    prompt_str = str(self.launch_prompt) if self.launch_prompt else ""
                    prompt_preview = prompt_str[:20] + "..." if len(prompt_str) > 20 else prompt_str
                    previews.append(f'prompt: "{prompt_preview}"')
                if self.launch_system_prompt != default_system:
                    sys_str = str(self.launch_system_prompt) if self.launch_system_prompt else ""
                    sys_preview = sys_str[:20] + "..." if len(sys_str) > 20 else sys_str
                    previews.append(f'system: "{sys_preview}"')
                if claude_args_val != claude_args_default_normalized:
                    args_str = str(claude_args_val) if claude_args_val else ""
                    args_preview = args_str[:25] + "..." if len(args_str) > 25 else args_str
                    previews.append(f'args: "{args_preview}"')
                if previews:
                    preview_text = ", ".join(previews)
                    lines.append(f"    {DIM}{FG_GRAY}{truncate_ansi(preview_text, width - 4)}{RESET}")
            else:
                lines.append(f"    {DIM}{FG_GRAY}prompt, system, headless, args{RESET}")

        # Claude fields (if expanded or cursor inside)
        result = self.render_section_fields(
            lines, claude_fields, self.claude_expanded,
            LaunchField.CLAUDE_SECTION, self.claude_cursor, width, FG_CLAUDE_ORANGE
        )
        if result is not None:
            selected_field_start_line = result

        # HCOM section header (with left padding)
        hcom_selected = (self.launch_field == LaunchField.HCOM_SECTION and self.hcom_cursor == -1)
        expand_marker = '\u25bc' if self.hcom_expanded else '\u25b6'
        hcom_fields = self.build_hcom_fields()
        # Count fields modified from defaults (considering runtime behavior)
        def is_field_modified(f):
            default = CONFIG_DEFAULTS.get(f.key, '')
            if not f.value:  # Empty
                # Fields where empty reverts to default at runtime
                if f.key in ('HCOM_TERMINAL', 'HCOM_HINTS', 'HCOM_TAG', 'HCOM_AGENT', 'HCOM_TIMEOUT', 'HCOM_SUBAGENT_TIMEOUT'):
                    return False  # Empty → uses default → NOT modified
                # Fields where empty stays empty (different from default if default is non-empty)
                # HCOM_CLAUDE_ARGS: empty → "" (not default "'say hi...'") → IS modified
                return bool(default.strip().strip("'\""))  # Modified if default is non-empty
            # Has value - check if different from default
            return f.value.strip().strip("'\"") != default.strip().strip("'\"")
        hcom_set = sum(1 for f in hcom_fields if is_field_modified(f))
        hcom_total = len(hcom_fields)
        hcom_count = f" \u2022 {hcom_set}/{hcom_total}"
        if hcom_selected:
            selected_field_start_line = len(lines)
            hcom_action = "\u2190 collapse" if self.hcom_expanded else "\u2192 expand"
            hcom_hint = f"{hcom_count} \u2022 {hcom_action}"
            line = f"  {BG_CHARCOAL}{FG_CYAN}{BOLD}{expand_marker} HCOM{RESET}{BG_CHARCOAL}  {FG_GRAY}{hcom_hint}{RESET}"
            lines.append(bg_ljust(line, width, BG_CHARCOAL))
        else:
            lines.append(f"  {FG_CYAN}{BOLD}{expand_marker} HCOM{RESET}{FG_GRAY}{hcom_count}{RESET}")

        # Preview modified fields when collapsed, or show description if none
        if not self.hcom_expanded:
            if hcom_set > 0:
                previews = []
                for field in hcom_fields:
                    if is_field_modified(field):
                        val = field.value or ""
                        if hasattr(field, 'type') and field.type == 'bool':
                            val_str = "true" if val == "true" else "false"
                        else:
                            val = str(val) if val else ""
                            val_str = val[:15] + "..." if len(val) > 15 else val
                        # Shorten field names
                        short_name = field.display_name.lower().replace("hcom ", "")
                        previews.append(f'{short_name}: {val_str}')
                if previews:
                    preview_text = ", ".join(previews)
                    lines.append(f"    {DIM}{FG_GRAY}{truncate_ansi(preview_text, width - 4)}{RESET}")
            else:
                lines.append(f"    {DIM}{FG_GRAY}agent, tag, hints, timeout, terminal{RESET}")

        # HCOM fields
        result = self.render_section_fields(
            lines, hcom_fields, self.hcom_expanded,
            LaunchField.HCOM_SECTION, self.hcom_cursor, width, FG_CYAN
        )
        if result is not None:
            selected_field_start_line = result

        # Custom Env section header (with left padding)
        custom_selected = (self.launch_field == LaunchField.CUSTOM_ENV_SECTION and self.custom_env_cursor == -1)
        expand_marker = '\u25bc' if self.custom_env_expanded else '\u25b6'
        custom_fields = self.build_custom_env_fields()
        custom_set = sum(1 for f in custom_fields if f.value)
        custom_total = len(custom_fields)
        custom_count = f" \u2022 {custom_set}/{custom_total}"
        if custom_selected:
            selected_field_start_line = len(lines)
            custom_action = "\u2190 collapse" if self.custom_env_expanded else "\u2192 expand"
            custom_hint = f"{custom_count} \u2022 {custom_action}"
            line = f"  {BG_CHARCOAL}{FG_CUSTOM_ENV}{BOLD}{expand_marker} Custom Env{RESET}{BG_CHARCOAL}  {FG_GRAY}{custom_hint}{RESET}"
            lines.append(bg_ljust(line, width, BG_CHARCOAL))
        else:
            lines.append(f"  {FG_CUSTOM_ENV}{BOLD}{expand_marker} Custom Env{RESET}{FG_GRAY}{custom_count}{RESET}")

        # Preview modified fields when collapsed, or show description if none
        if not self.custom_env_expanded:
            if custom_set > 0:
                previews = []
                for field in custom_fields:
                    if field.value:
                        val = str(field.value) if field.value else ""
                        val_str = val[:15] + "..." if len(val) > 15 else val
                        previews.append(f'{field.key}: {val_str}')
                if previews:
                    preview_text = ", ".join(previews)
                    lines.append(f"    {DIM}{FG_GRAY}{truncate_ansi(preview_text, width - 4)}{RESET}")
            else:
                lines.append(f"    {DIM}{FG_GRAY}arbitrary environment variables{RESET}")

        # Custom Env fields
        result = self.render_section_fields(
            lines, custom_fields, self.custom_env_expanded,
            LaunchField.CUSTOM_ENV_SECTION, self.custom_env_cursor, width, FG_CUSTOM_ENV
        )
        if result is not None:
            selected_field_start_line = result

        # Open config in editor entry (at bottom, less prominent)
        lines.append('')  # Spacer
        editor_cmd, editor_label = self.resolve_editor_command()
        editor_label_display = editor_label or 'VS Code'
        editor_available = editor_cmd is not None
        editor_selected = (self.launch_field == LaunchField.OPEN_EDITOR)

        if editor_selected:
            selected_field_start_line = len(lines)
            lines.append(
                bg_ljust(
                    f"  {BG_CHARCOAL}{FG_WHITE}\u2197 Open config in {editor_label_display}{RESET}"
                    f"{BG_CHARCOAL}  "
                    f"{(FG_GRAY if editor_available else FG_RED)}\u2022 "
                    f"{'enter: open' if editor_available else 'code CLI not found / set $EDITOR'}{RESET}",
                    width,
                    BG_CHARCOAL,
                )
            )
        else:
            # Less prominent when not selected
            if editor_available:
                lines.append(f"  {FG_GRAY}\u2197 Open config in {editor_label_display}{RESET}")
            else:
                lines.append(f"  {FG_GRAY}\u2197 Open config in {editor_label_display} {FG_RED}(not found){RESET}")

        # Auto-scroll to keep selected field visible
        if selected_field_start_line is not None:
            max_scroll = max(0, len(lines) - form_height)

            # Scroll up if selected field is above visible window
            if selected_field_start_line < self.launch_scroll_pos:
                self.launch_scroll_pos = selected_field_start_line
            # Scroll down if selected field is below visible window
            elif selected_field_start_line >= self.launch_scroll_pos + form_height:
                self.launch_scroll_pos = selected_field_start_line - form_height + 1

            # Clamp scroll position
            self.launch_scroll_pos = max(0, min(self.launch_scroll_pos, max_scroll))

        # Render visible window instead of truncating
        if len(lines) > form_height:
            # Extract visible slice based on scroll position
            visible_lines = lines[self.launch_scroll_pos:self.launch_scroll_pos + form_height]
            # Pad if needed (shouldn't happen, but for safety)
            while len(visible_lines) < form_height:
                visible_lines.append('')
            lines = visible_lines
        else:
            # Form fits entirely, no scrolling needed
            while len(lines) < form_height:
                lines.append('')

        # Editor (if active) - always fits because we reserved space
        if field_info:
            field_key, field_value, cursor_pos = field_info

            # Build descriptive header for each field with background
            if field_key == 'prompt':
                editor_color = FG_CLAUDE_ORANGE
                field_name = "Prompt"
                help_text = "initial prompt sent on launch"
            elif field_key == 'system_prompt':
                editor_color = FG_CLAUDE_ORANGE
                field_name = "System Prompt"
                help_text = "instructions that guide behavior"
            elif field_key == 'HCOM_CLAUDE_ARGS':
                editor_color = FG_CLAUDE_ORANGE
                field_name = "Claude Args"
                help_text = "raw flags passed to Claude CLI"
            elif field_key == 'HCOM_TIMEOUT':
                editor_color = FG_CYAN
                field_name = "Timeout"
                help_text = "seconds before disconnecting idle instance"
            elif field_key == 'HCOM_SUBAGENT_TIMEOUT':
                editor_color = FG_CYAN
                field_name = "Subagent Timeout"
                help_text = "seconds before disconnecting idle subagent"
            elif field_key == 'HCOM_TERMINAL':
                editor_color = FG_CYAN
                field_name = "Terminal"
                help_text = "launch in new window, current window, or custom terminal"
            elif field_key == 'HCOM_HINTS':
                editor_color = FG_CYAN
                field_name = "Hints"
                help_text = "text appended to all messages this instance receives"
            elif field_key == 'HCOM_TAG':
                editor_color = FG_CYAN
                field_name = "Tag"
                help_text = "identifier to create groups with @-mention"
            elif field_key == 'HCOM_AGENT':
                editor_color = FG_CYAN
                field_name = "Agent"
                help_text = "agent from .claude/agents • comma-separated for multiple"
            elif field_key.startswith('HCOM_'):
                # Other HCOM fields
                editor_color = FG_CYAN
                field_name = field_key.replace('HCOM_', '').replace('_', ' ').title()
                help_text = "HCOM configuration variable"
            else:
                # Custom env vars
                editor_color = FG_CUSTOM_ENV
                field_name = field_key
                help_text = "custom environment variable"

            # Header line - bold field name, regular help text
            header = f"{editor_color}{BOLD}{field_name}:{RESET} {FG_GRAY}{help_text}{RESET}"
            lines.append(f"{FG_GRAY}{'─' * width}{RESET}")
            lines.append(header)
            lines.append('')  # Blank line between header and input
            # Render editor with wrapping support
            editor_lines = render_text_input(field_value, cursor_pos, width, editor_content_rows, prefix="")
            lines.extend(editor_lines)
            # Separator after editor input
            lines.append(f"{FG_GRAY}{'─' * width}{RESET}")
        else:
            # Separator at bottom when no editor
            lines.append(f"{FG_GRAY}{'─' * width}{RESET}")

        return lines[:height]

    def get_launch_footer(self) -> str:
        """Return context-sensitive footer for Launch screen"""
        # Count field
        if self.launch_field == LaunchField.COUNT:
            return f"{FG_GRAY}tab: switch  ←→: adjust  esc: reset to 1  ctrl+r: reset config{RESET}"

        # Launch button
        elif self.launch_field == LaunchField.LAUNCH_BTN:
            return f"{FG_GRAY}tab: switch  enter: launch  ctrl+r: reset config{RESET}"
        elif self.launch_field == LaunchField.OPEN_EDITOR:
            cmd, label = self.resolve_editor_command()
            if cmd:
                friendly = label or 'VS Code'
                return f"{FG_GRAY}tab: switch  enter: open {friendly}{RESET}"
            return f"{FG_GRAY}tab: switch  enter: install code CLI or set $EDITOR{RESET}"

        # Section headers (cursor == -1)
        elif self.launch_field == LaunchField.CLAUDE_SECTION and self.claude_cursor == -1:
            return f"{FG_GRAY}tab: switch  enter: expand/collapse  ctrl+r: reset config{RESET}"
        elif self.launch_field == LaunchField.HCOM_SECTION and self.hcom_cursor == -1:
            return f"{FG_GRAY}tab: switch  enter: expand/collapse  ctrl+r: reset config{RESET}"
        elif self.launch_field == LaunchField.CUSTOM_ENV_SECTION and self.custom_env_cursor == -1:
            return f"{FG_GRAY}tab: switch  enter: expand/collapse  ctrl+r: reset config{RESET}"

        # Fields within sections (cursor >= 0)
        elif self.launch_field == LaunchField.CLAUDE_SECTION and self.claude_cursor >= 0:
            fields = self.build_claude_fields()
            if self.claude_cursor < len(fields):
                field = fields[self.claude_cursor]
                if field.field_type == 'checkbox':
                    return f"{FG_GRAY}tab: switch  enter: toggle  ctrl+r: reset config{RESET}"
                else:  # text fields
                    return f"{FG_GRAY}tab: switch  type: edit  ←→: cursor  esc: clear  ctrl+r: reset config{RESET}"

        elif self.launch_field == LaunchField.HCOM_SECTION and self.hcom_cursor >= 0:
            fields = self.build_hcom_fields()
            if self.hcom_cursor < len(fields):
                field = fields[self.hcom_cursor]
                if field.field_type == 'cycle':
                    return f"{FG_GRAY}tab: switch  ←→: cycle options  esc: clear  ctrl+r: reset config{RESET}"
                elif field.field_type == 'numeric':
                    return f"{FG_GRAY}tab: switch  type: digits  ←→: cursor  esc: clear  ctrl+r: reset config{RESET}"
                else:  # text fields
                    return f"{FG_GRAY}tab: switch  type: edit  ←→: cursor  esc: clear  ctrl+r: reset config{RESET}"

        elif self.launch_field == LaunchField.CUSTOM_ENV_SECTION and self.custom_env_cursor >= 0:
            return f"{FG_GRAY}tab: switch  type: edit  ←→: cursor  esc: clear  ctrl+r: reset config{RESET}"

        # Fallback (should not happen)
        return f"{FG_GRAY}tab: switch  ctrl+r: reset config{RESET}"

    def render(self):
        """Render current screen"""
        cols, rows = get_terminal_size()
        # Adapt to any terminal size
        rows = max(10, rows)

        frame = []

        # Header (compact - no separator)
        header = self.build_status_bar()
        frame.append(ansi_ljust(header, cols))

        # Flash row with separator line
        flash = self.build_flash()
        if flash:
            # Flash message on left, separator line fills rest of row
            flash_len = ansi_len(flash)
            remaining = cols - flash_len - 1  # -1 for space
            separator = f"{FG_GRAY}{'─' * remaining}{RESET}" if remaining > 0 else ""
            frame.append(f"{flash} {separator}")
        else:
            # Just separator line when no flash message
            frame.append(f"{FG_GRAY}{'─' * cols}{RESET}")

        # Welcome message on first render
        if self.first_render:
            self.flash("Welcome! Tab to switch screens")
            self.first_render = False

        # Body (subtract 3: header, flash, footer)
        body_rows = rows - 3

        if self.mode == Mode.MANAGE:
            manage_lines = self.build_manage_screen(body_rows, cols)
            for line in manage_lines:
                frame.append(ansi_ljust(line, cols))
        elif self.mode == Mode.LAUNCH:
            form_lines = self.build_launch_screen(body_rows, cols)
            for line in form_lines:
                frame.append(ansi_ljust(line, cols))

        # Footer - compact help text
        if self.mode == Mode.MANAGE:
            # Contextual footer based on state
            if self.message_buffer.strip():
                footer = f"{FG_GRAY}tab: switch  @: mention  enter: send  esc: clear{RESET}"
            elif self.pending_stop_all:
                footer = f"{FG_GRAY}ctrl+a: confirm stop all  esc: cancel{RESET}"
            elif self.pending_reset:
                footer = f"{FG_GRAY}ctrl+r: confirm reset  esc: cancel{RESET}"
            elif self.pending_toggle:
                footer = f"{FG_GRAY}enter: confirm  esc: cancel{RESET}"
            else:
                footer = f"{FG_GRAY}tab: switch  @: mention  enter: toggle  ctrl+a: stop all  ctrl+r: reset{RESET}"
        elif self.mode == Mode.LAUNCH:
            footer = self.get_launch_footer()
        frame.append(truncate_ansi(footer, cols))

        # Repaint if changed
        if frame != self.last_frame:
            sys.stdout.write(CLEAR_SCREEN + CURSOR_HOME)
            for i, line in enumerate(frame):
                sys.stdout.write(line)
                if i < len(frame) - 1:
                    sys.stdout.write('\n')
            sys.stdout.flush()
            self.last_frame = frame

    def handle_tab(self):
        """Cycle between Manage, Launch, and native Log view"""
        if self.mode == Mode.MANAGE:
            self.mode = Mode.LAUNCH
            self.flash("Launch Instances")
        elif self.mode == Mode.LAUNCH:
            # Go directly to native log view instead of LOG mode
            self.flash("Message History")
            self.show_log_native()
            # After returning from native view, go to MANAGE
            self.mode = Mode.MANAGE
            self.flash("Manage Instances")

    def handle_manage_key(self, key: str):
        """Handle keys in Manage mode"""
        # Sort by creation time (same as display) - stable, no jumping
        sorted_instances = sorted(
            self.instances.items(),
            key=lambda x: -x[1]['data'].get('created_at', 0.0)
        )

        if key == 'UP':
            if sorted_instances and self.cursor > 0:
                self.cursor -= 1
                # Update tracked instance name
                if self.cursor < len(sorted_instances):
                    self.cursor_instance_name = sorted_instances[self.cursor][0]
                self.clear_all_pending_confirmations()
                self.sync_scroll_to_cursor()
        elif key == 'DOWN':
            if sorted_instances and self.cursor < len(sorted_instances) - 1:
                self.cursor += 1
                # Update tracked instance name
                if self.cursor < len(sorted_instances):
                    self.cursor_instance_name = sorted_instances[self.cursor][0]
                self.clear_all_pending_confirmations()
                self.sync_scroll_to_cursor()
        elif key == '@':
            self.clear_all_pending_confirmations()
            # Add @mention of highlighted instance at cursor position
            if sorted_instances and self.cursor < len(sorted_instances):
                name, _ = sorted_instances[self.cursor]
                mention = f"@{name} "
                if mention not in self.message_buffer:
                    self.message_buffer, self.message_cursor_pos = text_input_insert(
                        self.message_buffer, self.message_cursor_pos, mention
                    )
        elif key == 'SPACE':
            self.clear_all_pending_confirmations()
            # Add space to message buffer at cursor position
            self.message_buffer, self.message_cursor_pos = text_input_insert(
                self.message_buffer, self.message_cursor_pos, ' '
            )
        elif key == 'LEFT':
            self.clear_all_pending_confirmations()
            # Move cursor left in message buffer
            self.message_cursor_pos = text_input_move_left(self.message_cursor_pos)
        elif key == 'RIGHT':
            self.clear_all_pending_confirmations()
            # Move cursor right in message buffer
            self.message_cursor_pos = text_input_move_right(self.message_buffer, self.message_cursor_pos)
        elif key == 'ESC':
            # Clear message buffer first, then cancel all pending confirmations
            if self.message_buffer:
                self.message_buffer = ""
                self.message_cursor_pos = 0
            else:
                self.clear_all_pending_confirmations()
        elif key == 'BACKSPACE':
            self.clear_all_pending_confirmations()
            # Delete character before cursor in message buffer
            self.message_buffer, self.message_cursor_pos = text_input_backspace(
                self.message_buffer, self.message_cursor_pos
            )
        elif key == 'ENTER':
            # Clear stop all and reset confirmations (toggle handled separately below)
            self.clear_pending_confirmations_except('toggle')

            # Smart Enter: send message if text exists, otherwise toggle instances
            if self.message_buffer.strip():
                # Send message using cmd_send for consistent validation and error handling
                try:
                    message = self.message_buffer.strip()
                    result = cmd_send([message])
                    if result == 0:
                        self.flash("Sent")
                        # Clear message buffer and cursor
                        self.message_buffer = ""
                        self.message_cursor_pos = 0
                    else:
                        self.flash_error("Send failed")
                except Exception as e:
                    self.flash_error(f"Error: {str(e)}")
            else:
                # No message text - toggle instance with two-step confirmation
                if not sorted_instances or self.cursor >= len(sorted_instances):
                    return

                name, info = sorted_instances[self.cursor]
                enabled = info['data'].get('enabled', False)
                action = "start" if not enabled else "stop"

                # Get status color for name
                status = info.get('status', "unknown")
                color = STATUS_FG.get(status, FG_WHITE)

                # Check if confirming previous toggle
                if self.pending_toggle == name and (time.time() - self.pending_toggle_time) <= self.CONFIRMATION_TIMEOUT:
                    # Execute toggle (confirmation received)
                    try:
                        if enabled:
                            cmd_stop([name])
                            self.flash(f"Stopped hcom for {color}{name}{RESET}")
                            self.completed_toggle = name
                            self.completed_toggle_time = time.time()
                        else:
                            cmd_start([name])
                            self.flash(f"Started hcom for {color}{name}{RESET}")
                            self.completed_toggle = name
                            self.completed_toggle_time = time.time()
                        self.load_status()
                    except Exception as e:
                        self.flash_error(f"Error: {str(e)}")
                    finally:
                        self.pending_toggle = None
                else:
                    # Show confirmation (first press) - 10s duration
                    self.pending_toggle = name
                    self.pending_toggle_time = time.time()
                    # Name with status color, action is plain text (no color clash)
                    name_colored = f"{color}{name}{FG_WHITE}"
                    self.flash(f"Confirm {action} {name_colored}? (press Enter again)", duration=self.CONFIRMATION_FLASH_DURATION, color='white')

        elif key == 'CTRL_A':
            # Check state before clearing
            is_confirming = self.pending_stop_all and (time.time() - self.pending_stop_all_time) <= self.CONFIRMATION_TIMEOUT
            self.clear_pending_confirmations_except('stop_all')

            # Two-step confirmation for stop all
            if is_confirming:
                # Execute stop all (confirmation received)
                self.stop_all_instances()
                self.pending_stop_all = False
            else:
                # Show confirmation (first press) - 10s duration
                self.pending_stop_all = True
                self.pending_stop_all_time = time.time()
                self.flash(f"{FG_WHITE}Confirm stop all instances? (press Ctrl+A again){RESET}", duration=self.CONFIRMATION_FLASH_DURATION, color='white')

        elif key == 'CTRL_R':
            # Check state before clearing
            is_confirming = self.pending_reset and (time.time() - self.pending_reset_time) <= self.CONFIRMATION_TIMEOUT
            self.clear_pending_confirmations_except('reset')

            # Two-step confirmation for reset
            if is_confirming:
                # Execute reset (confirmation received)
                self.reset_logs()
                self.pending_reset = False
            else:
                # Show confirmation (first press)
                self.pending_reset = True
                self.pending_reset_time = time.time()
                self.flash(f"{FG_WHITE}Confirm clear & archive (log + instance list)? (press Ctrl+R again){RESET}", duration=self.CONFIRMATION_FLASH_DURATION, color='white')

        elif key == '\n':
            # Handle pasted newlines - insert literally
            self.clear_all_pending_confirmations()
            self.message_buffer, self.message_cursor_pos = text_input_insert(
                self.message_buffer, self.message_cursor_pos, '\n'
            )

        elif key and len(key) == 1 and key.isprintable():
            self.clear_all_pending_confirmations()
            # Insert printable characters at cursor position
            self.message_buffer, self.message_cursor_pos = text_input_insert(
                self.message_buffer, self.message_cursor_pos, key
            )

    def handle_launch_key(self, key: str):
        """Handle keys in Launch mode - with cursor-based bottom bar editing"""
    
        # UP/DOWN navigation (unchanged)
        if key == 'UP':
            if self.launch_field == LaunchField.CLAUDE_SECTION:
                if self.claude_cursor > -1:
                    self.claude_cursor -= 1
                else:
                    self.launch_field = LaunchField.LAUNCH_BTN
            elif self.launch_field == LaunchField.HCOM_SECTION:
                if self.hcom_cursor > -1:
                    self.hcom_cursor -= 1
                else:
                    self.launch_field = LaunchField.CLAUDE_SECTION
                    self.claude_cursor = -1
            elif self.launch_field == LaunchField.CUSTOM_ENV_SECTION:
                if self.custom_env_cursor > -1:
                    self.custom_env_cursor -= 1
                else:
                    self.launch_field = LaunchField.HCOM_SECTION
                    self.hcom_cursor = -1
            elif self.launch_field == LaunchField.OPEN_EDITOR:
                self.launch_field = LaunchField.CUSTOM_ENV_SECTION
                self.custom_env_cursor = -1
            else:
                fields = list(LaunchField)
                idx = fields.index(self.launch_field)
                self.launch_field = fields[(idx - 1) % len(fields)]
    
        elif key == 'DOWN':
            if self.launch_field == LaunchField.CLAUDE_SECTION:
                if self.claude_cursor == -1 and not self.claude_expanded:
                    self.launch_field = LaunchField.HCOM_SECTION
                    self.hcom_cursor = -1
                elif self.claude_expanded:
                    max_idx = len(self.build_claude_fields()) - 1
                    if self.claude_cursor < max_idx:
                        self.claude_cursor += 1
                    else:
                        self.launch_field = LaunchField.HCOM_SECTION
                        self.hcom_cursor = -1
            elif self.launch_field == LaunchField.HCOM_SECTION:
                if self.hcom_cursor == -1 and not self.hcom_expanded:
                    self.launch_field = LaunchField.CUSTOM_ENV_SECTION
                    self.custom_env_cursor = -1
                elif self.hcom_expanded:
                    max_idx = len(self.build_hcom_fields()) - 1
                    if self.hcom_cursor < max_idx:
                        self.hcom_cursor += 1
                    else:
                        self.launch_field = LaunchField.CUSTOM_ENV_SECTION
                        self.custom_env_cursor = -1
            elif self.launch_field == LaunchField.CUSTOM_ENV_SECTION:
                if self.custom_env_cursor == -1 and not self.custom_env_expanded:
                    self.launch_field = LaunchField.OPEN_EDITOR
                elif self.custom_env_expanded:
                    max_idx = len(self.build_custom_env_fields()) - 1
                    if self.custom_env_cursor < max_idx:
                        self.custom_env_cursor += 1
                    else:
                        self.launch_field = LaunchField.OPEN_EDITOR
            else:
                fields = list(LaunchField)
                idx = fields.index(self.launch_field)
                self.launch_field = fields[(idx + 1) % len(fields)]
                if self.launch_field == LaunchField.CLAUDE_SECTION:
                    self.claude_cursor = -1
                elif self.launch_field == LaunchField.HCOM_SECTION:
                    self.hcom_cursor = -1
                elif self.launch_field == LaunchField.CUSTOM_ENV_SECTION:
                    self.custom_env_cursor = -1
    
        # LEFT/RIGHT: adjust count, cycle for cycle fields, cursor movement for text fields
        elif key == 'LEFT' or key == 'RIGHT':
            # COUNT field: adjust by ±1
            if self.launch_field == LaunchField.COUNT:
                try:
                    current = int(self.launch_count) if self.launch_count else 1
                    if key == 'RIGHT':
                        current = min(999, current + 1)
                    else:  # LEFT
                        current = max(1, current - 1)
                    self.launch_count = str(current)
                except ValueError:
                    self.launch_count = "1"
            else:
                field_info = self.get_current_launch_field_info()
                if field_info:
                    field_key, field_value, cursor_pos = field_info

                    # Get field object to check type
                    field_obj = None
                    if self.launch_field == LaunchField.HCOM_SECTION and self.hcom_cursor >= 0:
                        fields = self.build_hcom_fields()
                        if self.hcom_cursor < len(fields):
                            field_obj = fields[self.hcom_cursor]

                    # Check if it's a cycle field
                    if field_obj and field_obj.field_type == 'cycle':
                        # Cycle through options
                        options = field_obj.options or []
                        if options:
                            if field_value in options:
                                idx = options.index(field_value)
                                new_idx = (idx + 1) if key == 'RIGHT' else (idx - 1)
                                new_idx = new_idx % len(options)
                            else:
                                new_idx = 0
                            self.config_edit[field_key] = options[new_idx]
                            self.config_field_cursors[field_key] = len(options[new_idx])
                            self.save_config_to_file()
                    else:
                        # Text field: move cursor
                        if key == 'LEFT':
                            new_cursor = text_input_move_left(cursor_pos)
                        else:
                            new_cursor = text_input_move_right(field_value, cursor_pos)

                        # Update cursor
                        if field_key == 'prompt':
                            self.launch_prompt_cursor = new_cursor
                        elif field_key == 'system_prompt':
                            self.launch_system_prompt_cursor = new_cursor
                        else:
                            self.config_field_cursors[field_key] = new_cursor
    
        # ENTER: expand/collapse, toggle, cycle, launch
        elif key == 'ENTER':
            if self.launch_field == LaunchField.CLAUDE_SECTION and self.claude_cursor == -1:
                self.claude_expanded = not self.claude_expanded
            elif self.launch_field == LaunchField.HCOM_SECTION and self.hcom_cursor == -1:
                self.hcom_expanded = not self.hcom_expanded
            elif self.launch_field == LaunchField.CUSTOM_ENV_SECTION and self.custom_env_cursor == -1:
                self.custom_env_expanded = not self.custom_env_expanded
            elif self.launch_field == LaunchField.CLAUDE_SECTION and self.claude_cursor >= 0:
                fields = self.build_claude_fields()
                if self.claude_cursor < len(fields):
                    field = fields[self.claude_cursor]
                    if field.field_type == 'checkbox' and field.key == 'background':
                        self.launch_background = not self.launch_background
                        self.save_launch_state()
            elif self.launch_field == LaunchField.LAUNCH_BTN:
                self.do_launch()
            elif self.launch_field == LaunchField.OPEN_EDITOR:
                self.open_config_in_editor()

        # BACKSPACE: delete char before cursor
        elif key == 'BACKSPACE':
            field_info = self.get_current_launch_field_info()
            if field_info:
                field_key, field_value, cursor_pos = field_info
                new_value, new_cursor = text_input_backspace(field_value, cursor_pos)
                self.update_launch_field(field_key, new_value, new_cursor)
    
        # ESC: clear field
        elif key == 'ESC':
            if self.launch_field == LaunchField.CLAUDE_SECTION:
                if self.claude_cursor >= 0:
                    fields = self.build_claude_fields()
                    if self.claude_cursor < len(fields):
                        field = fields[self.claude_cursor]
                        if field.key == 'prompt':
                            self.launch_prompt = ""
                            self.launch_prompt_cursor = 0
                            self.save_launch_state()
                        elif field.key == 'system_prompt':
                            self.launch_system_prompt = ""
                            self.launch_system_prompt_cursor = 0
                            self.save_launch_state()
                        elif field.key == 'claude_args':
                            self.config_edit['HCOM_CLAUDE_ARGS'] = ""
                            self.config_field_cursors['HCOM_CLAUDE_ARGS'] = 0
                            self.save_config_to_file()
                            self.load_launch_state()
                else:
                    self.claude_expanded = False
                    self.claude_cursor = -1
            elif self.launch_field == LaunchField.HCOM_SECTION:
                if self.hcom_cursor >= 0:
                    fields = self.build_hcom_fields()
                    if self.hcom_cursor < len(fields):
                        field = fields[self.hcom_cursor]
                        self.config_edit[field.key] = ""
                        self.config_field_cursors[field.key] = 0
                        self.save_config_to_file()
                else:
                    self.hcom_expanded = False
                    self.hcom_cursor = -1
            elif self.launch_field == LaunchField.CUSTOM_ENV_SECTION:
                if self.custom_env_cursor >= 0:
                    fields = self.build_custom_env_fields()
                    if self.custom_env_cursor < len(fields):
                        field = fields[self.custom_env_cursor]
                        self.config_edit[field.key] = ""
                        self.config_field_cursors[field.key] = 0
                        self.save_config_to_file()
                else:
                    self.custom_env_expanded = False
                    self.custom_env_cursor = -1
            elif self.launch_field == LaunchField.COUNT:
                self.launch_count = "1"

        # CTRL_R: Reset config to defaults (two-step confirmation)
        elif key == 'CTRL_R':
            is_confirming = self.pending_reset and (time.time() - self.pending_reset_time) <= self.CONFIRMATION_TIMEOUT

            if is_confirming:
                # Execute config reset
                try:
                    cmd_reset(['config'])
                    self.load_config_from_file()
                    self.load_launch_state()
                    self.flash("Config reset to defaults")
                except Exception as e:
                    self.flash_error(f"Reset failed: {str(e)}")
                finally:
                    self.pending_reset = False
            else:
                # Show confirmation (first press)
                self.pending_reset = True
                self.pending_reset_time = time.time()
                self.flash(f"{FG_WHITE}Backup + reset config to defaults? (Ctrl+R again){RESET}", duration=self.CONFIRMATION_FLASH_DURATION, color='white')

        # SPACE and printable: insert at cursor
        elif key == 'SPACE' or (key and len(key) == 1 and key.isprintable()):
            char = ' ' if key == 'SPACE' else key
            field_info = self.get_current_launch_field_info()
            if field_info:
                field_key, field_value, cursor_pos = field_info

                # Validate for special fields
                if field_key == 'HCOM_TAG':
                    override = CONFIG_FIELD_OVERRIDES.get(field_key, {})
                    allowed_pattern = override.get('allowed_chars')
                    if allowed_pattern:
                        test_value = field_value[:cursor_pos] + char + field_value[cursor_pos:]
                        if not re.match(allowed_pattern, test_value):
                            return

                new_value, new_cursor = text_input_insert(field_value, cursor_pos, char)
                self.update_launch_field(field_key, new_value, new_cursor)

    def do_launch(self):
        """Execute launch using full spec integration"""
        # Check for validation errors first
        if self.validation_errors:
            error_fields = ', '.join(self.validation_errors.keys())
            self.flash_error(f"Fix config errors before launching: {error_fields}", duration=15.0)
            return

        # Parse count
        try:
            count = int(self.launch_count) if self.launch_count else 1
        except ValueError:
            self.flash_error("Invalid count - must be number")
            return

        # Load current spec from config
        try:
            claude_args_str = self.config_edit.get('HCOM_CLAUDE_ARGS', '')
            spec = resolve_claude_args(None, claude_args_str if claude_args_str else None)
        except Exception as e:
            self.flash_error(f"Failed to parse HCOM_CLAUDE_ARGS: {e}")
            return

        # Check for parse errors BEFORE update (update loses original errors)
        if spec.errors:
            self.flash_error(f"Invalid HCOM_CLAUDE_ARGS: {'; '.join(spec.errors)}")
            return

        # System flag matches background mode
        system_flag = None
        system_value = None
        if self.launch_system_prompt:
            system_flag = "--system-prompt" if self.launch_background else "--append-system-prompt"
            system_value = self.launch_system_prompt

        spec = spec.update(
            background=self.launch_background,
            system_flag=system_flag,
            system_value=system_value,
            prompt=self.launch_prompt,  # Always pass value (empty string deletes)
        )

        # Build argv using spec (preserves all flags from HCOM_CLAUDE_ARGS)
        argv = [str(count), 'claude'] + spec.rebuild_tokens(include_system=True)

        # Set env vars if specified (read from config_fields - source of truth)
        env_backup = {}
        try:
            agent = self.config_edit.get('HCOM_AGENT', '')
            if agent:
                env_backup['HCOM_AGENT'] = os.environ.get('HCOM_AGENT')
                os.environ['HCOM_AGENT'] = agent
            tag = self.config_edit.get('HCOM_TAG', '')
            if tag:
                env_backup['HCOM_TAG'] = os.environ.get('HCOM_TAG')
                os.environ['HCOM_TAG'] = tag

            # Show launching message
            self.flash(f"Launching {count} instances...")
            self.render()  # Force update to show message

            # Call hcom.cmd_launch (handles all validation)
            # Add --no-auto-watch flag to prevent opening another watch window
            reload_config()
            result = cmd_launch(argv + ['--no-auto-watch'])

            if result == 0:  # Success
                # Switch to Manage screen to see new instances
                self.mode = Mode.MANAGE
                self.flash(f"Launched {count} instances")
                self.load_status()  # Refresh immediately
            else:
                self.flash_error("Launch failed - check instances")

        except Exception as e:
            # cmd_launch raises CLIError for validation failures
            self.flash_error(str(e))
        finally:
            # Restore env (clean up)
            for key, val in env_backup.items():
                if val is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = val

    def format_multiline_log(self, display_time: str, sender: str, message: str) -> List[str]:
        """Format log message with multiline support (indented continuation lines)"""
        if '\n' not in message:
            return [f"{FG_GRAY}{display_time}{RESET} {FG_ORANGE}{sender}{RESET}: {message}"]

        lines = message.split('\n')
        result = [f"{FG_GRAY}{display_time}{RESET} {FG_ORANGE}{sender}{RESET}: {lines[0]}"]
        indent = ' ' * (len(display_time) + len(sender) + 2)
        result.extend(indent + line for line in lines[1:])
        return result

    def render_log_message(self, msg: dict):
        """Render a single log message (extracted helper)"""
        time_str = msg.get('timestamp', '')
        sender = msg.get('from', '')
        message = msg.get('message', '')
        display_time = format_timestamp(time_str)

        for line in self.format_multiline_log(display_time, sender, message):
            print(line)
        print()  # Empty line between messages

    def render_status_with_separator(self, highlight_tab: str = "LOG"):
        """Render separator line and status bar (extracted helper)"""
        cols, _ = get_terminal_size()

        # Separator or flash line
        flash = self.build_flash()
        if flash:
            flash_len = ansi_len(flash)
            remaining = cols - flash_len - 1
            separator = f"{FG_GRAY}{'─' * remaining}{RESET}" if remaining > 0 else ""
            print(f"{flash} {separator}")
        else:
            print(f"{FG_GRAY}{'─' * cols}{RESET}")

        # Status line
        safe_width = cols - 2
        status = truncate_ansi(self.build_status_bar(highlight_tab=highlight_tab), safe_width)
        sys.stdout.write(status)
        sys.stdout.flush()

    def show_log_native(self):
        """Exit TUI, show streaming log in native buffer with status line"""
        # Exit alt screen
        sys.stdout.write('\033[?1049l' + SHOW_CURSOR)
        sys.stdout.flush()

        log_file = self.hcom_dir / 'hcom.log'

        def redraw_all():
            """Redraw entire log and status (on entry or resize)"""
            # Clear screen
            sys.stdout.write('\033[2J\033[H')
            sys.stdout.flush()

            # Dump existing log with formatting
            has_messages = False
            if log_file.exists():
                try:
                    result = parse_log_messages(log_file)
                    if result and hasattr(result, 'messages') and result.messages:
                        for msg in result.messages:
                            self.render_log_message(msg)
                        has_messages = True
                except Exception:
                    pass

            # Separator and status
            if has_messages:
                self.render_status_with_separator("LOG")
            else:
                # No messages - show placeholder
                self.render_status_with_separator("LOG")
                print()
                print(f"{FG_GRAY}No messages - Tab to LAUNCH to create instances{RESET}")

            cols, _ = get_terminal_size()
            return log_file.stat().st_size if log_file.exists() else 0, cols

        # Initial draw
        last_pos, last_width = redraw_all()
        last_status_update = time.time()
        has_messages_state = last_pos > 0  # Track if we have messages

        with KeyboardInput() as kbd:
            while True:
                key = kbd.get_key()
                if key == 'TAB':
                    # Tab to exit back to TUI
                    sys.stdout.write('\r\033[K')  # Clear status line
                    break

                # Update status every 0.5s - also check for resize
                now = time.time()
                if now - last_status_update > 0.5:
                    current_cols, _ = get_terminal_size()
                    self.load_status()  # Refresh instance data

                    # Check if status line is too long for current terminal width
                    status_line = self.build_status_bar(highlight_tab="LOG")
                    status_len = ansi_len(status_line)

                    if status_len >= current_cols - 2:
                        # Status would wrap - need full redraw to fix it
                        last_pos, last_width = redraw_all()
                        has_messages_state = last_pos > 0
                    else:
                        # Status fits - just update it
                        safe_width = current_cols - 2
                        new_status = truncate_ansi(status_line, safe_width)

                        # If we were in "no messages" state, cursor is 2 lines below status
                        if not has_messages_state:
                            # Move up 2 lines to status, clear all 3 lines, update status, re-print message
                            sys.stdout.write('\033[A\033[A\r\033[K' + new_status + '\n\033[K\n\033[K')
                            sys.stdout.write(f"{FG_GRAY}No messages - Tab to LAUNCH to create instances{RESET}")
                        else:
                            # Normal update - update separator/flash line and status line
                            # Move up to separator line, update it, then update status
                            flash = self.build_flash()
                            if flash:
                                # Flash message on left, separator fills rest
                                flash_len = ansi_len(flash)
                                remaining = current_cols - flash_len - 1  # -1 for space
                                separator = f"{FG_GRAY}{'─' * remaining}{RESET}" if remaining > 0 else ""
                                separator_line = f"{flash} {separator}"
                            else:
                                separator_line = f"{FG_GRAY}{'─' * current_cols}{RESET}"
                            sys.stdout.write('\r\033[A\033[K' + separator_line + '\n\033[K' + new_status)

                        sys.stdout.flush()
                        last_width = current_cols

                    last_status_update = now

                # Stream new messages
                if log_file.exists():
                    current_size = log_file.stat().st_size
                    if current_size > last_pos:
                        try:
                            result = parse_log_messages(log_file, last_pos)
                            if result and hasattr(result, 'messages') and result.messages:
                                # Clear separator and status: move up to separator, clear it and status, return to position
                                sys.stdout.write('\r\033[A\033[K\n\033[K\033[A\r')

                                # Render new messages
                                for msg in result.messages:
                                    self.render_log_message(msg)

                                # Redraw separator and status
                                self.render_status_with_separator("LOG")
                                has_messages_state = True  # We now have messages
                            last_pos = current_size
                        except Exception:
                            # Parse failed - skip this update
                            pass

                time.sleep(0.01)

        # Return to TUI
        sys.stdout.write(HIDE_CURSOR + '\033[?1049h')
        sys.stdout.flush()

    def handle_key(self, key: str):
        """Handle key press based on current mode"""
        if self.mode == Mode.MANAGE:
            self.handle_manage_key(key)
        elif self.mode == Mode.LAUNCH:
            self.handle_launch_key(key)
