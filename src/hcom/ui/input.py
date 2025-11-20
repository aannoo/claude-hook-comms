"""Keyboard input handling and text editing"""
from __future__ import annotations
import os
import sys
import select
import re
import unicodedata
from typing import Optional, List

# Platform detection
IS_WINDOWS = os.name == 'nt'

# ANSI escape code pattern (for stripping from pasted input)
ANSI_RE = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')

# Import ANSI-aware utilities from rendering module
from .rendering import ansi_len

# Import platform-specific modules conditionally
if IS_WINDOWS:
    import msvcrt
else:
    import termios
    import tty

# Import ANSI codes from shared module
from ..shared import (
    HIDE_CURSOR, SHOW_CURSOR,
    FG_GRAY, FG_WHITE, RESET, REVERSE
)

# Constants
MAX_INPUT_ROWS = 8  # Cap input area at N rows


def slice_by_visual_width(text: str, max_width: int) -> tuple[str, int]:
    """Slice text to fit within visual width, accounting for wide chars and ANSI codes.

    Returns: (chunk_text, chars_consumed)
    """
    visual_width = 0
    char_pos = 0

    while char_pos < len(text) and visual_width < max_width:
        # Skip ANSI codes (preserve them but don't count their width)
        if char_pos < len(text) and text[char_pos:char_pos+1] == '\x1b':
            match = ANSI_RE.match(text, char_pos)
            if match:
                char_pos = match.end()
                continue

        if char_pos >= len(text):
            break

        # Check character width
        char = text[char_pos]
        ea_width = unicodedata.east_asian_width(char)
        if ea_width in ('F', 'W'):  # Fullwidth or Wide
            char_width = 2
        elif ea_width in ('Na', 'H', 'N', 'A'):  # Narrow, Half-width, Neutral, Ambiguous
            char_width = 1
        else:  # Zero-width characters (combining marks, etc.)
            char_width = 0

        # Check if it fits
        if visual_width + char_width <= max_width:
            visual_width += char_width
            char_pos += 1
        else:
            break  # No more space

    return text[:char_pos], char_pos


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
    # Strip ANSI codes from pasted text (prevents cursor/layout issues)
    clean_text = ANSI_RE.sub('', text)
    new_buffer = buffer[:cursor] + clean_text + buffer[cursor:]
    new_cursor = cursor + len(clean_text)
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

    # Guard against invalid width
    if width <= 0:
        return max_rows

    lines = text.split('\n')
    total_rows = 0
    for line in lines:
        if not line:
            total_rows += 1
        else:
            # Use visual width (accounts for wide chars and ANSI codes)
            total_rows += max(1, (ansi_len(line) + width - 1) // width)
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
        return [f"{FG_GRAY}{prefix}{REVERSE} {RESET}{RESET}"]

    line_width = width - len(prefix)
    # Guard against invalid width (terminal too narrow)
    if line_width <= 0:
        return [f"{FG_GRAY}{prefix}{RESET}"]  # Just show prefix if no room

    before = buffer[:cursor]

    # Cursor inverts colors of character at position (or shows inverted space at end)
    if cursor < len(buffer):
        # Cursor inverts the character at cursor position
        cursor_char = buffer[cursor]
        after = buffer[cursor+1:]
        full = before + REVERSE + cursor_char + RESET + after
    else:
        # Cursor at end - show inverted space after last char
        full = before + REVERSE + ' ' + RESET

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
            # Wrap long lines by visual width (handles wide chars and ANSI codes)
            char_offset = 0
            is_first_chunk = True
            while char_offset < len(line):
                chunk, consumed = slice_by_visual_width(line[char_offset:], line_width)
                if not consumed:  # Safety: avoid infinite loop
                    break
                line_prefix = prefix if line_idx == 0 and is_first_chunk else " " * len(prefix)
                wrapped.append(f"{FG_WHITE}{line_prefix}{RESET}{FG_WHITE}{chunk}{RESET}")
                char_offset += consumed
                is_first_chunk = False

    # Pad or truncate to max_rows
    result = wrapped + [''] * max(0, max_rows - len(wrapped))
    return result[:max_rows]
