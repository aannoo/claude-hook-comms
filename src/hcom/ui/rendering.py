"""ANSI rendering and text formatting utilities"""
import re
import shutil
import textwrap
import unicodedata
from typing import List, Tuple

# Import ANSI codes from shared
from ..shared import (
    RESET, FG_GRAY, FG_WHITE, BG_CHARCOAL
)

ANSI_RE = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')

# TUI Layout Constants
MAX_INPUT_ROWS = 8  # Cap input area at N rows


def ansi_len(text: str) -> int:
    """Get visible length of text (excluding ANSI codes), accounting for wide chars"""
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
