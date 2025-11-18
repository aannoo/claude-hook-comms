"""Manage mode screen implementation"""
from __future__ import annotations
import re
import time
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from .tui import HcomTUI
    from .types import UIState

# Import rendering utilities
from .rendering import (
    ansi_len, ansi_ljust, bg_ljust, truncate_ansi,
    smart_truncate_name, get_terminal_size, AnsiTextWrapper
)

# Import input utilities
from .input import (
    render_text_input, calculate_text_input_rows,
    text_input_insert, text_input_backspace,
    text_input_move_left, text_input_move_right
)

# Import from shared
from ..shared import (
    RESET, BOLD, DIM,
    FG_WHITE, FG_GRAY, FG_YELLOW, FG_LIGHTGRAY, FG_ORANGE, FG_GOLD,
    BG_CHARCOAL, BG_GRAY,
    STATUS_MAP, STATUS_FG,
)

# Import from api
from ..api import (
    get_config,
    cmd_send, cmd_start, cmd_stop,
)


class ManageScreen:
    """Manage mode: instance list + messages + input"""

    def __init__(self, state: UIState, tui: HcomTUI):
        self.state = state  # Shared state (explicit dependency)
        self.tui = tui      # For commands only (flash, stop_all, etc)

    def build(self, height: int, width: int) -> List[str]:
        """Build manage screen layout"""
        # Use minimum height for layout calculation to maintain structure
        layout_height = max(10, height)

        lines = []

        # Calculate layout using shared function
        instance_rows, message_rows, input_rows = self.calculate_layout(layout_height, width)

        # Sort instances by creation time (newest first) - stable, no jumping
        sorted_instances = sorted(
            self.state.instances.items(),
            key=lambda x: -x[1]['data'].get('created_at', 0.0)
        )

        total_instances = len(sorted_instances)

        # Restore cursor position by instance name (stable across sorts)
        if self.state.cursor_instance_name and sorted_instances:
            found = False
            for i, (name, _) in enumerate(sorted_instances):
                if name == self.state.cursor_instance_name:
                    self.state.cursor = i
                    found = True
                    break
            if not found:
                # Instance disappeared, reset cursor
                self.state.cursor = 0
                self.state.cursor_instance_name = None
                self.sync_scroll_to_cursor()

        # Ensure cursor is valid
        if sorted_instances:
            self.state.cursor = max(0, min(self.state.cursor, total_instances - 1))
            # Update tracked instance name
            if self.state.cursor < len(sorted_instances):
                self.state.cursor_instance_name = sorted_instances[self.state.cursor][0]
        else:
            self.state.cursor = 0
            self.state.cursor_instance_name = None

        # Empty state - no instances
        if total_instances == 0:
            lines.append('')
            lines.append(f"{FG_GRAY}No instances - Press Tab → LAUNCH{RESET}")
            lines.append('')
            # Pad to instance_rows
            while len(lines) < instance_rows:
                lines.append('')
        else:
            # Calculate visible window
            max_scroll = max(0, total_instances - instance_rows)
            self.state.instance_scroll_pos = max(0, min(self.state.instance_scroll_pos, max_scroll))

            visible_start = self.state.instance_scroll_pos
            visible_end = min(visible_start + instance_rows, total_instances)
            visible_instances = sorted_instances[visible_start:visible_end]

            # Calculate dynamic name column width based on actual names
            max_instance_name_len = max((len(name) for name, _ in sorted_instances), default=0)
            # Check if any instance has badges
            from ..core.instances import is_external_sender
            has_background = any(info.get('data', {}).get('background', False) for _, info in sorted_instances)
            has_external = any(is_external_sender(info.get('data', {})) for _, info in sorted_instances)
            # Calculate max badge length: " [headless]" (11) + " [external]" (11) = 22
            badge_len = 0
            if has_background: badge_len += 11
            if has_external: badge_len += 11
            # Add space for state symbol on cursor row (2 chars: " +")
            name_col_width = max_instance_name_len + badge_len + 2
            # Set bounds: min 20, max based on terminal width
            # Reserve: 2 (icon) + 10 (age) + 2 (sep) + 30 (desc min) = 44
            max_name_width = max(20, width - 44)
            name_col_width = max(20, min(name_col_width, max_name_width))

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

                # Badges - include in name before padding
                is_background = info.get('data', {}).get('background', False)
                is_external = is_external_sender(info.get('data', {}))
                bg_marker_text = " [headless]" if is_background else ""
                ext_marker_text = " [external]" if is_external else ""
                badges = bg_marker_text + ext_marker_text
                badge_visible_len = len(bg_marker_text) + len(ext_marker_text)

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

                # Smart truncate name to fit in dynamic column width
                # Available: name_col_width - bg_marker_len - (2 for " +/-" on cursor row)
                max_name_len = name_col_width - badge_visible_len - 2  # Leave 2 chars for " +" or " -"
                display_name = smart_truncate_name(name, max_name_len)

                # State indicator (only on cursor row)
                if absolute_idx == self.state.cursor:
                    is_pending = self.state.pending_toggle == name and (time.time() - self.state.pending_toggle_time) <= self.tui.CONFIRMATION_TIMEOUT
                    if is_pending:
                        state_symbol = "±"
                        state_color = FG_GOLD
                    elif enabled:
                        state_symbol = "+"
                        state_color = color
                    else:
                        state_symbol = "-"
                        state_color = color
                    # Format: name [badges] +/-
                    name_with_marker = f"{display_name}{badges} {state_symbol}"
                    name_padded = ansi_ljust(name_with_marker, name_col_width)
                else:
                    # Format: name [badges]
                    name_with_marker = f"{display_name}{badges}"
                    name_padded = ansi_ljust(name_with_marker, name_col_width)

                # Description separator - only show if description exists
                desc_sep = ": " if display_text else ""

                # Bold if enabled, dim if disabled
                weight = BOLD if enabled else DIM

                if absolute_idx == self.state.cursor:
                    # Highlighted row - Format: icon name [badges] +/-  age ago: description [timeout]
                    line = f"{BG_CHARCOAL}{color}{icon} {weight}{color}{name_padded}{RESET}{BG_CHARCOAL}{weight}{FG_GRAY}{age_padded}{desc_sep}{display_text}{timeout_marker}{RESET}"
                    line = truncate_ansi(line, width)
                    line = bg_ljust(line, width, BG_CHARCOAL)
                else:
                    # Normal row - Format: icon name [badges]  age ago: description [timeout]
                    line = f"{color}{icon}{RESET} {weight}{color}{name_padded}{RESET}{weight}{FG_GRAY}{age_padded}{desc_sep}{display_text}{timeout_marker}{RESET}"
                    line = truncate_ansi(line, width)

                lines.append(line)

            # Add scroll indicators if needed (indicator stays at edge, cursor moves if conflict)
            if total_instances > instance_rows:
                # If cursor will conflict with indicator, move cursor line first
                if visible_start > 0 and self.state.cursor == visible_start:
                    # Save cursor line (at position 0), move to position 1
                    cursor_line = lines[0]
                    lines[0] = lines[1] if len(lines) > 1 else ""
                    if len(lines) > 1:
                        lines[1] = cursor_line

                if visible_end < total_instances and self.state.cursor == visible_end - 1:
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

        # Messages - Slack-style format with sender on separate line
        if self.state.messages:
            all_wrapped_lines = []

            # Get instance read positions for read receipt calculation
            instance_reads = {}
            try:
                from ..core.db import get_db
                conn = get_db()
                rows = conn.execute("SELECT name, last_event_id FROM instances WHERE previously_enabled = 1").fetchall()
                instance_reads = {row['name']: row['last_event_id'] for row in rows}
            except Exception:
                pass  # No read receipts if DB query fails

            for time_str, sender, message, recipients, event_id in self.state.messages:
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

                # Build recipient list with read receipts (width-aware truncation)
                recipient_str = ""
                if recipients:
                    # Calculate available width for recipients
                    # Format: "HH:MM sender → recipients"
                    base_len = len(display_time) + 1 + len(sender) + 3  # +1 space, +3 for " → "
                    available = width - base_len - 5  # Reserve 5 chars for potential "+N more"

                    recipient_parts = []
                    current_len = 0
                    shown = 0

                    for recipient in recipients:
                        # Check if recipient has read this message
                        has_read = instance_reads.get(recipient, 0) >= event_id
                        tick = " ✓" if has_read else ""
                        part = f"{recipient}{tick}"

                        # Calculate length with separator
                        part_len = ansi_len(part) + (2 if shown > 0 else 0)  # +2 for ", "

                        if current_len + part_len <= available:
                            recipient_parts.append(part)
                            current_len += part_len
                            shown += 1
                        else:
                            break

                    if recipient_parts:
                        recipient_str = ", ".join(recipient_parts)
                        remaining = len(recipients) - shown
                        if remaining > 0:
                            recipient_str += f" {FG_GRAY}+{remaining} more{RESET}"
                        recipient_str = f" {FG_GRAY}→{RESET} {recipient_str}"

                # Header line: timestamp + sender + recipients (truncated to width)
                header = f"{FG_GRAY}{display_time}{RESET} {BOLD}{sender}{RESET}{recipient_str}"
                header = truncate_ansi(header, width)
                all_wrapped_lines.append(header)

                # Replace literal newlines with space for preview
                display_message = message.replace('\n', ' ')

                # Bold @mentions in message (e.g., @name becomes **@name**)
                if '@' in display_message:
                    display_message = re.sub(r'(@[\w\-_]+)', f'{BOLD}\\1{RESET}{FG_LIGHTGRAY}', display_message)

                # Message lines with indent (6 spaces for visual balance)
                indent = '      '
                max_msg_len = width - len(indent)

                # Wrap message text
                if max_msg_len > 0:
                    wrapper = AnsiTextWrapper(width=max_msg_len)
                    wrapped = wrapper.wrap(display_message)

                    # All message lines indented uniformly
                    for wrapped_line in wrapped:
                        line = f"{indent}{FG_LIGHTGRAY}{wrapped_line}{RESET}"
                        all_wrapped_lines.append(line)
                else:
                    # Fallback if width too small
                    all_wrapped_lines.append(f"{indent}{FG_LIGHTGRAY}{display_message[:width-len(indent)]}{RESET}")

                # Blank line after each message (for separation)
                all_wrapped_lines.append('')

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


    def handle_key(self, key: str):
        """Handle keys in Manage mode"""
        # Sort by creation time (same as display) - stable, no jumping
        sorted_instances = sorted(
            self.state.instances.items(),
            key=lambda x: -x[1]['data'].get('created_at', 0.0)
        )

        if key == 'UP':
            if sorted_instances and self.state.cursor > 0:
                self.state.cursor -= 1
                # Update tracked instance name
                if self.state.cursor < len(sorted_instances):
                    self.state.cursor_instance_name = sorted_instances[self.state.cursor][0]
                self.tui.clear_all_pending_confirmations()
                self.sync_scroll_to_cursor()
        elif key == 'DOWN':
            if sorted_instances and self.state.cursor < len(sorted_instances) - 1:
                self.state.cursor += 1
                # Update tracked instance name
                if self.state.cursor < len(sorted_instances):
                    self.state.cursor_instance_name = sorted_instances[self.state.cursor][0]
                self.tui.clear_all_pending_confirmations()
                self.sync_scroll_to_cursor()
        elif key == '@':
            self.tui.clear_all_pending_confirmations()
            # Add @mention of highlighted instance at cursor position
            if sorted_instances and self.state.cursor < len(sorted_instances):
                name, _ = sorted_instances[self.state.cursor]
                mention = f"@{name} "
                if mention not in self.state.message_buffer:
                    self.state.message_buffer, self.state.message_cursor_pos = text_input_insert(
                        self.state.message_buffer, self.state.message_cursor_pos, mention
                    )
        elif key == 'SPACE':
            self.tui.clear_all_pending_confirmations()
            # Add space to message buffer at cursor position
            self.state.message_buffer, self.state.message_cursor_pos = text_input_insert(
                self.state.message_buffer, self.state.message_cursor_pos, ' '
            )
        elif key == 'LEFT':
            self.tui.clear_all_pending_confirmations()
            # Move cursor left in message buffer
            self.state.message_cursor_pos = text_input_move_left(self.state.message_cursor_pos)
        elif key == 'RIGHT':
            self.tui.clear_all_pending_confirmations()
            # Move cursor right in message buffer
            self.state.message_cursor_pos = text_input_move_right(self.state.message_buffer, self.state.message_cursor_pos)
        elif key == 'ESC':
            # Clear message buffer first, then cancel all pending confirmations
            if self.state.message_buffer:
                self.state.message_buffer = ""
                self.state.message_cursor_pos = 0
            else:
                self.tui.clear_all_pending_confirmations()
        elif key == 'BACKSPACE':
            self.tui.clear_all_pending_confirmations()
            # Delete character before cursor in message buffer
            self.state.message_buffer, self.state.message_cursor_pos = text_input_backspace(
                self.state.message_buffer, self.state.message_cursor_pos
            )
        elif key == 'ENTER':
            # Clear stop all and reset confirmations (toggle handled separately below)
            self.tui.clear_pending_confirmations_except('toggle')

            # Smart Enter: send message if text exists, otherwise toggle instances
            if self.state.message_buffer.strip():
                # Send message using cmd_send for consistent validation and error handling
                try:
                    message = self.state.message_buffer.strip()
                    result = cmd_send([message])
                    if result == 0:
                        self.tui.flash("Sent")
                        # Clear message buffer and cursor
                        self.state.message_buffer = ""
                        self.state.message_cursor_pos = 0
                    else:
                        self.tui.flash_error("Send failed")
                except Exception as e:
                    self.tui.flash_error(f"Error: {str(e)}")
            else:
                # No message text - toggle instance with two-step confirmation
                if not sorted_instances or self.state.cursor >= len(sorted_instances):
                    return

                name, info = sorted_instances[self.state.cursor]
                enabled = info['data'].get('enabled', False)
                action = "start" if not enabled else "stop"

                # Get status color for name
                status = info.get('status', "unknown")
                color = STATUS_FG.get(status, FG_WHITE)

                # Check if confirming previous toggle
                if self.state.pending_toggle == name and (time.time() - self.state.pending_toggle_time) <= self.tui.CONFIRMATION_TIMEOUT:
                    # Execute toggle (confirmation received)
                    try:
                        if enabled:
                            cmd_stop([name])
                            self.tui.flash(f"Stopped hcom for {color}{name}{RESET}")
                            self.state.completed_toggle = name
                            self.state.completed_toggle_time = time.time()
                        else:
                            cmd_start([name])
                            self.tui.flash(f"Started hcom for {color}{name}{RESET}")
                            self.state.completed_toggle = name
                            self.state.completed_toggle_time = time.time()
                        self.tui.load_status()
                    except Exception as e:
                        self.tui.flash_error(f"Error: {str(e)}")
                    finally:
                        self.state.pending_toggle = None
                else:
                    # Show confirmation (first press) - 10s duration
                    self.state.pending_toggle = name
                    self.state.pending_toggle_time = time.time()
                    # Name with status color, action is plain text (no color clash)
                    name_colored = f"{color}{name}{FG_WHITE}"
                    self.tui.flash(f"Confirm {action} {name_colored}? (press Enter again)", duration=self.tui.CONFIRMATION_FLASH_DURATION, color='white')

        elif key == 'CTRL_A':
            # Check state before clearing
            is_confirming = self.state.pending_stop_all and (time.time() - self.state.pending_stop_all_time) <= self.tui.CONFIRMATION_TIMEOUT
            self.tui.clear_pending_confirmations_except('stop_all')

            # Two-step confirmation for stop all
            if is_confirming:
                # Execute stop all (confirmation received)
                self.tui.stop_all_instances()
                self.state.pending_stop_all = False
            else:
                # Show confirmation (first press) - 10s duration
                self.state.pending_stop_all = True
                self.state.pending_stop_all_time = time.time()
                self.tui.flash(f"{FG_WHITE}Confirm stop all instances? (press Ctrl+A again){RESET}", duration=self.tui.CONFIRMATION_FLASH_DURATION, color='white')

        elif key == 'CTRL_R':
            # Check state before clearing
            is_confirming = self.state.pending_reset and (time.time() - self.state.pending_reset_time) <= self.tui.CONFIRMATION_TIMEOUT
            self.tui.clear_pending_confirmations_except('reset')

            # Two-step confirmation for reset
            if is_confirming:
                # Execute reset (confirmation received)
                self.tui.reset_logs()
                self.state.pending_reset = False
            else:
                # Show confirmation (first press)
                self.state.pending_reset = True
                self.state.pending_reset_time = time.time()
                self.tui.flash(f"{FG_WHITE}Confirm clear & archive (conversation + instance list)? (press Ctrl+R again){RESET}", duration=self.tui.CONFIRMATION_FLASH_DURATION, color='white')

        elif key == '\n':
            # Handle pasted newlines - insert literally
            self.tui.clear_all_pending_confirmations()
            self.state.message_buffer, self.state.message_cursor_pos = text_input_insert(
                self.state.message_buffer, self.state.message_cursor_pos, '\n'
            )

        elif key and len(key) == 1 and key.isprintable():
            self.tui.clear_all_pending_confirmations()
            # Insert printable characters at cursor position
            self.state.message_buffer, self.state.message_cursor_pos = text_input_insert(
                self.state.message_buffer, self.state.message_cursor_pos, key
            )

    def calculate_layout(self, height: int, width: int) -> tuple[int, int, int]:
        """Calculate instance/message/input row allocation"""
        # Dynamic input area based on buffer size
        input_rows = calculate_text_input_rows(self.state.message_buffer, width)
        # Space budget
        separator_rows = 3  # One separator between instances and messages, one before input, one after input
        min_instance_rows = 3

        available = height - input_rows - separator_rows
        # Instance rows = num instances (capped at 60% of available)
        instance_count = len(self.state.instances)
        max_instance_rows = int(available * 0.6)
        instance_rows = max(min_instance_rows, min(instance_count, max_instance_rows))
        message_rows = available - instance_rows

        return instance_rows, message_rows, input_rows

    def sync_scroll_to_cursor(self):
        """Sync scroll position to cursor"""
        # Calculate visible rows using shared layout function
        width, rows = get_terminal_size()
        body_height = max(10, rows - 3)  # Header, flash, footer
        instance_rows, _, _ = self.calculate_layout(body_height, width)
        visible_instance_rows = instance_rows  # Full instance section is visible

        # Scroll up if cursor moved above visible window
        if self.state.cursor < self.state.instance_scroll_pos:
            self.state.instance_scroll_pos = self.state.cursor
        # Scroll down if cursor moved below visible window
        elif self.state.cursor >= self.state.instance_scroll_pos + visible_instance_rows:
            self.state.instance_scroll_pos = self.state.cursor - visible_instance_rows + 1

    def render_wrapped_input(self, width: int, input_rows: int) -> List[str]:
        """Render message input (delegates to shared helper)"""
        return render_text_input(
            self.state.message_buffer,
            self.state.message_cursor_pos,
            width,
            input_rows,
            prefix="> "
        )
