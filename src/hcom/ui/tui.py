"""Main TUI orchestration"""
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

# Import types
from .types import Mode, UIState
from .rendering import (
    ansi_len, ansi_ljust, truncate_ansi, get_terminal_size,
    get_message_pulse_colors,
)
from .input import KeyboardInput

# Import from shared and api
from ..shared import (
    # ANSI codes
    RESET, BOLD, DIM,
    FG_GREEN, FG_CYAN, FG_WHITE, FG_BLACK, FG_GRAY, FG_YELLOW, FG_RED, FG_ORANGE,
    BG_ORANGE, BG_CHARCOAL, BG_YELLOW,
    CLEAR_SCREEN, CURSOR_HOME, HIDE_CURSOR, SHOW_CURSOR,
    # Config
    DEFAULT_CONFIG_HEADER,
    # Status configuration
    STATUS_ORDER, STATUS_BG_MAP,
    # Utilities
    format_timestamp, get_status_counts,
    resolve_claude_args,
)
from ..api import (
    # Instance operations
    get_instance_status, should_show_in_watch,
    # Path utilities
    hcom_path, ensure_hcom_directories,
    # Configuration
    HcomConfig, reload_config,
    ConfigSnapshot, load_config_snapshot, save_config,
    dict_to_hcom_config, HcomConfigError,
    # Commands
    cmd_launch, cmd_start, cmd_stop, cmd_reset, cmd_send,
    # Utilities
    list_available_agents,
)

# Import screens
from .manage import ManageScreen
from .launch import LaunchScreen

# Import config from parent package
from . import CONFIG_DEFAULTS

# TUI Layout Constants
MESSAGE_PREVIEW_LIMIT = 100  # Keep last N messages in message preview


class HcomTUI:
    """Main TUI application - orchestration only"""

    # Confirmation timeout constants
    CONFIRMATION_TIMEOUT = 10.0  # State cleared after this
    CONFIRMATION_FLASH_DURATION = 10.0  # Flash duration matches timeout

    def __init__(self, hcom_dir: Path):
        self.hcom_dir = hcom_dir
        self.mode = Mode.MANAGE
        self.state = UIState()  # All shared state in one place

        # Runtime orchestrator fields (not in UIState)
        self.last_frame = []
        self.last_status_update = 0.0
        self.last_config_check = 0.0
        self.first_render = True

        # Screen instances (pass state + self)
        self.manage_screen = ManageScreen(self.state, self)
        self.launch_screen = LaunchScreen(self.state, self)

    def flash(self, msg: str, duration: float = 2.0, color: str = 'orange'):
        """Show temporary flash message

        Args:
            msg: Message text
            duration: Display time in seconds
            color: 'red', 'white', or 'orange' (default)
        """
        self.state.flash_message = msg
        self.state.flash_until = time.time() + duration
        self.state.flash_color = color

    def flash_error(self, msg: str, duration: float = 10.0):
        """Show error flash in red"""
        self.state.flash_message = msg
        self.state.flash_until = time.time() + duration
        self.state.flash_color = 'red'

    def parse_validation_errors(self, error_str: str):
        """Parse ValueError message from HcomConfig into field-specific errors"""
        self.state.validation_errors.clear()

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
                if 'HCOM_TIMEOUT' not in self.state.validation_errors:
                    self.state.validation_errors['HCOM_TIMEOUT'] = line
            elif 'subagent_timeout' in line_lower or 'subagent timeout' in line_lower:
                if 'HCOM_SUBAGENT_TIMEOUT' not in self.state.validation_errors:
                    self.state.validation_errors['HCOM_SUBAGENT_TIMEOUT'] = line
            elif 'terminal' in line_lower:
                if 'HCOM_TERMINAL' not in self.state.validation_errors:
                    self.state.validation_errors['HCOM_TERMINAL'] = line
            elif 'tag' in line_lower:
                if 'HCOM_TAG' not in self.state.validation_errors:
                    self.state.validation_errors['HCOM_TAG'] = line
            elif 'agent' in line_lower and 'subagent' not in line_lower:
                # Agent can have multiple errors - store first one
                if 'HCOM_AGENT' not in self.state.validation_errors:
                    self.state.validation_errors['HCOM_AGENT'] = line
            elif 'claude_args' in line_lower:
                if 'HCOM_CLAUDE_ARGS' not in self.state.validation_errors:
                    self.state.validation_errors['HCOM_CLAUDE_ARGS'] = line
            elif 'hints' in line_lower:
                if 'HCOM_HINTS' not in self.state.validation_errors:
                    self.state.validation_errors['HCOM_HINTS'] = line

    def clear_all_pending_confirmations(self):
        """Clear all pending confirmation states and flash if any were active"""
        had_pending = self.state.pending_toggle or self.state.pending_stop_all or self.state.pending_reset

        self.state.pending_toggle = None
        self.state.pending_stop_all = False
        self.state.pending_reset = False

        if had_pending:
            self.state.flash_message = None

    def clear_pending_confirmations_except(self, keep: str):
        """Clear all pending confirmations except the specified one ('toggle', 'stop_all', 'reset')"""
        had_pending = False

        if keep != 'toggle' and self.state.pending_toggle:
            self.state.pending_toggle = None
            had_pending = True
        if keep != 'stop_all' and self.state.pending_stop_all:
            self.state.pending_stop_all = False
            had_pending = True
        if keep != 'reset' and self.state.pending_reset:
            self.state.pending_reset = False
            had_pending = True

        if had_pending:
            self.state.flash_message = None

    def stop_all_instances(self):
        """Stop all enabled instances"""
        try:
            stopped_count = 0
            for name, info in self.state.instances.items():
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
        """Load instance status from DB (streamed, not all at once)"""
        from ..core.db import iter_instances

        # Stream instances from DB, filter using same logic as watch
        instances = {}
        for data in iter_instances():
            if should_show_in_watch(data):
                instances[data['name']] = data

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

        self.state.instances = new_instances
        self.state.status_counts = get_status_counts(self.state.instances)

    def save_launch_state(self):
        """Save launch form values to config.env via claude args parser"""
        # Phase 3: Save Claude args to HCOM_CLAUDE_ARGS in config.env
        try:
            # Load current spec
            claude_args_str = self.state.config_edit.get('HCOM_CLAUDE_ARGS', '')
            spec = resolve_claude_args(None, claude_args_str if claude_args_str else None)

            # System flag matches background mode
            system_flag = None
            system_value = None
            if self.state.launch_system_prompt:
                system_flag = "--system-prompt" if self.state.launch_background else "--append-system-prompt"
                system_value = self.state.launch_system_prompt
            else:
                system_value = ""

            # Update spec with form values
            spec = spec.update(
                background=self.state.launch_background,
                system_flag=system_flag,
                system_value=system_value,
                prompt=self.state.launch_prompt,  # Always pass value (empty string deletes)
            )

            # Persist to in-memory edits
            self.state.config_edit['HCOM_CLAUDE_ARGS'] = spec.to_env_string()

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
            claude_args_str = self.state.config_edit.get('HCOM_CLAUDE_ARGS', '')
            spec = resolve_claude_args(None, claude_args_str if claude_args_str else None)

            # Check for parse errors and surface them
            if spec.errors:
                self.flash_error(f"Parse error: {spec.errors[0]}")

            # Extract Claude-related fields from spec
            self.state.launch_background = spec.is_background
            self.state.launch_prompt = spec.positional_tokens[0] if spec.positional_tokens else ""

            # Extract system prompt (prefer user_system, fallback to user_append)
            if spec.user_system:
                self.state.launch_system_prompt = spec.user_system
            elif spec.user_append:
                self.state.launch_system_prompt = spec.user_append
            else:
                self.state.launch_system_prompt = ""

            # Initialize cursors to end of text for first-time navigation
            self.state.launch_prompt_cursor = len(self.state.launch_prompt)
            self.state.launch_system_prompt_cursor = len(self.state.launch_system_prompt)
        except Exception as e:
            # Failed to parse - use defaults and log warning
            sys.stderr.write(f"Warning: Failed to load launch state (using defaults): {e}\n")

    def load_config_from_file(self, *, raise_on_error: bool = False):
        """Load all vars from ~/.hcom/config.env into editable dict"""
        config_path = Path.home() / '.hcom' / 'config.env'
        try:
            snapshot = load_config_snapshot()
            combined: dict[str, str] = {}
            combined.update(snapshot.values)
            combined.update(snapshot.extras)
            self.state.config_edit = combined
            self.state.validation_errors.clear()
            # Track mtime for external change detection
            try:
                self.state.config_mtime = config_path.stat().st_mtime
            except FileNotFoundError:
                self.state.config_mtime = 0.0
        except Exception as e:
            if raise_on_error:
                raise
            sys.stderr.write(f"Warning: Failed to load config.env (using defaults): {e}\n")
            self.state.config_edit = dict(CONFIG_DEFAULTS)
            for line in DEFAULT_CONFIG_HEADER:
                stripped = line.strip()
                if stripped and not stripped.startswith('#') and '=' in line:
                    key, _, value = line.partition('=')
                    key = key.strip()
                    raw = value.strip()
                    if (raw.startswith('"') and raw.endswith('"')) or (raw.startswith("'") and raw.endswith("'")):
                        raw = raw[1:-1]
                    self.state.config_edit.setdefault(key, raw)
            self.state.config_mtime = 0.0

    def save_config_to_file(self):
        """Write current config edits back to ~/.hcom/config.env using canonical writer."""
        known_values = {key: self.state.config_edit.get(key, '') for key in CONFIG_DEFAULTS.keys()}
        extras = {
            key: value
            for key, value in self.state.config_edit.items()
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
            self.state.validation_errors.clear()
            for field, message in exc.errors.items():
                env_key = field_map.get(field, field.upper())
                self.state.validation_errors[env_key] = message
            first_error = next(iter(self.state.validation_errors.values()), "Invalid config")
            self.flash_error(first_error)
            return
        except Exception as exc:
            self.flash_error(f"Validation error: {exc}")
            return

        try:
            save_config(core, extras)
            self.state.validation_errors.clear()
            self.state.flash_message = None
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

        if mtime <= self.state.config_mtime:
            return  # No change

        # Save what's currently being edited
        active_field = self.launch_screen.get_current_field_info()

        # Backup current edits
        old_edit = dict(self.state.config_edit)

        # Reload from disk
        try:
            self.load_config_from_file()
            self.load_launch_state()
        except Exception as exc:
            self.flash_error(f"Failed to reload config.env: {exc}")
            return

        # Update mtime
        try:
            self.state.config_mtime = config_path.stat().st_mtime
        except FileNotFoundError:
            self.state.config_mtime = 0.0

        # Restore in-progress edit if field changed externally
        if active_field and active_field[0]:
            key, value, cursor = active_field
            # Check if the field we're editing changed externally
            if key in old_edit and old_edit.get(key) != self.state.config_edit.get(key):
                # External change to field you're editing - keep your version
                self.state.config_edit[key] = value
                if key in self.state.config_field_cursors:
                    self.state.config_field_cursors[key] = cursor
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
        if self.state.pending_toggle and (now - self.state.pending_toggle_time) > self.CONFIRMATION_TIMEOUT:
            self.state.pending_toggle = None

        # Clear completed toggle display after 2s (match flash default)
        if self.state.completed_toggle and (now - self.state.completed_toggle_time) >= 2.0:
            self.state.completed_toggle = None

        # Clear pending stop all after timeout
        if self.state.pending_stop_all and (now - self.state.pending_stop_all_time) > self.CONFIRMATION_TIMEOUT:
            self.state.pending_stop_all = False

        # Clear pending reset after timeout
        if self.state.pending_reset and (now - self.state.pending_reset_time) > self.CONFIRMATION_TIMEOUT:
            self.state.pending_reset = False

        # Periodic config reload check (only in Launch mode)
        if self.mode == Mode.LAUNCH:
            if not hasattr(self, 'last_config_check'):
                self.last_config_check = 0.0

            if (now - self.last_config_check) >= 0.5:
                self.last_config_check = now
                self.check_external_config_changes()

        # Load messages for MANAGE screen preview (with event ID caching)
        if self.mode == Mode.MANAGE:
            from ..core.db import get_last_event_id, get_events_since
            import json

            try:
                current_max_id = get_last_event_id()
                if current_max_id != self.state.last_event_id:
                    events = get_events_since(self.state.last_event_id, event_type='message')
                    new_messages = []
                    for e in events:
                        event_data = e['data']  # Already a dict from db.py
                        new_messages.append((e['timestamp'], event_data.get('from', ''), event_data.get('text', '')))

                    # Append new messages and keep last N
                    all_messages = list(self.state.messages) + new_messages
                    self.state.messages = all_messages[-MESSAGE_PREVIEW_LIMIT:] if len(all_messages) > MESSAGE_PREVIEW_LIMIT else all_messages

                    # Update last message time for LOG tab pulse
                    if all_messages:
                        last_msg_timestamp = all_messages[-1][0]
                        try:
                            from datetime import datetime
                            if 'T' in last_msg_timestamp:
                                dt = datetime.fromisoformat(last_msg_timestamp.replace('Z', '+00:00'))
                                self.state.last_message_time = dt.timestamp()
                        except Exception:
                            self.state.last_message_time = 0.0
                    else:
                        self.state.last_message_time = 0.0

                    self.state.last_event_id = current_max_id
            except Exception:
                # DB query failed - keep existing messages
                pass

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
        if self.state.last_message_time > 0:
            seconds_since_msg = time.time() - self.state.last_message_time
        else:
            seconds_since_msg = 9999.0  # No messages yet - use quiet state
        log_bg_color, log_fg_color = get_message_pulse_colors(seconds_since_msg)

        # Build status display (colored blocks for unselected, orange for selected)
        is_manage_selected = (highlight_tab == "MANAGE")
        status_parts = []

        # Use shared status configuration (background colors for statusline blocks)
        for status_type in STATUS_ORDER:
            count = self.state.status_counts.get(status_type, 0)
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
        if self.state.flash_message and time.time() < self.state.flash_until:
            color_map = {
                'red': FG_RED,
                'white': FG_WHITE,
                'orange': FG_ORANGE
            }
            color_code = color_map.get(self.state.flash_color, FG_ORANGE)
            cols, _ = get_terminal_size()
            # Reserve space for "• " prefix and separator/padding
            max_msg_width = cols - 10
            msg = truncate_ansi(self.state.flash_message, max_msg_width) if len(self.state.flash_message) > max_msg_width else self.state.flash_message
            return f"{BOLD}{color_code}• {msg}{RESET}"
        return None

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
            manage_lines = self.manage_screen.build(body_rows, cols)
            for line in manage_lines:
                frame.append(ansi_ljust(line, cols))
        elif self.mode == Mode.LAUNCH:
            form_lines = self.launch_screen.build(body_rows, cols)
            for line in form_lines:
                frame.append(ansi_ljust(line, cols))

        # Footer - compact help text
        if self.mode == Mode.MANAGE:
            # Contextual footer based on state
            if self.state.message_buffer.strip():
                footer = f"{FG_GRAY}tab: switch  @: mention  enter: send  esc: clear{RESET}"
            elif self.state.pending_stop_all:
                footer = f"{FG_GRAY}ctrl+a: confirm stop all  esc: cancel{RESET}"
            elif self.state.pending_reset:
                footer = f"{FG_GRAY}ctrl+r: confirm reset  esc: cancel{RESET}"
            elif self.state.pending_toggle:
                footer = f"{FG_GRAY}enter: confirm  esc: cancel{RESET}"
            else:
                footer = f"{FG_GRAY}tab: switch  @: mention  enter: toggle  ctrl+a: stop all  ctrl+r: reset{RESET}"
        elif self.mode == Mode.LAUNCH:
            footer = self.launch_screen.get_footer()
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

        def redraw_all():
            """Redraw entire log and status (on entry or resize)"""
            from ..core.db import get_events_since
            import json

            # Clear screen
            sys.stdout.write('\033[2J\033[H')
            sys.stdout.flush()

            # Dump existing log with formatting
            has_messages = False
            try:
                events = get_events_since(0, event_type='message')
                has_messages = bool(events)
                if events:
                    for event in events:
                        event_data = event['data']  # Already a dict from db.py
                        msg = {
                            'timestamp': event['timestamp'],
                            'from': event_data.get('from', ''),
                            'message': event_data.get('text', '')
                        }
                        self.render_log_message(msg)
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
            from ..core.db import get_last_event_id
            return get_last_event_id(), cols

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
                from ..core.db import get_last_event_id, get_events_since
                import json

                try:
                    current_max_id = get_last_event_id()
                    if current_max_id > last_pos:
                        events = get_events_since(last_pos, event_type='message')
                        if events:
                            # Clear separator and status: move up to separator, clear it and status, return to position
                            sys.stdout.write('\r\033[A\033[K\n\033[K\033[A\r')

                            # Render new messages
                            for event in events:
                                event_data = event['data']  # Already a dict from db.py
                                msg = {
                                    'timestamp': event['timestamp'],
                                    'from': event_data.get('from', ''),
                                    'message': event_data.get('text', '')
                                }
                                self.render_log_message(msg)

                            # Redraw separator and status
                            self.render_status_with_separator("LOG")
                            has_messages_state = True  # We now have messages
                        last_pos = current_max_id
                except Exception:
                    # DB query failed - skip this update
                    pass

                time.sleep(0.01)

        # Return to TUI
        sys.stdout.write(HIDE_CURSOR + '\033[?1049h')
        sys.stdout.flush()

    def handle_key(self, key: str):
        """Handle key press based on current mode"""
        if self.mode == Mode.MANAGE:
            self.manage_screen.handle_key(key)
        elif self.mode == Mode.LAUNCH:
            self.launch_screen.handle_key(key)
