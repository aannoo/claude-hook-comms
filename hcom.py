#!/usr/bin/env python3
"""
hcom
CLI tool for launching multiple Claude Code terminals with interactive subagents, headless persistence, and real-time communication via hooks
"""

import os
import sys
import json
import io
import tempfile
import shutil
import shlex
import re
import subprocess
import time
import select
import platform
import random
from pathlib import Path
from datetime import datetime, timedelta
from typing import Any, Callable, NamedTuple, TextIO
from dataclasses import dataclass

if sys.version_info < (3, 10):
    sys.exit("Error: hcom requires Python 3.10 or higher")

__version__ = "0.5.0"

# ==================== Constants ====================

IS_WINDOWS = sys.platform == 'win32'

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


# Windows API constants
CREATE_NO_WINDOW = 0x08000000  # Prevent console window creation

# Timing constants
FILE_RETRY_DELAY = 0.01  # 10ms delay for file lock retries
STOP_HOOK_POLL_INTERVAL = 0.1     # 100ms between stop hook polls

MENTION_PATTERN = re.compile(r'(?<![a-zA-Z0-9._-])@(\w+)')
AGENT_NAME_PATTERN = re.compile(r'^[a-z-]+$')
TIMESTAMP_SPLIT_PATTERN = re.compile(r'\n(?=\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+\|)')

RESET = "\033[0m"
DIM = "\033[2m"
BOLD = "\033[1m"
FG_GREEN = "\033[32m"
FG_CYAN = "\033[36m"
FG_WHITE = "\033[37m"
FG_BLACK = "\033[30m"
BG_BLUE = "\033[44m"
BG_GREEN = "\033[42m"
BG_CYAN = "\033[46m"
BG_YELLOW = "\033[43m"
BG_RED = "\033[41m"
BG_GRAY = "\033[100m"

STATUS_MAP = {
    "waiting": (BG_BLUE, "◉"),
    "delivered": (BG_CYAN, "▷"),
    "active": (BG_GREEN, "▶"),
    "blocked": (BG_YELLOW, "■"),
    "inactive": (BG_RED, "○"),
    "unknown": (BG_GRAY, "○")
}

# Map status events to (display_category, description_template)
STATUS_INFO = {
    'session_start': ('active', 'started'),
    'tool_pending': ('active', '{} executing'),
    'waiting': ('waiting', 'idle'),
    'message_delivered': ('delivered', 'msg from {}'),
    'timeout': ('inactive', 'timeout'),
    'stopped': ('inactive', 'stopped'),
    'force_stopped': ('inactive', 'force stopped'),
    'started': ('active', 'starting'),
    'session_ended': ('inactive', 'ended: {}'),
    'blocked': ('blocked', '{} blocked'),
    'unknown': ('unknown', 'unknown'),
}

# ==================== Windows/WSL Console Unicode ====================

# Apply UTF-8 encoding for Windows and WSL
if IS_WINDOWS or is_wsl():
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    except (AttributeError, OSError):
        pass # Fallback if stream redirection fails

# ==================== Error Handling Strategy ====================
# Hooks: Must never raise exceptions (breaks hcom). Functions return True/False.
# CLI: Can raise exceptions for user feedback. Check return values.
# Critical I/O: atomic_write, save_instance_position
# Pattern: Try/except/return False in hooks, raise in CLI operations.

# ==================== CLI Errors ====================

class CLIError(Exception):
    """Raised when arguments cannot be mapped to command semantics."""

# ==================== Help Text ====================

HELP_TEXT = """hcom 0.5.0

Usage: [ENV_VARS] hcom <COUNT> [claude <ARGS>...]
       hcom watch [--logs|--status|--wait [SEC]]
       hcom send "message"
       hcom stop [alias|all] [--force]
       hcom start [alias]
       hcom reset [logs|hooks|config]

Launch Examples:
  hcom 3             Open 3 terminals with claude connected to hcom
  hcom 3 claude -p                            + Background/headless
  HCOM_TAG=api hcom 3 claude -p               + @-mention group tag
  claude 'run hcom start'    claude code with prompt will also work

Commands:
  watch              Interactive messaging/status dashboard
    --logs           Print all messages
    --status         Print instance status JSON
    --wait [SEC]     Wait and notify for new message

  send "msg"         Send message to all instances
  send "@alias msg"  Send to specific instance/group

  stop               Stop current instance (from inside Claude)
  stop <alias>       Stop specific instance
  stop all           Stop all instances
    --force          Emergency stop (denies Bash tool)

  start              Start current instance (from inside Claude)
  start <alias>      Start specific instance

  reset              Stop all + archive logs + remove hooks + clear config
  reset logs         Clear + archive conversation log
  reset hooks        Safely remove hcom hooks from claude settings.json
  reset config       Clear + backup config.env

Environment Variables:
  HCOM_TAG=name      Group tag (creates name-* instances)
  HCOM_AGENT=type    Agent type (comma-separated for multiple)
  HCOM_TERMINAL=mode Terminal: new|here|print|"custom {script}"
  HCOM_PROMPT=text   "Say hi in hcom chat" (default)
  HCOM_HINTS=text    Text appended to all messages received by instance
  HCOM_TIMEOUT=secs  Time until disconnected from hcom chat (default 1800s / 30mins)

Config: ~/.hcom/config.env
Docs: https://github.com/aannoo/claude-hook-comms"""


# ==================== Logging ====================

def log_hook_error(hook_name: str, error: Exception | str | None = None) -> None:
    """Log hook exceptions or just general logging to ~/.hcom/scripts/hooks.log for debugging"""
    import traceback
    try:
        log_file = hcom_path(SCRIPTS_DIR) / "hooks.log"
        timestamp = datetime.now().isoformat()
        if error and isinstance(error, Exception):
            tb = ''.join(traceback.format_exception(type(error), error, error.__traceback__))
            with open(log_file, 'a') as f:
                f.write(f"{timestamp}|{hook_name}|{type(error).__name__}: {error}\n{tb}\n")
        else:
            with open(log_file, 'a') as f:
                f.write(f"{timestamp}|{hook_name}|{error or 'checkpoint'}\n")
    except (OSError, PermissionError):
        pass  # Silent failure in error logging

# ==================== Config Defaults ====================
# Config precedence: env var > ~/.hcom/config.env > defaults
# All config via HcomConfig dataclass (timeout, terminal, prompt, hints, tag, agent)

# Constants (not configurable)
MAX_MESSAGE_SIZE = 1048576  # 1MB
MAX_MESSAGES_PER_DELIVERY = 50
SENDER = 'bigboss'
SENDER_EMOJI = '🐳'
SKIP_HISTORY = True  # New instances start at current log position (skip old messages)

# Path constants
LOG_FILE = "hcom.log"
INSTANCES_DIR = "instances"
LOGS_DIR = ".tmp/logs"
SCRIPTS_DIR = ".tmp/scripts"
FLAGS_DIR = ".tmp/flags"
CONFIG_FILE = "config.env"
ARCHIVE_DIR = "archive"

# Hook configuration - single source of truth for setup_hooks() and verify_hooks_installed()
# Format: (hook_type, matcher, command_suffix, timeout)
# Command gets built as: hook_cmd_base + ' ' + command_suffix (e.g., '${HCOM} poll')
HOOK_CONFIGS = [
    ('SessionStart', '', 'sessionstart', None),
    ('UserPromptSubmit', '', 'userpromptsubmit', None),
    ('PreToolUse', 'Bash', 'pre', None),
    ('PostToolUse', 'Bash', 'post', None),   # Match Bash only
    ('Stop', '', 'poll', 86400),          # Poll for messages (24hr max timeout)
    ('Notification', '', 'notify', None),
    ('SessionEnd', '', 'sessionend', None),
]

# Derived from HOOK_CONFIGS - guaranteed to stay in sync
ACTIVE_HOOK_TYPES = [cfg[0] for cfg in HOOK_CONFIGS]
HOOK_COMMANDS = [cfg[2] for cfg in HOOK_CONFIGS]
LEGACY_HOOK_TYPES = ACTIVE_HOOK_TYPES
LEGACY_HOOK_COMMANDS = HOOK_COMMANDS

# Hook removal patterns - used by _remove_hcom_hooks_from_settings()
# Dynamically build from LEGACY_HOOK_COMMANDS to match current and legacy hook formats
_HOOK_ARGS_PATTERN = '|'.join(LEGACY_HOOK_COMMANDS)
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

# PreToolUse hook pattern - matches hcom commands for session_id injection and auto-approval
# - hcom send (any args)
# - hcom stop (no args) | hcom start (no args)
# - hcom help | hcom --help | hcom -h
# - hcom watch --status | hcom watch --launch | hcom watch --logs | hcom watch --wait
# Negative lookahead (?!\s+[-\w]) ensures stop/start not followed by arguments or flags
HCOM_COMMAND_PATTERN = re.compile(
    r'((?:uvx\s+)?hcom|(?:python3?\s+)?\S*hcom\.py)\s+'
    r'(?:send\b|(?:stop|start)(?!\s+[-\w])|(?:help|--help|-h)\b|watch\s+(?:--status|--launch|--logs|--wait)\b)'
)

# ==================== File System Utilities ====================

def hcom_path(*parts: str, ensure_parent: bool = False) -> Path:
    """Build path under ~/.hcom"""
    path = Path.home() / ".hcom"
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
        for dir_name in [INSTANCES_DIR, LOGS_DIR, SCRIPTS_DIR, FLAGS_DIR, ARCHIVE_DIR]:
            hcom_path(dir_name).mkdir(parents=True, exist_ok=True)
        return True
    except (OSError, PermissionError):
        return False

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

def get_instance_file(instance_name: str) -> Path:
    """Get path to instance's position file with path traversal protection"""
    # Sanitize instance name to prevent directory traversal
    if not instance_name:
        instance_name = "unknown"
    safe_name = instance_name.replace('..', '').replace('/', '-').replace('\\', '-').replace(os.sep, '-')
    if not safe_name:
        safe_name = "unknown"

    return hcom_path(INSTANCES_DIR, f"{safe_name}.json")

def load_instance_position(instance_name: str) -> dict[str, Any]:
    """Load position data for a single instance"""
    instance_file = get_instance_file(instance_name)

    data = read_file_with_retry(
        instance_file,
        lambda f: json.load(f),
        default={}
    )

    return data

def save_instance_position(instance_name: str, data: dict[str, Any]) -> bool:
    """Save position data for a single instance. Returns True on success, False on failure."""
    try:
        instance_file = hcom_path(INSTANCES_DIR, f"{instance_name}.json")
        return atomic_write(instance_file, json.dumps(data, indent=2))
    except (OSError, PermissionError, ValueError):
        return False

def get_claude_settings_path() -> Path:
    """Get path to global Claude settings file"""
    return Path.home() / '.claude' / 'settings.json'

def load_settings_json(settings_path: Path, default: Any = None) -> dict[str, Any] | None:
    """Load and parse settings JSON file with retry logic"""
    return read_file_with_retry(
        settings_path,
        lambda f: json.load(f),
        default=default
    )

def load_all_positions() -> dict[str, dict[str, Any]]:
    """Load positions from all instance files"""
    instances_dir = hcom_path(INSTANCES_DIR)
    if not instances_dir.exists():
        return {}

    positions = {}
    for instance_file in instances_dir.glob("*.json"):
        instance_name = instance_file.stem
        data = read_file_with_retry(
            instance_file,
            lambda f: json.load(f),
            default={}
        )
        if data:
            positions[instance_name] = data
    return positions

def clear_all_positions() -> None:
    """Clear all instance position files and related mapping files"""
    instances_dir = hcom_path(INSTANCES_DIR)
    if instances_dir.exists():
        for f in instances_dir.glob('*.json'):
            f.unlink()

# ==================== Configuration System ====================

@dataclass
class HcomConfig:
    """HCOM configuration with validation. Load priority: env → file → defaults"""
    timeout: int = 1800
    terminal: str = 'new'
    prompt: str = 'say hi in hcom chat'
    hints: str = ''
    tag: str = ''
    agent: str = ''

    def __post_init__(self):
        """Validate configuration on construction"""
        errors = self.validate()
        if errors:
            raise ValueError(f"Invalid config:\n" + "\n".join(f"  - {e}" for e in errors))

    def validate(self) -> list[str]:
        """Validate all fields, return list of errors"""
        errors = []

        # Validate timeout
        # Validate timeout (bool is subclass of int in Python, must check explicitly)
        if isinstance(self.timeout, bool):
            errors.append(f"timeout must be an integer, not boolean (got {self.timeout})")
        elif not isinstance(self.timeout, int):
            errors.append(f"timeout must be an integer, got {type(self.timeout).__name__}")
        elif not 1 <= self.timeout <= 86400:
            errors.append(f"timeout must be 1-86400 seconds (24 hours), got {self.timeout}")

        # Validate terminal
        if not isinstance(self.terminal, str):
            errors.append(f"terminal must be a string, got {type(self.terminal).__name__}")
        else:
            valid_modes = ('new', 'here', 'print')
            if self.terminal not in valid_modes and '{script}' not in self.terminal:
                errors.append(
                    f"terminal must be one of {valid_modes} or custom command with {{script}}, "
                    f"got '{self.terminal}'"
                )

        # Validate tag (only alphanumeric and hyphens - security: prevent log delimiter injection)
        if not isinstance(self.tag, str):
            errors.append(f"tag must be a string, got {type(self.tag).__name__}")
        elif self.tag and not re.match(r'^[a-zA-Z0-9-]+$', self.tag):
            errors.append("tag can only contain letters, numbers, and hyphens")

        # Validate agent
        if not isinstance(self.agent, str):
            errors.append(f"agent must be a string, got {type(self.agent).__name__}")

        return errors

    @classmethod
    def load(cls) -> 'HcomConfig':
        """Load config with precedence: env var → file → defaults"""
        # Ensure config file exists
        config_path = hcom_path(CONFIG_FILE, ensure_parent=True)
        created_config = False
        if not config_path.exists():
            _write_default_config(config_path)
            created_config = True

        # Warn once if legacy config.json still exists when creating config.env
        legacy_config = hcom_path('config.json')
        if created_config and legacy_config.exists():
            print(
                format_error(
                    "Found legacy ~/.hcom/config.json; new config file is: ~/.hcom/config.env."
                ),
                file=sys.stderr,
            )

        # Parse config file once
        file_config = _parse_env_file(config_path) if config_path.exists() else {}

        def get_var(key: str) -> str | None:
            """Get variable with precedence: env → file"""
            if key in os.environ:
                return os.environ[key]
            if key in file_config:
                return file_config[key]
            return None

        data = {}

        # Load timeout (requires int conversion)
        timeout_str = get_var('HCOM_TIMEOUT')
        if timeout_str is not None:
            try:
                data['timeout'] = int(timeout_str)
            except (ValueError, TypeError):
                pass  # Use default

        # Load string values
        terminal = get_var('HCOM_TERMINAL')
        if terminal is not None:
            data['terminal'] = terminal
        prompt = get_var('HCOM_PROMPT')
        if prompt is not None:
            data['prompt'] = prompt
        hints = get_var('HCOM_HINTS')
        if hints is not None:
            data['hints'] = hints
        tag = get_var('HCOM_TAG')
        if tag is not None:
            data['tag'] = tag
        agent = get_var('HCOM_AGENT')
        if agent is not None:
            data['agent'] = agent

        return cls(**data)  # Validation happens in __post_init__

def _parse_env_file(config_path: Path) -> dict[str, str]:
    """Parse ENV file (KEY=VALUE format) with security validation"""
    config = {}

    # Dangerous shell metacharacters that enable command injection
    DANGEROUS_CHARS = ['`', '$', ';', '|', '&', '\n', '\r']

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

                # Security: Validate HCOM_TERMINAL for command injection
                if key == 'HCOM_TERMINAL':
                    if any(c in value for c in DANGEROUS_CHARS):
                        print(
                            f"Warning: Unsafe characters in HCOM_TERMINAL "
                            f"({', '.join(repr(c) for c in DANGEROUS_CHARS if c in value)}), "
                            f"ignoring custom terminal command",
                            file=sys.stderr
                        )
                        continue
                    # Additional check: custom commands must contain {script} placeholder
                    if value not in ('new', 'here', 'print') and '{script}' not in value:
                        print(
                            f"Warning: HCOM_TERMINAL custom command must include {{script}} placeholder, "
                            f"ignoring",
                            file=sys.stderr
                        )
                        continue

                # Remove outer quotes only if they match
                if len(value) >= 2:
                    if (value[0] == value[-1]) and value[0] in ('"', "'"):
                        value = value[1:-1]
                if key:
                    config[key] = value
    except (FileNotFoundError, PermissionError, UnicodeDecodeError):
        pass
    return config

def _write_default_config(config_path: Path) -> None:
    """Write default config file with documentation"""
    header = """# HCOM Configuration
#
# All HCOM_* settings (and any env var ie. Claude Code settings)
# can be set here or via environment variables.
# Environment variables override config file values.
#
# HCOM settings:
#   HCOM_TIMEOUT - Instance Stop hook wait timeout in seconds (default: 1800)
#   HCOM_TERMINAL - Terminal mode: "new", "here", "print", or custom command with {script}
#   HCOM_PROMPT - Initial prompt for new instances (empty = no auto prompt)
#   HCOM_HINTS - Text appended to all messages received by instances
#   HCOM_TAG - Group tag for instances (creates tag-* instances)
#   HCOM_AGENT - Claude code subagent from .claude/agents/, comma-separated for multiple
#
# Put each value on separate lines without comments.
#
#
"""
    defaults = [
        'HCOM_TIMEOUT=1800',
        'HCOM_TERMINAL=new',
        'HCOM_PROMPT=say hi in hcom chat',
        'HCOM_HINTS=',
        'HCOM_TAG=',
        'HCOM_AGENT=',
    ]
    try:
        atomic_write(config_path, header + '\n'.join(defaults) + '\n')
    except Exception:
        pass

# Global config instance (cached)
_config: HcomConfig | None = None

def get_config() -> HcomConfig:
    """Get cached config, loading if needed"""
    global _config
    if _config is None:
        _config = HcomConfig.load()
    return _config

def _build_quoted_invocation() -> str:
    """Build properly quoted python + script path for current platform"""
    python_path = sys.executable
    script_path = str(Path(__file__).resolve())

    if IS_WINDOWS:
        if ' ' in python_path or ' ' in script_path:
            return f'"{python_path}" "{script_path}"'
        return f'{python_path} {script_path}'
    else:
        return f'{shlex.quote(python_path)} {shlex.quote(script_path)}'

def get_hook_command() -> tuple[str, dict[str, Any]]:
    """Get hook command - hooks always run, Python code gates participation

    Uses ${HCOM} environment variable set in settings.json, with fallback to direct python invocation.
    Participation is controlled by enabled flag in instance JSON files.
    """
    if IS_WINDOWS:
        # Windows: use python path directly
        return _build_quoted_invocation(), {}
    else:
        # Unix: Use HCOM env var from settings.json
        return '${HCOM}', {}

def _detect_hcom_command_type() -> str:
    """Detect how to invoke hcom based on execution context
    Priority:
    1. uvx - If running in uv-managed Python and uvx available
           (works for both temporary uvx runs and permanent uv tool install)
    2. short - If hcom binary in PATH
    3. full - Fallback to full python invocation
    """
    if 'uv' in Path(sys.executable).resolve().parts and shutil.which('uvx'):
        return 'uvx'
    elif shutil.which('hcom'):
        return 'short'
    else:
        return 'full'

def _parse_version(v: str) -> tuple:
    """Parse version string to comparable tuple"""
    return tuple(int(x) for x in v.split('.') if x.isdigit())

def get_update_notice() -> str | None:
    """Check PyPI for updates (once daily), return message if available"""
    flag = hcom_path(FLAGS_DIR, 'update_available')

    # Check PyPI if flag missing or >24hrs old
    should_check = not flag.exists() or time.time() - flag.stat().st_mtime > 86400

    if should_check:
        try:
            import urllib.request
            with urllib.request.urlopen('https://pypi.org/pypi/hcom/json', timeout=2) as f:
                latest = json.load(f)['info']['version']

            if _parse_version(latest) > _parse_version(__version__):
                atomic_write(flag, latest)  # mtime = cache timestamp
            else:
                flag.unlink(missing_ok=True)
                return None
        except Exception:
            pass  # Network error, use cached value if exists

    # Return message if update available
    if not flag.exists():
        return None

    try:
        latest = flag.read_text().strip()
        # Double-check version (handles manual upgrades)
        if _parse_version(__version__) >= _parse_version(latest):
            flag.unlink(missing_ok=True)
            return None

        cmd = "uv tool upgrade hcom" if _detect_hcom_command_type() == 'uvx' else "pip install -U hcom"
        return f"→ hcom v{latest} available: {cmd}"
    except Exception:
        return None

def _build_hcom_env_value() -> str:
    """Build the value for settings['env']['HCOM'] based on current execution context
    Uses build_hcom_command() without caching for fresh detection on every call.
    """
    return build_hcom_command(None)

def build_hcom_command(instance_name: str | None = None) -> str:
    """Build base hcom command - caches PATH check in instance file on first use"""
    # Determine command type (cached or detect)
    cmd_type = None
    if instance_name:
        data = load_instance_position(instance_name)
        if data.get('session_id'):
            if 'hcom_cmd_type' not in data:
                cmd_type = _detect_hcom_command_type()
                data['hcom_cmd_type'] = cmd_type
                save_instance_position(instance_name, data)
            else:
                cmd_type = data.get('hcom_cmd_type')

    if not cmd_type:
        cmd_type = _detect_hcom_command_type()

    # Build command based on type
    if cmd_type == 'short':
        return 'hcom'
    elif cmd_type == 'uvx':
        return 'uvx hcom'
    else:
        # Full path fallback
        return _build_quoted_invocation()

def build_send_command(example_msg: str = '', instance_name: str | None = None) -> str:
    """Build send command - caches PATH check in instance file on first use"""
    msg = f" '{example_msg}'" if example_msg else ''
    base_cmd = build_hcom_command(instance_name)
    return f'{base_cmd} send{msg}'

def build_claude_env() -> dict[str, str]:
    """Build environment variables for Claude instances

    Passes current environment to Claude, with config.env providing defaults.
    HCOM_* variables are filtered out (consumed by hcom, not passed to Claude).
    """
    env = {}

    # Read config file directly for Claude Code env vars (non-HCOM_ keys)
    config_path = hcom_path(CONFIG_FILE)
    if config_path.exists():
        file_config = _parse_env_file(config_path)
        for key, value in file_config.items():
            if not key.startswith('HCOM_'):
                env[key] = str(value)

    # Overlay with current environment (except HCOM_*)
    # This ensures user's shell environment is respected
    for key, value in os.environ.items():
        if not key.startswith('HCOM_'):
            env[key] = value

    return env

# ==================== Message System ====================

def validate_message(message: str) -> str | None:
    """Validate message size and content. Returns error message or None if valid."""
    if not message or not message.strip():
        return format_error("Message required")

    # Reject control characters (except \n, \r, \t)
    if re.search(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\u0080-\u009F]', message):
        return format_error("Message contains control characters")

    if len(message) > MAX_MESSAGE_SIZE:
        return format_error(f"Message too large (max {MAX_MESSAGE_SIZE} chars)")

    return None

def send_message(from_instance: str, message: str) -> bool:
    """Send a message to the log"""
    try:
        log_file = hcom_path(LOG_FILE)

        escaped_message = message.replace('|', '\\|')
        escaped_from = from_instance.replace('|', '\\|')

        timestamp = datetime.now().isoformat()
        line = f"{timestamp}|{escaped_from}|{escaped_message}\n"

        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(line)
            f.flush()

        return True
    except Exception:
        return False

def build_hcom_bootstrap_text(instance_name: str) -> str:
    """Build comprehensive HCOM bootstrap context for instances"""
    hcom_cmd = build_hcom_command(instance_name=instance_name)

    # Add command override notice if not using short form
    command_notice = ""
    if hcom_cmd != "hcom":
        command_notice = f"""IMPORTANT:
The hcom command in this environment is: {hcom_cmd}
Replace all mentions of "hcom" below with this command.

"""

    # Add tag-specific notice if instance is tagged
    tag = get_config().tag
    tag_notice = ""
    if tag:
        tag_notice = f"""
GROUP TAG: You are in the '{tag}' group.
- To message your group: hcom send "@{tag} your message"
- Group messages are targeted - only instances with an alias starting with {tag}-* receive them
- If someone outside the group sends you @{tag} messages, they won't see your @{tag} replies. To reply to non-group members, either @mention them directly or broadcast.
"""


    return f"""{command_notice}{tag_notice}
[HCOM SESSION CONFIG]
- HCOM is a communication tool for you, other claude code instances, and the human user.
- Your HCOM alias for this session: {instance_name}

Your HCOM Tools:
- hcom send "msg" (broadcast) / "@alias msg" (direct) / "@tag msg" (tag) - for you
- hcom watch --status  → See participants JSON (for you only)
- hcom watch --launch   → Open interactive TUI messaging+status dashboard in new terminal (for the human user)
- hcom start/stop   → Connect/disconnect from chat (you run these, user can't run it themselves unless they specify an alias)
- hcom <count>  → Launch instances in new terminal (you MUST run 'hcom help' first to understand context)

UI Separation:
- The human user has 'hcom watch' (interactive TUI dashboard).
- You use 'hcom watch --launch' to open it for them (offer to do so).
- In conversation, call it "hcom watch" (the command they'd run themselves).

Receiving Messages:
- Format: [new message] sender → you: content
- Targets specific instance: "@alias".
- Targets all api-* tagged instances: "@api message".
- Arrives automatically via hooks/bash. No proactive checking needed.
- Stop hook feedback shows: {{"decision": "block"}} (this is normal operation).

Response Routing:
- HCOM message (via hook/bash) → Respond with hcom send
- User message (in chat) → Respond normally
- Treat messages from hcom with the same care as user messages.
- Authority: Prioritize @{SENDER} over other participants.

This is context for YOUR hcom session config. The human user cannot see this config text (but they can see subsequent hcom messages you receive).
On connection, tell the human user about only these commands: 'hcom <count>', 'hcom watch', 'hcom start', 'hcom stop'
Report to the human user using first-person, for example: "I'm connected to HCOM as {instance_name}, cool!"
"""

def build_launch_context(instance_name: str) -> str:
    """Build context for launch command"""
    return f"""[HCOM LAUNCH INFORMATION]
BASIC USAGE:
[ENV_VARS] hcom <COUNT> [claude <ARGS>...]
- directory-specific (always cd to project directory first)
- default to foreground instances unless told otherwise/good reason to do bg
- Everyone shares the same conversation log, isolation is possible with tags and at-mentions.

ENV VARS INFO:
- YOU cannot use 'HCOM_TERMINAL=here' - Claude cannot launch claude within itself, must be in a new or custom terminal
- HCOM_AGENT(s) are custom system prompt files created by users/Claude beforehand.
- HCOM_AGENT(s) load from .claude/agents/<name>.md if they have been created

KEY CLAUDE ARGS:
Run 'claude --help' for all claude code CLI args. hcom 1 claude [options] [command] [prompt]
-p           background/headless instance
--allowedTools=Bash   (background can only hcom chat otherwise, 'claude help' for more tools)
--model sonnet/haiku/opus
--resume <sessionid>       (get sessionid from hcom watch --status)
--system-prompt (for foreground instances) --append-system-prompt (for background instances)
Example: HCOM_HINTS='essential responses only' hcom 2 claude --model sonnet -p "do task x"

CONTROL: 
  hcom watch --status    JSON status of all instances
  hcom watch --logs      All messages (pipe to tail)
  hcom watch --wait      Block until next message (only use when hcom stopped (started is automatic already))

STATUS INDICATORS:
  "active", "delivered" | "idle" - waiting for new messages
  "blocked" - permission request (needs user approval)
  "inactive" - timed out, disconnected etc
  "unknown" / "stale" - could be dead

LAUNCH PATTERNS:
- HCOM_AGENT=reviewer,tester hcom 2 claude "do task x"  # 2x reviewers + 2x testers (4 in total) with initial prompt
- clone with same context:
    1. hcom 1 then hcom send 'analyze api' then hcom watch --status (get sessionid)
    2. HCOM_TAG=clone hcom 3 claude --resume sessionid
- System prompt (or agent file) + initial prompt + hcom_hints is a powerful combination.

"""

def should_deliver_message(msg: dict[str, str], instance_name: str, all_instance_names: list[str] | None = None) -> bool:
    """Check if message should be delivered based on @-mentions"""
    text = msg['message']
    
    if '@' not in text:
        return True
    
    mentions = MENTION_PATTERN.findall(text)
    
    if not mentions:
        return True
    
    # Check if this instance matches any mention
    this_instance_matches = any(instance_name.lower().startswith(mention.lower()) for mention in mentions)
    
    if this_instance_matches:
        return True
    
    # Check if any mention is for the CLI sender (bigboss)
    sender_mentioned = any(SENDER.lower().startswith(mention.lower()) for mention in mentions)
    
    # If we have all_instance_names, check if ANY mention matches ANY instance or sender
    if all_instance_names:
        any_mention_matches = any(
            any(name.lower().startswith(mention.lower()) for name in all_instance_names)
            for mention in mentions
        ) or sender_mentioned
        
        if not any_mention_matches:
            return True  # No matches anywhere = broadcast to all
    
    return False  # This instance doesn't match, but others might

# ==================== Parsing & Utilities ====================

def extract_agent_config(content: str) -> dict[str, str]:
    """Extract configuration from agent YAML frontmatter"""
    if not content.startswith('---'):
        return {}
    
    # Find YAML section between --- markers
    if (yaml_end := content.find('\n---', 3)) < 0:
        return {}  # No closing marker
    
    yaml_section = content[3:yaml_end]
    config = {}
    
    # Extract model field
    if model_match := re.search(r'^model:\s*(.+)$', yaml_section, re.MULTILINE):
        value = model_match.group(1).strip()
        if value and value.lower() != 'inherit':
            config['model'] = value

    # Extract tools field
    if tools_match := re.search(r'^tools:\s*(.+)$', yaml_section, re.MULTILINE):
        value = tools_match.group(1).strip()
        if value:
            config['tools'] = value.replace(', ', ',')
    
    return config

def resolve_agent(name: str) -> tuple[str, dict[str, str]]:
    """Resolve agent file by name with validation.
    Looks for agent files in:
    1. .claude/agents/{name}.md (local)
    2. ~/.claude/agents/{name}.md (global)
    Returns tuple: (content without YAML frontmatter, config dict)
    """
    hint = 'Agent names must use lowercase letters and dashes only'

    if not isinstance(name, str):
        raise FileNotFoundError(format_error(
            f"Agent '{name}' not found",
            hint
        ))

    candidate = name.strip()
    display_name = candidate or name

    if not candidate or not AGENT_NAME_PATTERN.fullmatch(candidate):
        raise FileNotFoundError(format_error(
            f"Agent '{display_name}' not found",
            hint
        ))

    for base_path in (Path.cwd(), Path.home()):
        agents_dir = base_path / '.claude' / 'agents'
        try:
            agents_dir_resolved = agents_dir.resolve(strict=True)
        except FileNotFoundError:
            continue

        agent_path = agents_dir / f'{candidate}.md'
        if not agent_path.exists():
            continue

        try:
            resolved_agent_path = agent_path.resolve(strict=True)
        except FileNotFoundError:
            continue

        try:
            resolved_agent_path.relative_to(agents_dir_resolved)
        except ValueError:
            continue

        content = read_file_with_retry(
            agent_path,
            lambda f: f.read(),
            default=None
        )
        if content is None:
            continue

        config = extract_agent_config(content)
        stripped = strip_frontmatter(content)
        if not stripped.strip():
            raise ValueError(format_error(
                f"Agent '{candidate}' has empty content",
                'Check the agent file is a valid format and contains text'
            ))
        return stripped, config

    raise FileNotFoundError(format_error(
        f"Agent '{candidate}' not found in project or user .claude/agents/ folder",
        'Check available agents or create the agent file'
    ))

def strip_frontmatter(content: str) -> str:
    """Strip YAML frontmatter from agent file"""
    if content.startswith('---'):
        # Find the closing --- on its own line
        lines = content.splitlines()
        for i, line in enumerate(lines[1:], 1):
            if line.strip() == '---':
                return '\n'.join(lines[i+1:]).strip()
    return content

def get_display_name(session_id: str | None, tag: str | None = None) -> str:
    """Get display name for instance using session_id"""
    # 50 most recognizable 3-letter words
    words = [
        'ace', 'air', 'ant', 'arm', 'art', 'axe', 'bad', 'bag', 'bar', 'bat',
        'bed', 'bee', 'big', 'box', 'boy', 'bug', 'bus', 'cab', 'can', 'cap',
        'car', 'cat', 'cop', 'cow', 'cry', 'cup', 'cut', 'day', 'dog', 'dry',
        'ear', 'egg', 'eye', 'fan', 'fin', 'fly', 'fox', 'fun', 'gem', 'gun',
        'hat', 'hit', 'hot', 'ice', 'ink', 'jet', 'key', 'law', 'map', 'mix',
    ]

    # Use session_id directly instead of extracting UUID from transcript
    if session_id:
        # Hash to select word
        hash_val = sum(ord(c) for c in session_id)
        word = words[hash_val % len(words)]

        # Add letter suffix that flows naturally with the word
        last_char = word[-1]
        if last_char in 'aeiou':
            # After vowel: s/n/r/l creates plural/noun/verb patterns
            suffix_options = 'snrl'
        else:
            # After consonant: add vowel or y for pronounceability
            suffix_options = 'aeiouy'

        letter_hash = sum(ord(c) for c in session_id[1:]) if len(session_id) > 1 else hash_val
        suffix = suffix_options[letter_hash % len(suffix_options)]

        base_name = f"{word}{suffix}"
        collision_attempt = 0

        # Collision detection: keep adding words until unique
        while True:
            instance_file = hcom_path(INSTANCES_DIR, f"{base_name}.json")
            if not instance_file.exists():
                break  # Name is unique

            try:
                with open(instance_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                their_session_id = data.get('session_id', '')

                # Same session_id = our file, reuse name
                if their_session_id == session_id:
                    break
                # No session_id = stale/malformed file, use name
                if not their_session_id:
                    break

                # Real collision - add another word
                collision_hash = sum(ord(c) * (i + collision_attempt) for i, c in enumerate(session_id))
                collision_word = words[collision_hash % len(words)]
                base_name = f"{base_name}{collision_word}"
                collision_attempt += 1

            except (json.JSONDecodeError, KeyError, ValueError, OSError):
                break  # Malformed file - assume stale, use base name
    else:
        # session_id is required - fail gracefully
        raise ValueError("session_id required for instance naming")

    if tag:
        # Security: Sanitize tag to prevent log delimiter injection (defense-in-depth)
        # Remove dangerous characters that could break message log parsing
        sanitized_tag = ''.join(c for c in tag if c not in '|\n\r\t')
        if not sanitized_tag:
            raise ValueError("Tag contains only invalid characters")
        if sanitized_tag != tag:
            print(f"Warning: Tag contained invalid characters, sanitized to '{sanitized_tag}'", file=sys.stderr)
        return f"{sanitized_tag}-{base_name}"
    return base_name

def resolve_instance_name(session_id: str, tag: str | None = None) -> tuple[str, dict | None]:
    """
    Resolve instance name for a session_id.
    Searches existing instances first (reuses if found), generates new name if not found.
    Returns: (instance_name, existing_data_or_none)
    """
    instances_dir = hcom_path(INSTANCES_DIR)

    # Search for existing instance with this session_id
    if session_id and instances_dir.exists():
        for instance_file in instances_dir.glob("*.json"):
            try:
                data = load_instance_position(instance_file.stem)
                if session_id == data.get('session_id'):
                    return instance_file.stem, data
            except (json.JSONDecodeError, OSError, KeyError):
                continue

    # Not found - generate new name
    instance_name = get_display_name(session_id, tag)
    return instance_name, None

def _remove_hcom_hooks_from_settings(settings: dict[str, Any]) -> None:
    """Remove hcom hooks from settings dict"""
    if not isinstance(settings, dict) or 'hooks' not in settings:
        return

    if not isinstance(settings['hooks'], dict):
        return

    import copy

    # Check all hook types including PostToolUse for backward compatibility cleanup
    for event in LEGACY_HOOK_TYPES:
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


def build_env_string(env_vars: dict[str, Any], format_type: str = "bash") -> str:
    """Build environment variable string for bash shells"""
    if format_type == "bash_export":
        # Properly escape values for bash
        return ' '.join(f'export {k}={shlex.quote(str(v))};' for k, v in env_vars.items())
    else:
        return ' '.join(f'{k}={shlex.quote(str(v))}' for k, v in env_vars.items())


def format_error(message: str, suggestion: str | None = None) -> str:
    """Format error message consistently"""
    base = f"Error: {message}"
    if suggestion:
        base += f". {suggestion}"
    return base


def has_claude_arg(claude_args: list[str] | None, arg_names: list[str], arg_prefixes: tuple[str, ...]) -> bool:
    """Check if argument already exists in claude_args"""
    return bool(claude_args and any(
        arg in arg_names or arg.startswith(arg_prefixes)
        for arg in claude_args
    ))

def build_claude_command(agent_content: str | None = None, claude_args: list[str] | None = None, initial_prompt: str = "Say hi in chat", model: str | None = None, tools: str | None = None) -> tuple[str, str | None]:
    """Build Claude command with proper argument handling
    Returns tuple: (command_string, temp_file_path_or_none)
    For agent content, writes to temp file and uses cat to read it.
    """
    cmd_parts = ['claude']
    temp_file_path = None

    # Add model if specified and not already in claude_args
    if model:
        if not has_claude_arg(claude_args, ['--model', '-m'], ('--model=', '-m=')):
            cmd_parts.extend(['--model', model])

    # Add allowed tools if specified and not already in claude_args
    if tools:
        if not has_claude_arg(claude_args, ['--allowedTools', '--allowed-tools'],
                              ('--allowedTools=', '--allowed-tools=')):
            cmd_parts.extend(['--allowedTools', tools])
    
    if claude_args:
        for arg in claude_args:
            cmd_parts.append(shlex.quote(arg))
    
    if agent_content:
        # Create agent files in scripts directory for unified cleanup
        scripts_dir = hcom_path(SCRIPTS_DIR)
        temp_file = tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', suffix='.txt', delete=False,
                                              prefix='hcom_agent_', dir=str(scripts_dir))
        temp_file.write(agent_content)
        temp_file.close()
        temp_file_path = temp_file.name
        
        if claude_args and any(arg in claude_args for arg in ['-p', '--print']):
            flag = '--system-prompt'
        else:
            flag = '--append-system-prompt'
        
        cmd_parts.append(flag)
        cmd_parts.append(f'"$(cat {shlex.quote(temp_file_path)})"')

    # Add initial prompt if non-empty
    if initial_prompt:
        cmd_parts.append(shlex.quote(initial_prompt))

    return ' '.join(cmd_parts), temp_file_path

def create_bash_script(script_file: str, env: dict[str, Any], cwd: str | None, command_str: str, background: bool = False) -> None:
    """Create a bash script for terminal launch
    Scripts provide uniform execution across all platforms/terminals.
    Cleanup behavior:
    - Normal scripts: append 'rm -f' command for self-deletion
    - Background scripts: persist until `hcom reset logs` cleanup (24 hours)
    - Agent scripts: treated like background (contain 'hcom_agent_')
    """
    try:
        script_path = Path(script_file)
    except (OSError, IOError) as e:
        raise Exception(f"Cannot create script directory: {e}")

    with open(script_file, 'w', encoding='utf-8') as f:
        f.write('#!/bin/bash\n')
        f.write('echo "Starting Claude Code..."\n')

        if platform.system() != 'Windows':
            # 1. Discover paths once
            claude_path = shutil.which('claude')
            node_path = shutil.which('node')

            # 2. Add to PATH for minimal environments
            paths_to_add = []
            for p in [node_path, claude_path]:
                if p:
                    dir_path = str(Path(p).resolve().parent)
                    if dir_path not in paths_to_add:
                        paths_to_add.append(dir_path)

            if paths_to_add:
                path_addition = ':'.join(paths_to_add)
                f.write(f'export PATH="{path_addition}:$PATH"\n')
            elif not claude_path:
                # Warning for debugging
                print("Warning: Could not locate 'claude' in PATH", file=sys.stderr)

            # 3. Write environment variables
            f.write(build_env_string(env, "bash_export") + '\n')

            if cwd:
                f.write(f'cd {shlex.quote(cwd)}\n')

            # 4. Platform-specific command modifications
            if claude_path:
                if is_termux():
                    # Termux: explicit node to bypass shebang issues
                    final_node = node_path or '/data/data/com.termux/files/usr/bin/node'
                    # Quote paths for safety
                    command_str = command_str.replace(
                        'claude ',
                        f'{shlex.quote(final_node)} {shlex.quote(claude_path)} ',
                        1
                    )
                else:
                    # Mac/Linux: use full path (PATH now has node if needed)
                    command_str = command_str.replace('claude ', f'{shlex.quote(claude_path)} ', 1)
        else:
            # Windows: no PATH modification needed
            f.write(build_env_string(env, "bash_export") + '\n')
            if cwd:
                f.write(f'cd {shlex.quote(cwd)}\n')

        f.write(f'{command_str}\n')

        # Self-delete for normal mode (not background or agent)
        if not background and 'hcom_agent_' not in command_str:
            f.write(f'rm -f {shlex.quote(script_file)}\n')

    # Make executable on Unix
    if platform.system() != 'Windows':
        os.chmod(script_file, 0o755)

def find_bash_on_windows() -> str | None:
    """Find Git Bash on Windows, avoiding WSL's bash launcher"""
    # Build prioritized list of bash candidates
    candidates = []
    # 1. Common Git Bash locations (highest priority)
    for base in [os.environ.get('PROGRAMFILES', r'C:\Program Files'),
                 os.environ.get('PROGRAMFILES(X86)', r'C:\Program Files (x86)')]:
        if base:
            candidates.extend([
                str(Path(base) / 'Git' / 'usr' / 'bin' / 'bash.exe'),  # usr/bin is more common
                str(Path(base) / 'Git' / 'bin' / 'bash.exe')
            ])
    # 2. Portable Git installation
    if local_appdata := os.environ.get('LOCALAPPDATA', ''):
        git_portable = Path(local_appdata) / 'Programs' / 'Git'
        candidates.extend([
            str(git_portable / 'usr' / 'bin' / 'bash.exe'),
            str(git_portable / 'bin' / 'bash.exe')
        ])
    # 3. PATH bash (if not WSL's launcher)
    if (path_bash := shutil.which('bash')) and not path_bash.lower().endswith(r'system32\bash.exe'):
        candidates.append(path_bash)
    # 4. Hardcoded fallbacks (last resort)
    candidates.extend([
        r'C:\Program Files\Git\usr\bin\bash.exe',
        r'C:\Program Files\Git\bin\bash.exe',
        r'C:\Program Files (x86)\Git\usr\bin\bash.exe',
        r'C:\Program Files (x86)\Git\bin\bash.exe'
    ])
    # Find first existing bash
    for bash in candidates:
        if bash and Path(bash).exists():
            return bash

    return None

# New helper functions for platform-specific terminal launching
def get_macos_terminal_argv() -> list[str]:
    """Return macOS Terminal.app launch command as argv list."""
    return ['osascript', '-e', 'tell app "Terminal" to do script "bash {script}"', '-e', 'tell app "Terminal" to activate']

def get_windows_terminal_argv() -> list[str]:
    """Return Windows terminal launcher as argv list."""
    if not (bash_exe := find_bash_on_windows()):
        raise Exception(format_error("Git Bash not found"))

    if shutil.which('wt'):
        return ['wt', bash_exe, '{script}']
    return ['cmd', '/c', 'start', 'Claude Code', bash_exe, '{script}']

def get_linux_terminal_argv() -> list[str] | None:
    """Return first available Linux terminal as argv list."""
    terminals = [
        ('gnome-terminal', ['gnome-terminal', '--', 'bash', '{script}']),
        ('konsole', ['konsole', '-e', 'bash', '{script}']),
        ('xterm', ['xterm', '-e', 'bash', '{script}']),
    ]
    for term_name, argv_template in terminals:
        if shutil.which(term_name):
            return argv_template

    # WSL fallback integrated here
    if is_wsl() and shutil.which('cmd.exe'):
        if shutil.which('wt.exe'):
            return ['cmd.exe', '/c', 'start', 'wt.exe', 'bash', '{script}']
        return ['cmd.exe', '/c', 'start', 'bash', '{script}']

    return None

def windows_hidden_popen(argv: list[str], *, env: dict[str, str] | None = None, cwd: str | None = None, stdout: Any = None) -> subprocess.Popen:
    """Create hidden Windows process without console window."""
    if IS_WINDOWS:
        startupinfo = subprocess.STARTUPINFO()  # type: ignore[attr-defined]
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW  # type: ignore[attr-defined]
        startupinfo.wShowWindow = subprocess.SW_HIDE  # type: ignore[attr-defined]

        return subprocess.Popen(
            argv,
            env=env,
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=subprocess.STDOUT,
            startupinfo=startupinfo,
            creationflags=CREATE_NO_WINDOW
        )
    else:
        raise RuntimeError("windows_hidden_popen called on non-Windows platform")

# Platform dispatch map
PLATFORM_TERMINAL_GETTERS = {
    'Darwin': get_macos_terminal_argv,
    'Windows': get_windows_terminal_argv,
    'Linux': get_linux_terminal_argv,
}

def _parse_terminal_command(template: str, script_file: str) -> list[str]:
    """Parse terminal command template safely to prevent shell injection.
    Parses the template FIRST, then replaces {script} placeholder in the
    parsed tokens. This avoids shell injection and handles paths with spaces.
    Args:
        template: Terminal command template with {script} placeholder
        script_file: Path to script file to substitute
    Returns:
        list: Parsed command as argv array
    Raises:
        ValueError: If template is invalid or missing {script} placeholder
    """
    if '{script}' not in template:
        raise ValueError(format_error("Custom terminal command must include {script} placeholder",
                                    'Example: open -n -a kitty.app --args bash "{script}"'))

    try:
        parts = shlex.split(template)
    except ValueError as e:
        raise ValueError(format_error(f"Invalid terminal command syntax: {e}",
                                    "Check for unmatched quotes or invalid shell syntax"))

    # Replace {script} in parsed tokens
    replaced = []
    placeholder_found = False
    for part in parts:
        if '{script}' in part:
            replaced.append(part.replace('{script}', script_file))
            placeholder_found = True
        else:
            replaced.append(part)

    if not placeholder_found:
        raise ValueError(format_error("{script} placeholder not found after parsing",
                                    "Ensure {script} is not inside environment variables"))

    return replaced

def launch_terminal(command: str, env: dict[str, str], cwd: str | None = None, background: bool = False) -> str | bool | None:
    """Launch terminal with command using unified script-first approach
    Args:
        command: Command string from build_claude_command
        env: Environment variables to set
        cwd: Working directory
        background: Launch as background process
    """
    env_vars = os.environ.copy()
    env_vars.update(env)
    command_str = command

    # 1) Always create a script
    script_file = str(hcom_path(SCRIPTS_DIR,
        f'hcom_{os.getpid()}_{random.randint(1000,9999)}.sh'))
    create_bash_script(script_file, env, cwd, command_str, background)

    # 2) Background mode
    if background:
        logs_dir = hcom_path(LOGS_DIR)
        log_file = logs_dir / env['HCOM_BACKGROUND']

        try:
            with open(log_file, 'w', encoding='utf-8') as log_handle:
                if IS_WINDOWS:
                    # Windows: hidden bash execution with Python-piped logs
                    bash_exe = find_bash_on_windows()
                    if not bash_exe:
                        raise Exception("Git Bash not found")

                    process = windows_hidden_popen(
                        [bash_exe, script_file],
                        env=env_vars,
                        cwd=cwd,
                        stdout=log_handle
                    )
                else:
                    # Unix(Mac/Linux/Termux): detached bash execution with Python-piped logs
                    process = subprocess.Popen(
                        ['bash', script_file],
                        env=env_vars, cwd=cwd,
                        stdin=subprocess.DEVNULL,
                        stdout=log_handle, stderr=subprocess.STDOUT,
                        start_new_session=True
                    )

        except OSError as e:
            print(format_error(f"Failed to launch background instance: {e}"), file=sys.stderr)
            return None

        # Health check
        time.sleep(0.2)
        if process.poll() is not None:
            error_output = read_file_with_retry(log_file, lambda f: f.read()[:1000], default="")
            print(format_error("Background instance failed immediately"), file=sys.stderr)
            if error_output:
                print(f"  Output: {error_output}", file=sys.stderr)
            return None

        return str(log_file)

    # 3) Terminal modes
    terminal_mode = get_config().terminal

    if terminal_mode == 'print':
        # Print script path and contents
        try:
            with open(script_file, 'r', encoding='utf-8') as f:
                script_content = f.read()
            print(f"# Script: {script_file}")
            print(script_content)
            Path(script_file).unlink()  # Clean up immediately
            return True
        except Exception as e:
            print(format_error(f"Failed to read script: {e}"), file=sys.stderr)
            return False

    if terminal_mode == 'here':
        print("Launching Claude in current terminal...")
        if IS_WINDOWS:
            bash_exe = find_bash_on_windows()
            if not bash_exe:
                print(format_error("Git Bash not found"), file=sys.stderr)
                return False
            result = subprocess.run([bash_exe, script_file], env=env_vars, cwd=cwd)
        else:
            result = subprocess.run(['bash', script_file], env=env_vars, cwd=cwd)
        return result.returncode == 0

    # 4) New window or custom command mode
    # If terminal is not 'here' or 'print', it's either 'new' (platform default) or a custom command
    custom_cmd = None if terminal_mode == 'new' else terminal_mode

    if not custom_cmd:  # Platform default 'new' mode
        if is_termux():
            # Keep Termux as special case
            am_cmd = [
                'am', 'startservice', '--user', '0',
                '-n', 'com.termux/com.termux.app.RunCommandService',
                '-a', 'com.termux.RUN_COMMAND',
                '--es', 'com.termux.RUN_COMMAND_PATH', script_file,
                '--ez', 'com.termux.RUN_COMMAND_BACKGROUND', 'false'
            ]
            try:
                subprocess.run(am_cmd, check=False)
                return True
            except Exception as e:
                print(format_error(f"Failed to launch Termux: {e}"), file=sys.stderr)
                return False

        # Unified platform handling via helpers
        system = platform.system()
        if not (terminal_getter := PLATFORM_TERMINAL_GETTERS.get(system)):
            raise Exception(format_error(f"Unsupported platform: {system}"))

        custom_cmd = terminal_getter()
        if not custom_cmd:  # e.g., Linux with no terminals
            raise Exception(format_error("No supported terminal emulator found",
                                       "Install gnome-terminal, konsole, or xterm"))

    # Type-based dispatch for execution
    if isinstance(custom_cmd, list):
        # Our argv commands - safe execution without shell
        final_argv = [arg.replace('{script}', script_file) for arg in custom_cmd]
        try:
            if platform.system() == 'Windows':
                # Windows needs non-blocking for parallel launches
                subprocess.Popen(final_argv)
                return True  # Popen is non-blocking, can't check success
            else:
                result = subprocess.run(final_argv)
                if result.returncode != 0:
                    return False
                return True
        except Exception as e:
            print(format_error(f"Failed to launch terminal: {e}"), file=sys.stderr)
            return False
    else:
        # User-provided string commands - parse safely without shell=True
        try:
            final_argv = _parse_terminal_command(custom_cmd, script_file)
        except ValueError as e:
            print(str(e), file=sys.stderr)
            return False

        try:
            if platform.system() == 'Windows':
                # Windows needs non-blocking for parallel launches
                subprocess.Popen(final_argv)
                return True  # Popen is non-blocking, can't check success
            else:
                result = subprocess.run(final_argv)
                if result.returncode != 0:
                    return False
                return True
        except Exception as e:
            print(format_error(f"Failed to execute terminal command: {e}"), file=sys.stderr)
            return False

def setup_hooks() -> bool:
    """Set up Claude hooks globally in ~/.claude/settings.json"""

    # TODO: Remove after v0.6.0 - cleanup legacy per-directory hooks
    try:
        positions = load_all_positions()
        if positions:
            directories = set()
            for instance_data in positions.values():
                if isinstance(instance_data, dict) and 'directory' in instance_data:
                    directories.add(instance_data['directory'])
            for directory in directories:
                if Path(directory).exists():
                    cleanup_directory_hooks(Path(directory))
    except Exception:
        pass  # Don't fail hook setup if cleanup fails

    # Install to global user settings
    settings_path = get_claude_settings_path()
    settings_path.parent.mkdir(exist_ok=True)
    try:
        settings = load_settings_json(settings_path, default={})
        if settings is None:
            settings = {}
    except (json.JSONDecodeError, PermissionError) as e:
        raise Exception(format_error(f"Cannot read settings: {e}"))
    
    if 'hooks' not in settings:
        settings['hooks'] = {}

    _remove_hcom_hooks_from_settings(settings)
        
    # Get the hook command template
    hook_cmd_base, _ = get_hook_command()

    # Build hook commands from HOOK_CONFIGS
    hook_configs = [
        (hook_type, matcher, f'{hook_cmd_base} {cmd_suffix}', timeout)
        for hook_type, matcher, cmd_suffix, timeout in HOOK_CONFIGS
    ]

    for hook_type, matcher, command, timeout in hook_configs:
        if hook_type not in settings['hooks']:
            settings['hooks'][hook_type] = []

        hook_dict = {
            'matcher': matcher,
            'hooks': [{
                'type': 'command',
                'command': command
            }]
        }
        if timeout is not None:
            hook_dict['hooks'][0]['timeout'] = timeout

        settings['hooks'][hook_type].append(hook_dict)

    # Set $HCOM environment variable for all Claude instances (vanilla + hcom-launched)
    if 'env' not in settings:
        settings['env'] = {}

    # Set HCOM based on current execution context (uvx, hcom binary, or full path)
    settings['env']['HCOM'] = _build_hcom_env_value()

    # Write settings atomically
    try:
        atomic_write(settings_path, json.dumps(settings, indent=2))
    except Exception as e:
        raise Exception(format_error(f"Cannot write settings: {e}"))
    
    # Quick verification
    if not verify_hooks_installed(settings_path):
        raise Exception(format_error("Hook installation failed"))
    
    return True

def verify_hooks_installed(settings_path: Path) -> bool:
    """Verify that HCOM hooks were installed correctly with correct commands"""
    try:
        settings = load_settings_json(settings_path, default=None)
        if not settings:
            return False

        # Check all hook types have correct commands (exactly one HCOM hook per type)
        # Derive from HOOK_CONFIGS (single source of truth)
        hooks = settings.get('hooks', {})
        for hook_type, _, cmd_suffix, _ in HOOK_CONFIGS:
            hook_matchers = hooks.get(hook_type, [])
            if not hook_matchers:
                return False

            # Count HCOM hooks for this type
            hcom_hook_count = 0
            for matcher in hook_matchers:
                for hook in matcher.get('hooks', []):
                    command = hook.get('command', '')
                    # Check for HCOM and the correct subcommand
                    if ('${HCOM}' in command or 'hcom' in command.lower()) and cmd_suffix in command:
                        hcom_hook_count += 1

            # Must have exactly one HCOM hook (not zero, not duplicates)
            if hcom_hook_count != 1:
                return False

        # Check that HCOM env var is set
        env = settings.get('env', {})
        if 'HCOM' not in env:
            return False

        return True
    except Exception:
        return False

def is_interactive() -> bool:
    """Check if running in interactive mode"""
    return sys.stdin.isatty() and sys.stdout.isatty()

def get_archive_timestamp() -> str:
    """Get timestamp for archive files"""
    return datetime.now().strftime("%Y-%m-%d_%H%M%S")

class LogParseResult(NamedTuple):
    """Result from parsing log messages"""
    messages: list[dict[str, str]]
    end_position: int

def parse_log_messages(log_file: Path, start_pos: int = 0) -> LogParseResult:
    """Parse messages from log file
    Args:
        log_file: Path to log file
        start_pos: Position to start reading from
    Returns:
        LogParseResult containing messages and end position
    """
    if not log_file.exists():
        return LogParseResult([], start_pos)

    def read_messages(f):
        f.seek(start_pos)
        content = f.read()
        end_pos = f.tell()  # Capture actual end position

        if not content.strip():
            return LogParseResult([], end_pos)

        messages = []
        message_entries = TIMESTAMP_SPLIT_PATTERN.split(content.strip())

        for entry in message_entries:
            if not entry or '|' not in entry:
                continue

            parts = entry.split('|', 2)
            if len(parts) == 3:
                timestamp, from_instance, message = parts
                messages.append({
                    'timestamp': timestamp,
                    'from': from_instance.replace('\\|', '|'),
                    'message': message.replace('\\|', '|')
                })

        return LogParseResult(messages, end_pos)

    return read_file_with_retry(
        log_file,
        read_messages,
        default=LogParseResult([], start_pos)
    )

def get_unread_messages(instance_name: str, update_position: bool = False) -> list[dict[str, str]]:
    """Get unread messages for instance with @-mention filtering
    Args:
        instance_name: Name of instance to get messages for
        update_position: If True, mark messages as read by updating position
    """
    log_file = hcom_path(LOG_FILE)

    if not log_file.exists():
        return []

    positions = load_all_positions()

    # Get last position for this instance
    last_pos = 0
    if instance_name in positions:
        pos_data = positions.get(instance_name, {})
        last_pos = pos_data.get('pos', 0) if isinstance(pos_data, dict) else pos_data

    # Atomic read with position tracking
    result = parse_log_messages(log_file, last_pos)
    all_messages, new_pos = result.messages, result.end_position

    # Filter messages:
    # 1. Exclude own messages
    # 2. Apply @-mention filtering
    all_instance_names = list(positions.keys())
    messages = []
    for msg in all_messages:
        if msg['from'] != instance_name:
            if should_deliver_message(msg, instance_name, all_instance_names):
                messages.append(msg)

    # Only update position (ie mark as read) if explicitly requested (after successful delivery)
    if update_position:
        update_instance_position(instance_name, {'pos': new_pos})

    return messages

def format_age(seconds: float) -> str:
    """Format time ago in human readable form"""
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        return f"{int(seconds/60)}m"
    else:
        return f"{int(seconds/3600)}h"

def get_instance_status(pos_data: dict[str, Any]) -> tuple[str, str, str]:
    """Get current status of instance. Returns (status_type, age_string, description)."""
    # Returns: (display_category, formatted_age, status_description)
    now = int(time.time())

    # Get last known status
    last_status = pos_data.get('last_status', '')
    last_status_time = pos_data.get('last_status_time', 0)
    last_context = pos_data.get('last_status_context', '')

    if not last_status or not last_status_time:
        return "unknown", "", "unknown"

    # Get display category and description template from STATUS_INFO
    display_status, desc_template = STATUS_INFO.get(last_status, ('unknown', 'unknown'))

    # Check timeout
    age = now - last_status_time
    timeout = pos_data.get('wait_timeout', get_config().timeout)
    if age > timeout:
        return "inactive", "", "timeout"

    # Check Stop hook heartbeat for both blocked-generic and waiting-stale detection
    last_stop = pos_data.get('last_stop', 0)
    heartbeat_age = now - last_stop if last_stop else 999999

    # Generic "Claude is waiting for your input" from Notification hook is meaningless
    # If Stop hook is actively polling (heartbeat < 2s), instance is actually idle
    if last_status == 'blocked' and last_context == "Claude is waiting for your input" and heartbeat_age < 2:
        last_status = 'waiting'
        display_status, desc_template = 'waiting', 'idle'

    # Detect stale 'waiting' status - check heartbeat, not status timestamp
    if last_status == 'waiting' and heartbeat_age > 2:
        status_suffix = " (bg)" if pos_data.get('background') else ""
        return "unknown", f"({format_age(heartbeat_age)}){status_suffix}", "stale"

    # Format description with context if template has {}
    if '{}' in desc_template and last_context:
        status_desc = desc_template.format(last_context)
    else:
        status_desc = desc_template

    status_suffix = " (bg)" if pos_data.get('background') else ""
    return display_status, f"({format_age(age)}){status_suffix}", status_desc

def get_status_block(status_type: str) -> str:
    """Get colored status block for a status type"""
    color, symbol = STATUS_MAP.get(status_type, (BG_RED, "?"))
    text_color = FG_BLACK if color == BG_YELLOW else FG_WHITE
    return f"{text_color}{BOLD}{color} {symbol} {RESET}"

def format_message_line(msg: dict[str, str], truncate: bool = False) -> str:
    """Format a message for display"""
    time_obj = datetime.fromisoformat(msg['timestamp'])
    time_str = time_obj.strftime("%H:%M")

    display_name = f"{SENDER_EMOJI} {msg['from']}" if msg['from'] == SENDER else msg['from']
    
    if truncate:
        sender = display_name[:10]
        message = msg['message'][:50]
        return f"   {DIM}{time_str}{RESET} {BOLD}{sender}{RESET}: {message}"
    else:
        return f"{DIM}{time_str}{RESET} {BOLD}{display_name}{RESET}: {msg['message']}"

def show_recent_messages(messages: list[dict[str, str]], limit: int | None = None, truncate: bool = False) -> None:
    """Show recent messages"""
    if limit is None:
        messages_to_show = messages
    else:
        start_idx = max(0, len(messages) - limit)
        messages_to_show = messages[start_idx:]
    
    for msg in messages_to_show:
        print(format_message_line(msg, truncate))


def get_terminal_height() -> int:
    """Get current terminal height"""
    try:
        return shutil.get_terminal_size().lines
    except (AttributeError, OSError):
        return 24

def show_recent_activity_alt_screen(limit: int | None = None) -> None:
    """Show recent messages in alt screen format with dynamic height"""
    if limit is None:
        # Calculate available height: total - header(8) - instances(varies) - footer(4) - input(3)
        available_height = get_terminal_height() - 20
        limit = max(2, available_height // 2)
    
    log_file = hcom_path(LOG_FILE)
    if log_file.exists():
        messages = parse_log_messages(log_file).messages
        show_recent_messages(messages, limit, truncate=True)

def should_show_in_watch(d: dict[str, Any]) -> bool:
    """Show only enabled instances by default"""
    # Hide disabled instances
    if not d.get('enabled', False):
        return False

    # Hide truly ended sessions
    if d.get('session_ended'):
        return False

    # Show all other instances (including 'closed' during transition)
    return True

def show_instances_by_directory() -> None:
    """Show instances organized by their working directories"""
    positions = load_all_positions()
    if not positions:
        print(f"   {DIM}No Claude instances connected{RESET}")
        return

    if positions:
        directories = {}
        for instance_name, pos_data in positions.items():
            if not should_show_in_watch(pos_data):
                continue
            directory = pos_data.get("directory", "unknown")
            if directory not in directories:
                directories[directory] = []
            directories[directory].append((instance_name, pos_data))

        for directory, instances in directories.items():
            print(f" {directory}")
            for instance_name, pos_data in instances:
                status_type, age, status_desc = get_instance_status(pos_data)
                status_block = get_status_block(status_type)

                print(f"   {FG_GREEN}->{RESET} {BOLD}{instance_name}{RESET} {status_block} {DIM}{status_desc} {age}{RESET}")
            print()
    else:
        print(f"   {DIM}Error reading instance data{RESET}")

def alt_screen_detailed_status_and_input() -> str:
    """Show detailed status in alt screen and get user input"""
    sys.stdout.write("\033[?1049h\033[2J\033[H")
    
    try:
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"{BOLD}HCOM{RESET} STATUS {DIM}- UPDATED: {timestamp}{RESET}")
        print(f"{DIM}{'─' * 40}{RESET}")
        print()
        
        show_instances_by_directory()
        
        print()
        print(f"{BOLD} RECENT ACTIVITY:{RESET}")
        
        show_recent_activity_alt_screen()
        
        print()
        print(f"{DIM}{'─' * 40}{RESET}")
        print(f"{FG_GREEN} Press Enter to send message (empty to cancel):{RESET}")
        message = input(f"{FG_CYAN} > {RESET}")

        print(f"{DIM}{'─' * 40}{RESET}")
        
    finally:
        sys.stdout.write("\033[?1049l")
    
    return message

def get_status_summary() -> str:
    """Get a one-line summary of all instance statuses"""
    positions = load_all_positions()
    if not positions:
        return f"{BG_BLUE}{BOLD}{FG_WHITE} no instances {RESET}"

    status_counts = {status: 0 for status in STATUS_MAP.keys()}

    for _, pos_data in positions.items():
        # Only count instances that should be shown in watch
        if not should_show_in_watch(pos_data):
            continue
        status_type, _, _ = get_instance_status(pos_data)
        if status_type in status_counts:
            status_counts[status_type] += 1

    parts = []
    status_order = ["active", "delivered", "waiting", "blocked", "inactive", "unknown"]

    for status_type in status_order:
        count = status_counts[status_type]
        if count > 0:
            color, symbol = STATUS_MAP[status_type]
            text_color = FG_BLACK if color == BG_YELLOW else FG_WHITE
            part = f"{text_color}{BOLD}{color} {count} {symbol} {RESET}"
            parts.append(part)

    if parts:
        result = "".join(parts)
        return result
    else:
        return f"{BG_BLUE}{BOLD}{FG_WHITE} no instances {RESET}"

def update_status(s: str) -> None:
    """Update status line in place"""
    sys.stdout.write("\r\033[K" + s)
    sys.stdout.flush()

def log_line_with_status(message: str, status: str) -> None:
    """Print message and immediately restore status"""
    sys.stdout.write("\r\033[K" + message + "\n")
    sys.stdout.write("\033[K" + status)
    sys.stdout.flush()

def initialize_instance_in_position_file(instance_name: str, session_id: str | None = None) -> bool:
    """Initialize instance file with required fields (idempotent). Returns True on success, False on failure."""
    try:
        data = load_instance_position(instance_name)

        # Determine default enabled state: True for hcom-launched, False for vanilla
        is_hcom_launched = os.environ.get('HCOM_LAUNCHED') == '1'

        # Determine starting position: skip history or read from beginning (or last max_msgs num)
        initial_pos = 0
        if SKIP_HISTORY:
            log_file = hcom_path(LOG_FILE)
            if log_file.exists():
                initial_pos = log_file.stat().st_size

        defaults = {
            "pos": initial_pos,
            "enabled": is_hcom_launched,
            "directory": str(Path.cwd()),
            "last_stop": 0,
            "session_id": session_id or "",
            "transcript_path": "",
            "notification_message": "",
            "alias_announced": False,
            "tag": None
        }

        # Add missing fields (preserve existing)
        for key, value in defaults.items():
            data.setdefault(key, value)

        return save_instance_position(instance_name, data)
    except Exception:
        return False

def update_instance_position(instance_name: str, update_fields: dict[str, Any]) -> None:
    """Update instance position (with NEW and IMPROVED Windows file locking tolerance!!)"""
    try:
        data = load_instance_position(instance_name)

        if not data: # If file empty/missing, initialize first
            initialize_instance_in_position_file(instance_name)
            data = load_instance_position(instance_name)

        data.update(update_fields)
        save_instance_position(instance_name, data)
    except PermissionError: # Expected on Windows during file locks, silently continue
        pass
    except Exception: # Other exceptions on Windows may also be file locking related
        if IS_WINDOWS:
            pass
        else:
            raise

def enable_instance(instance_name: str) -> None:
    """Enable instance - clears all stop flags and enables Stop hook polling"""
    update_instance_position(instance_name, {
        'enabled': True,
        'force_closed': False,
        'session_ended': False
    })
    set_status(instance_name, 'started')

def disable_instance(instance_name: str, force: bool = False) -> None:
    """Disable instance - stops Stop hook polling"""
    updates = {
        'enabled': False
    }
    if force:
        updates['force_closed'] = True
    update_instance_position(instance_name, updates)
    set_status(instance_name, 'force_stopped' if force else 'stopped')

def set_status(instance_name: str, status: str, context: str = ''):
    """Set instance status event with timestamp"""
    update_instance_position(instance_name, {
        'last_status': status,
        'last_status_time': int(time.time()),
        'last_status_context': context
    })
    log_hook_error('set_status', f'Setting status to {status} with context {context} for {instance_name}')

# ==================== Command Functions ====================

def show_main_screen_header() -> list[dict[str, str]]:
    """Show header for main screen"""
    sys.stdout.write("\033[2J\033[H")

    log_file = hcom_path(LOG_FILE)
    all_messages = []
    if log_file.exists():
        all_messages = parse_log_messages(log_file).messages

    print(f"{BOLD}HCOM{RESET} LOGS")
    print(f"{DIM}{'─'*40}{RESET}\n")
    
    return all_messages

def cmd_help() -> int:
    """Show help text"""
    print(HELP_TEXT)
    return 0

def cmd_launch(argv: list[str]) -> int:
    """Launch Claude instances: hcom [N] [claude] [args]"""
    try:
        # Parse arguments: hcom [N] [claude] [args]
        count = 1
        forwarded = []

        # Extract count if first arg is digit
        if argv and argv[0].isdigit():
            count = int(argv[0])
            if count <= 0:
                raise CLIError('Count must be positive.')
            if count > 100:
                raise CLIError('Too many instances requested (max 100).')
            argv = argv[1:]

        # Skip 'claude' keyword if present
        if argv and argv[0] == 'claude':
            argv = argv[1:]

        # Forward all remaining args to claude CLI
        forwarded = argv

        # Get tag from config
        tag = get_config().tag
        if tag and '|' in tag:
            raise CLIError('Tag cannot contain "|" characters.')

        # Get agents from config (comma-separated)
        agent_env = get_config().agent
        agents = [a.strip() for a in agent_env.split(',') if a.strip()] if agent_env else ['']

        # Detect background mode from -p/--print flags in forwarded args
        background = '-p' in forwarded or '--print' in forwarded

        # Add -p flag and stream-json output for background mode if not already present
        claude_args = forwarded
        if background and '-p' not in claude_args and '--print' not in claude_args:
            claude_args = ['-p', '--output-format', 'stream-json', '--verbose'] + (claude_args or [])

        terminal_mode = get_config().terminal

        # Calculate total instances to launch
        total_instances = count * len(agents)

        # Fail fast for here mode with multiple instances
        if terminal_mode == 'here' and total_instances > 1:
            print(format_error(
                f"'here' mode cannot launch {total_instances} instances (it's one terminal window)",
                "Use 'hcom 1' for one generic instance"
            ), file=sys.stderr)
            return 1

        log_file = hcom_path(LOG_FILE)
        instances_dir = hcom_path(INSTANCES_DIR)

        if not log_file.exists():
            log_file.touch()

        # Build environment variables for Claude instances
        base_env = build_claude_env()

        # Add tag-specific hints if provided
        if tag:
            base_env['HCOM_TAG'] = tag

        launched = 0
        initial_prompt = get_config().prompt

        # Launch count instances of each agent
        for agent in agents:
            for _ in range(count):
                instance_type = agent
                instance_env = base_env.copy()

                # Mark all hcom-launched instances
                instance_env['HCOM_LAUNCHED'] = '1'

                # Mark background instances via environment with log filename
                if background:
                    # Generate unique log filename
                    log_filename = f'background_{int(time.time())}_{random.randint(1000, 9999)}.log'
                    instance_env['HCOM_BACKGROUND'] = log_filename

                # Build claude command
                if not instance_type:
                    # No agent - no agent content
                    claude_cmd, _ = build_claude_command(
                        agent_content=None,
                        claude_args=claude_args,
                        initial_prompt=initial_prompt
                    )
                else:
                    # Agent instance
                    try:
                        agent_content, agent_config = resolve_agent(instance_type)
                        # Mark this as a subagent instance for SessionStart hook
                        instance_env['HCOM_SUBAGENT_TYPE'] = instance_type
                        # Prepend agent instance awareness to system prompt
                        agent_prefix = f"You are an instance of {instance_type}. Do not start a subagent with {instance_type} unless explicitly asked.\n\n"
                        agent_content = agent_prefix + agent_content
                        # Use agent's model and tools if specified and not overridden in claude_args
                        agent_model = agent_config.get('model')
                        agent_tools = agent_config.get('tools')
                        claude_cmd, _ = build_claude_command(
                            agent_content=agent_content,
                            claude_args=claude_args,
                            initial_prompt=initial_prompt,
                            model=agent_model,
                            tools=agent_tools
                        )
                        # Agent temp files live under ~/.hcom/scripts/ for unified housekeeping cleanup
                    except (FileNotFoundError, ValueError) as e:
                        print(str(e), file=sys.stderr)
                        continue

                try:
                    if background:
                        log_file = launch_terminal(claude_cmd, instance_env, cwd=os.getcwd(), background=True)
                        if log_file:
                            print(f"Background instance launched, log: {log_file}")
                            launched += 1
                    else:
                        if launch_terminal(claude_cmd, instance_env, cwd=os.getcwd()):
                            launched += 1
                except Exception as e:
                    print(format_error(f"Failed to launch terminal: {e}"), file=sys.stderr)

        requested = total_instances
        failed = requested - launched

        if launched == 0:
            print(format_error(f"No instances launched (0/{requested})"), file=sys.stderr)
            return 1

        # Show results
        if failed > 0:
            print(f"Launched {launched}/{requested} Claude instance{'s' if requested != 1 else ''} ({failed} failed)")
        else:
            print(f"Launched {launched} Claude instance{'s' if launched != 1 else ''}")

        # Auto-launch watch dashboard if in new window mode (new or custom) and all instances launched successfully
        terminal_mode = get_config().terminal

        # Only auto-watch if ALL instances launched successfully and launches windows (not 'here' or 'print')
        if terminal_mode not in ('here', 'print') and failed == 0 and is_interactive():
            # Show tips first if needed
            if tag:
                print(f"\n  • Send to {tag} team: hcom send '@{tag} message'")

            # Clear transition message
            print("\nOpening hcom watch...")
            time.sleep(2)  # Brief pause so user sees the message

            # Launch interactive watch dashboard in current terminal
            return cmd_watch([])  # Empty argv = interactive mode
        else:
            tips = [
                "Run 'hcom watch' to view/send in conversation dashboard",
            ]
            if tag:
                tips.append(f"Send to {tag} team: hcom send '@{tag} message'")

            if tips:
                print("\n" + "\n".join(f"  • {tip}" for tip in tips) + "\n")

            return 0
        
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 1
    except Exception as e:
        print(str(e), file=sys.stderr)
        return 1

def cmd_watch(argv: list[str]) -> int:
    """View conversation dashboard: hcom watch [--logs|--status|--wait [SEC]]"""
    # Extract launch flag for external terminals (used by claude code bootstrap)
    cleaned_args: list[str] = []
    for arg in argv:
        if arg == '--launch':
            watch_cmd = f"{build_hcom_command()} watch"
            result = launch_terminal(watch_cmd, build_claude_env(), cwd=os.getcwd())
            return 0 if result else 1
        else:
            cleaned_args.append(arg)
    argv = cleaned_args

    # Parse arguments
    show_logs = '--logs' in argv
    show_status = '--status' in argv
    wait_timeout = None

    # Check for --wait flag
    if '--wait' in argv:
        idx = argv.index('--wait')
        if idx + 1 < len(argv):
            try:
                wait_timeout = int(argv[idx + 1])
                if wait_timeout < 0:
                    raise CLIError('--wait expects a non-negative number of seconds.')
            except ValueError:
                wait_timeout = 60  # Default for non-numeric values
        else:
            wait_timeout = 60  # Default timeout
        show_logs = True  # --wait implies logs mode

    log_file = hcom_path(LOG_FILE)
    instances_dir = hcom_path(INSTANCES_DIR)

    if not log_file.exists() and not instances_dir.exists():
        print(format_error("No conversation log found", "Run 'hcom' first"), file=sys.stderr)
        return 1

    # Non-interactive mode (no TTY or flags specified)
    if not is_interactive() or show_logs or show_status:
        if show_logs:
            # Atomic position capture BEFORE parsing (prevents race condition)
            if log_file.exists():
                last_pos = log_file.stat().st_size  # Capture position first
                messages = parse_log_messages(log_file).messages
            else:
                last_pos = 0
                messages = []
            
            # If --wait, show recent messages (max of: last 3 messages OR all messages in last 5 seconds)
            if wait_timeout is not None:
                cutoff = datetime.now() - timedelta(seconds=5)
                recent_by_time = [m for m in messages if datetime.fromisoformat(m['timestamp']) > cutoff]
                last_three = messages[-3:] if len(messages) >= 3 else messages
                # Show whichever is larger: recent by time or last 3
                recent_messages = recent_by_time if len(recent_by_time) > len(last_three) else last_three
                # Status to stderr, data to stdout
                if recent_messages:
                    print(f'---Showing recent messages---', file=sys.stderr)
                    for msg in recent_messages:
                        print(f"[{msg['timestamp']}] {msg['from']}: {msg['message']}")
                    print(f'---Waiting for new message... (exits on receipt or after {wait_timeout} seconds)---', file=sys.stderr)
                else:
                    print(f'---Waiting for new message... (exits on receipt or after {wait_timeout} seconds)---', file=sys.stderr)
                
                
                # Wait loop
                start_time = time.time()
                while time.time() - start_time < wait_timeout:
                    if log_file.exists():
                        current_size = log_file.stat().st_size
                        new_messages = []
                        if current_size > last_pos:
                            # Capture new position BEFORE parsing (atomic)
                            new_messages = parse_log_messages(log_file, last_pos).messages
                        if new_messages:
                            for msg in new_messages:
                                print(f"[{msg['timestamp']}] {msg['from']}: {msg['message']}")
                            last_pos = current_size  # Update only after successful processing
                            return 0  # Success - got new messages
                        if current_size > last_pos:
                            last_pos = current_size  # Update even if no messages (file grew but no complete messages yet)
                    time.sleep(0.1)
                
                # Timeout message to stderr
                print(f'[TIMED OUT] No new messages received after {wait_timeout} seconds.', file=sys.stderr)
                return 1  # Timeout - no new messages
            
            # Regular --logs (no --wait): print all messages to stdout
            else:
                if messages:
                    for msg in messages:
                        print(f"[{msg['timestamp']}] {msg['from']}: {msg['message']}")
                else:
                    print("No messages yet", file=sys.stderr)
            
                    
        elif show_status:
            # Build JSON output
            positions = load_all_positions()

            instances = {}
            status_counts = {}

            for name, data in positions.items():
                if not should_show_in_watch(data):
                    continue
                status, age, _ = get_instance_status(data)
                instances[name] = {
                    "status": status,
                    "age": age.strip() if age else "",
                    "directory": data.get("directory", "unknown"),
                    "session_id": data.get("session_id", ""),
                    "last_status": data.get("last_status", ""),
                    "last_status_time": data.get("last_status_time", 0),
                    "last_status_context": data.get("last_status_context", ""),
                    "background": bool(data.get("background"))
                }
                status_counts[status] = status_counts.get(status, 0) + 1

            # Get recent messages
            messages = []
            if log_file.exists():
                all_messages = parse_log_messages(log_file).messages
                messages = all_messages[-5:] if all_messages else []

            # Output JSON
            output = {
                "instances": instances,
                "recent_messages": messages,
                "status_summary": status_counts,
                "log_file": str(log_file),
                "timestamp": datetime.now().isoformat()
            }

            print(json.dumps(output, indent=2))
        else:
            print("No TTY - Automation usage:", file=sys.stderr)
            print("  hcom watch --logs      Show message history", file=sys.stderr)
            print("  hcom watch --status    Show instance status", file=sys.stderr)
            print("  hcom watch --wait      Wait for new messages", file=sys.stderr)
            print("  hcom watch --launch    Launch interactive dashboard in new terminal", file=sys.stderr)
            print("  Full information: hcom --help")
            
        return 0
    
    # Interactive dashboard mode
    status_suffix = f"{DIM} [⏎]...{RESET}"

    # Atomic position capture BEFORE showing messages (prevents race condition)
    if log_file.exists():
        last_pos = log_file.stat().st_size
    else:
        last_pos = 0
    
    all_messages = show_main_screen_header()
    
    show_recent_messages(all_messages, limit=5)
    print(f"\n{DIM}· · · · watching for new messages · · · ·{RESET}")

    # Print newline to ensure status starts on its own line
    print()
    
    current_status = get_status_summary()
    update_status(f"{current_status}{status_suffix}")
    last_status_update = time.time()
    
    last_status = current_status
    
    try:
        while True:
            now = time.time()
            if now - last_status_update > 0.1:  # 100ms
                current_status = get_status_summary()
                
                # Only redraw if status text changed
                if current_status != last_status:
                    update_status(f"{current_status}{status_suffix}")
                    last_status = current_status
                
                last_status_update = now
            
            if log_file.exists():
                current_size = log_file.stat().st_size
                if current_size > last_pos:
                    new_messages = parse_log_messages(log_file, last_pos).messages
                    # Use the last known status for consistency
                    status_line_text = f"{last_status}{status_suffix}"
                    for msg in new_messages:
                        log_line_with_status(format_message_line(msg), status_line_text)
                    last_pos = current_size
            
            # Check for keyboard input
            ready_for_input = False
            if IS_WINDOWS:
                import msvcrt  # type: ignore[import]
                if msvcrt.kbhit():  # type: ignore[attr-defined]
                    msvcrt.getch()  # type: ignore[attr-defined]
                    ready_for_input = True
            else:
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    sys.stdin.readline()
                    ready_for_input = True
            
            if ready_for_input:
                sys.stdout.write("\r\033[K")
                
                message = alt_screen_detailed_status_and_input()
                
                all_messages = show_main_screen_header()
                show_recent_messages(all_messages)
                print(f"\n{DIM}· · · · watching for new messages · · · ·{RESET}")
                print(f"{DIM}{'─' * 40}{RESET}")
                
                if log_file.exists():
                    last_pos = log_file.stat().st_size
                
                if message and message.strip():
                    send_cli(message.strip(), quiet=True)
                    print(f"{FG_GREEN}✓ Sent{RESET}")

                print()
                
                current_status = get_status_summary()
                update_status(f"{current_status}{status_suffix}")
            
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        sys.stdout.write("\033[?1049l\r\033[K")
        print(f"\n{DIM}[stopped]{RESET}")
        
    return 0

def clear() -> int:
    """Clear and archive conversation"""
    log_file = hcom_path(LOG_FILE)
    instances_dir = hcom_path(INSTANCES_DIR)
    archive_folder = hcom_path(ARCHIVE_DIR)

    # cleanup: temp files, old scripts, old outbox files
    cutoff_time = time.time() - (24 * 60 * 60)  # 24 hours ago
    if instances_dir.exists():
        sum(1 for f in instances_dir.glob('*.tmp') if f.unlink(missing_ok=True) is None)

    scripts_dir = hcom_path(SCRIPTS_DIR)
    if scripts_dir.exists():
        sum(1 for f in scripts_dir.glob('*') if f.is_file() and f.stat().st_mtime < cutoff_time and f.unlink(missing_ok=True) is None)

    # Check if hcom files exist
    if not log_file.exists() and not instances_dir.exists():
        print("No HCOM conversation to clear")
        return 0

    # Archive existing files if they have content
    timestamp = get_archive_timestamp()
    archived = False

    try:
        has_log = log_file.exists() and log_file.stat().st_size > 0
        has_instances = instances_dir.exists() and any(instances_dir.glob('*.json'))
        
        if has_log or has_instances:
            # Create session archive folder with timestamp
            session_archive = hcom_path(ARCHIVE_DIR, f'session-{timestamp}')
            session_archive.mkdir(parents=True, exist_ok=True)
            
            # Archive log file
            if has_log:
                archive_log = session_archive / LOG_FILE
                log_file.rename(archive_log)
                archived = True
            elif log_file.exists():
                log_file.unlink()
            
            # Archive instances
            if has_instances:
                archive_instances = session_archive / INSTANCES_DIR
                archive_instances.mkdir(parents=True, exist_ok=True)

                # Move json files only
                for f in instances_dir.glob('*.json'):
                    f.rename(archive_instances / f.name)

                archived = True
        else:
            # Clean up empty files/dirs
            if log_file.exists():
                log_file.unlink()
            if instances_dir.exists():
                shutil.rmtree(instances_dir)
        
        log_file.touch()
        clear_all_positions()

        if archived:
            print(f"Archived to archive/session-{timestamp}/")
        print("Started fresh HCOM conversation log")
        return 0
        
    except Exception as e:
        print(format_error(f"Failed to archive: {e}"), file=sys.stderr)
        return 1

def remove_global_hooks() -> bool:
    """Remove HCOM hooks from ~/.claude/settings.json
    Returns True on success, False on failure."""
    settings_path = get_claude_settings_path()

    if not settings_path.exists():
        return True  # No settings = no hooks to remove

    try:
        settings = load_settings_json(settings_path, default=None)
        if not settings:
            return False

        _remove_hcom_hooks_from_settings(settings)
        atomic_write(settings_path, json.dumps(settings, indent=2))
        return True
    except Exception:
        return False

def cleanup_directory_hooks(directory: Path | str) -> tuple[int, str]:
    """Remove hcom hooks from a specific directory
    Returns tuple: (exit_code, message)
        exit_code: 0 for success, 1 for error
        message: what happened
    """
    settings_path = Path(directory) / '.claude' / 'settings.local.json'
    
    if not settings_path.exists():
        return 0, "No Claude settings found"
    
    try:
        # Load existing settings
        settings = load_settings_json(settings_path, default=None)
        if not settings:
            return 1, "Cannot read Claude settings"
        
        hooks_found = False

        # Include PostToolUse for backward compatibility cleanup
        original_hook_count = sum(len(settings.get('hooks', {}).get(event, []))
                                  for event in LEGACY_HOOK_TYPES)

        _remove_hcom_hooks_from_settings(settings)

        # Check if any were removed
        new_hook_count = sum(len(settings.get('hooks', {}).get(event, []))
                             for event in LEGACY_HOOK_TYPES)
        if new_hook_count < original_hook_count:
            hooks_found = True
                
        if not hooks_found:
            return 0, "No hcom hooks found"
        
        # Write back or delete settings
        if not settings or (len(settings) == 0):
            # Delete empty settings file
            settings_path.unlink()
            return 0, "Removed hcom hooks (settings file deleted)"
        else:
            # Write updated settings
            atomic_write(settings_path, json.dumps(settings, indent=2))
            return 0, "Removed hcom hooks from settings"
        
    except json.JSONDecodeError:
        return 1, format_error("Corrupted settings.local.json file")
    except Exception as e:
        return 1, format_error(f"Cannot modify settings.local.json: {e}")


def cmd_stop(argv: list[str]) -> int:
    """Stop instances: hcom stop [alias|all] [--force] [--_hcom_session ID]"""
    # Parse arguments
    target = None
    force = '--force' in argv
    session_id = None

    # Extract --_hcom_session if present
    if '--_hcom_session' in argv:
        idx = argv.index('--_hcom_session')
        if idx + 1 < len(argv):
            session_id = argv[idx + 1]
            argv = argv[:idx] + argv[idx + 2:]

    # Remove flags to get target
    args_without_flags = [a for a in argv if not a.startswith('--')]
    if args_without_flags:
        target = args_without_flags[0]

    # Handle 'all' target
    if target == 'all':
        positions = load_all_positions()

        if not positions:
            print("No instances found")
            return 0

        stopped_count = 0
        bg_logs = []
        stopped_names = []
        for instance_name, instance_data in positions.items():
            if instance_data.get('enabled', False):
                disable_instance(instance_name)
                stopped_names.append(instance_name)
                stopped_count += 1

                # Track background logs
                if instance_data.get('background'):
                    log_file = instance_data.get('background_log_file', '')
                    if log_file:
                        bg_logs.append((instance_name, log_file))

        if stopped_count == 0:
            print("No instances to stop")
        else:
            print(f"Stopped {stopped_count} instance(s): {', '.join(stopped_names)}")

            # Show background logs if any
            if bg_logs:
                print()
                print("Background instance logs:")
                for name, log_file in bg_logs:
                    print(f"  {name}: {log_file}")

        return 0

    # Stop specific instance or self
    # Get instance name from injected session or target
    if session_id and not target:
        instance_name, _ = resolve_instance_name(session_id, get_config().tag)
    else:
        instance_name = target

    position = load_instance_position(instance_name) if instance_name else None

    if not instance_name:
        if os.environ.get('CLAUDECODE') == '1':
            print("Error: Cannot determine instance", file=sys.stderr)
            print("Usage: Prompt Claude to run 'hcom stop' (or directly use: hcom stop <alias> or hcom stop all)", file=sys.stderr)
        else:
            print("Error: Alias required", file=sys.stderr)
            print("Usage: hcom stop <alias>", file=sys.stderr)
            print("   Or: hcom stop all", file=sys.stderr)
            print("   Or: prompt claude to run 'hcom stop' on itself", file=sys.stderr)
            positions = load_all_positions()
            visible = [alias for alias, data in positions.items() if should_show_in_watch(data)]
            if visible:
                print(f"Active aliases: {', '.join(sorted(visible))}", file=sys.stderr)
        return 1

    if not position:
        print(f"No instance found for {instance_name}")
        return 1

    # Skip already stopped instances (unless forcing)
    if not position.get('enabled', False) and not force:
        print(f"HCOM already stopped for {instance_name}")
        return 0

    # Disable instance (optionally with force)
    disable_instance(instance_name, force=force)

    if force:
        print(f"⚠️  Force stopped HCOM for {instance_name}.")
        print(f"    Bash tool use is now DENIED.")
        print(f"    To restart: hcom start {instance_name}")
    else:
        print(f"Stopped HCOM for {instance_name}. Will no longer receive chat messages automatically.")

    # Show background log location if applicable
    if position.get('background'):
        log_file = position.get('background_log_file', '')
        if log_file:
            print(f"\nBackground log: {log_file}")
            print(f"Monitor: tail -f {log_file}")
            if not force:
                print(f"Force stop: hcom stop --force {instance_name}")

    return 0

def cmd_start(argv: list[str]) -> int:
    """Enable HCOM participation: hcom start [alias] [--_hcom_session ID]"""
    # Parse arguments
    target = None
    session_id = None

    # Extract --_hcom_session if present
    if '--_hcom_session' in argv:
        idx = argv.index('--_hcom_session')
        if idx + 1 < len(argv):
            session_id = argv[idx + 1]
            argv = argv[:idx] + argv[idx + 2:]

    # Remove flags to get target
    args_without_flags = [a for a in argv if not a.startswith('--')]
    if args_without_flags:
        target = args_without_flags[0]

    # Get instance name from injected session or target
    if session_id and not target:
        instance_name, existing_data = resolve_instance_name(session_id, get_config().tag)

        # Create instance if it doesn't exist (opt-in for vanilla instances)
        if not existing_data:
            initialize_instance_in_position_file(instance_name, session_id)
            # Enable instance (clears all stop flags)
            enable_instance(instance_name)
            print(f"\nStarted HCOM for {instance_name}")
        else:
            # Skip already started instances
            if existing_data.get('enabled', False):
                print(f"HCOM already started for {instance_name}")
                return 0

            # Check if background instance has exited permanently
            if existing_data.get('session_ended') and existing_data.get('background'):
                session = existing_data.get('session_id', '')
                print(f"Cannot start {instance_name}: background instance has exited permanently")
                print(f"Background instances terminate when stopped and cannot be restarted")
                if session:
                    print(f"Resume conversation with same alias: hcom 1 claude -p --resume {session}")
                return 1

            # Re-enabling existing instance
            enable_instance(instance_name)
            print(f"Started HCOM for {instance_name}")

        return 0

    # CLI path: start specific instance
    positions = load_all_positions()

    # Handle missing target from external CLI
    if not target:
        if os.environ.get('CLAUDECODE') == '1':
            print("Error: Cannot determine instance", file=sys.stderr)
            print("Usage: Prompt Claude to run 'hcom start' (or: hcom start <alias>)", file=sys.stderr)
        else:
            print("Error: Alias required", file=sys.stderr)
            print("Usage: hcom start <alias> (or: prompt claude to run 'hcom start')", file=sys.stderr)
            print("To launch new instances: hcom <count>", file=sys.stderr)
        return 1

    # Start specific instance
    instance_name = target
    position = positions.get(instance_name)

    if not position:
        print(f"Instance not found: {instance_name}")
        return 1

    # Skip already started instances
    if position.get('enabled', False):
        print(f"HCOM already started for {instance_name}")
        return 0

    # Check if background instance has exited permanently
    if position.get('session_ended') and position.get('background'):
        session = position.get('session_id', '')
        print(f"Cannot start {instance_name}: background instance has exited permanently")
        print(f"Background instances terminate when stopped and cannot be restarted")
        if session:
            print(f"Resume conversation with same alias: hcom 1 claude -p --resume {session}")
        return 1

    # Enable instance (clears all stop flags)
    enable_instance(instance_name)

    print(f"Started HCOM for {instance_name}. Rejoined chat.")
    return 0

def cmd_reset(argv: list[str]) -> int:
    """Reset HCOM components: logs, hooks, config
    Usage:
        hcom reset              # Everything (stop all + logs + hooks + config)
        hcom reset logs         # Archive conversation only
        hcom reset hooks        # Remove hooks only
        hcom reset config       # Clear config (backup to config.env.TIMESTAMP)
        hcom reset logs hooks   # Combine targets
    """
    # No args = everything
    do_everything = not argv
    targets = argv if argv else ['logs', 'hooks', 'config']

    # Validate targets
    valid = {'logs', 'hooks', 'config'}
    invalid = [t for t in targets if t not in valid]
    if invalid:
        print(f"Invalid target(s): {', '.join(invalid)}", file=sys.stderr)
        print("Valid targets: logs, hooks, config", file=sys.stderr)
        return 1

    exit_codes = []

    # Stop all instances if doing everything
    if do_everything:
        exit_codes.append(cmd_stop(['all']))

    # Execute based on targets
    if 'logs' in targets:
        exit_codes.append(clear())

    if 'hooks' in targets:
        exit_codes.append(cleanup('--all'))
        if remove_global_hooks():
            print("Removed hooks")
        else:
            print("Warning: Could not remove hooks. Check your claude settings.json file it might be invalid", file=sys.stderr)
            exit_codes.append(1)

    if 'config' in targets:
        config_path = hcom_path(CONFIG_FILE)
        if config_path.exists():
            # Backup with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_path = hcom_path(f'config.env.{timestamp}')
            shutil.copy2(config_path, backup_path)
            config_path.unlink()
            print(f"Config backed up to config.env.{timestamp} and cleared")
            exit_codes.append(0)
        else:
            print("No config file to clear")
            exit_codes.append(0)

    return max(exit_codes) if exit_codes else 0

def cleanup(*args: str) -> int:
    """Remove hcom hooks from current directory or all directories"""
    if args and args[0] == '--all':
        directories = set()

        # Get all directories from current instances
        try:
            positions = load_all_positions()
            if positions:
                for instance_data in positions.values():
                    if isinstance(instance_data, dict) and 'directory' in instance_data:
                        directories.add(instance_data['directory'])
        except Exception as e:
            print(f"Warning: Could not read current instances: {e}")

        # Also check archived instances for directories (until 0.5.0)
        try:
            archive_dir = hcom_path(ARCHIVE_DIR)
            if archive_dir.exists():
                for session_dir in archive_dir.iterdir():
                    if session_dir.is_dir() and session_dir.name.startswith('session-'):
                        instances_dir = session_dir / 'instances'
                        if instances_dir.exists():
                            for instance_file in instances_dir.glob('*.json'):
                                try:
                                    data = json.loads(instance_file.read_text())
                                    if 'directory' in data:
                                        directories.add(data['directory'])
                                except Exception:
                                    pass
        except Exception as e:
            print(f"Warning: Could not read archived instances: {e}")
        
        if not directories:
            print("No directories found in current HCOM tracking")
            return 0
        
        print(f"Found {len(directories)} unique directories to check")
        cleaned = 0
        failed = 0
        already_clean = 0
        
        for directory in sorted(directories):
            # Check if directory exists
            if not Path(directory).exists():
                print(f"\nSkipping {directory} (directory no longer exists)")
                continue
                
            print(f"\nChecking {directory}...")

            exit_code, message = cleanup_directory_hooks(Path(directory))
            if exit_code == 0:
                if "No hcom hooks found" in message or "No Claude settings found" in message:
                    already_clean += 1
                    print(f"  {message}")
                else:
                    cleaned += 1
                    print(f"  {message}")
            else:
                failed += 1
                print(f"  {message}")
        
        print(f"\nSummary:")
        print(f"  Cleaned: {cleaned} directories")
        print(f"  Already clean: {already_clean} directories")
        if failed > 0:
            print(f"  Failed: {failed} directories")
            return 1
        return 0
            
    else:
        exit_code, message = cleanup_directory_hooks(Path.cwd())
        print(message)
        return exit_code

def ensure_hooks_current() -> bool:
    """Ensure hooks match current execution context - called on EVERY command.
    Auto-updates hooks if execution context changes (e.g., pip → uvx).
    Always returns True (warns but never blocks - Claude Code is fault-tolerant)."""

    # Verify hooks exist and match current execution context
    global_settings = get_claude_settings_path()

    # Check if hooks are valid (exist + env var matches current context)
    hooks_exist = verify_hooks_installed(global_settings)
    env_var_matches = False

    if hooks_exist:
        try:
            settings = load_settings_json(global_settings, default={})
            if settings is None:
                settings = {}
            current_hcom = _build_hcom_env_value()
            installed_hcom = settings.get('env', {}).get('HCOM')
            env_var_matches = (installed_hcom == current_hcom)
        except Exception:
            # Failed to read settings - try to fix by updating
            env_var_matches = False

    # Install/update hooks if missing or env var wrong
    if not hooks_exist or not env_var_matches:
        try:
            setup_hooks()
            if os.environ.get('CLAUDECODE') == '1':
                print("HCOM hooks updated. Please restart Claude Code to apply changes.", file=sys.stderr)
                print("=" * 60, file=sys.stderr)
        except Exception as e:
            # Failed to verify/update hooks, but they might still work
            # Claude Code is fault-tolerant with malformed JSON
            print(f"⚠️  Could not verify/update hooks: {e}", file=sys.stderr)
            print("If HCOM doesn't work, check ~/.claude/settings.json", file=sys.stderr)

    return True

def cmd_send(argv: list[str], force_cli: bool = False, quiet: bool = False) -> int:
    """Send message to hcom: hcom send "message" [--_hcom_session ID]"""
    # Parse message and session_id
    message = None
    session_id = None

    # Extract --_hcom_session if present (injected by PreToolUse hook)
    if '--_hcom_session' in argv:
        idx = argv.index('--_hcom_session')
        if idx + 1 < len(argv):
            session_id = argv[idx + 1]
            argv = argv[:idx] + argv[idx + 2:]  # Remove flag and value

    # First non-flag argument is the message
    if argv:
        message = argv[0]

    # Check message is provided
    if not message:
        print(format_error("No message provided"), file=sys.stderr)
        return 1

    # Check if hcom files exist
    log_file = hcom_path(LOG_FILE)
    instances_dir = hcom_path(INSTANCES_DIR)

    if not log_file.exists() and not instances_dir.exists():
        print(format_error("No conversation found", "Run 'hcom <count>' first"), file=sys.stderr)
        return 1

    # Validate message
    error = validate_message(message)
    if error:
        print(error, file=sys.stderr)
        return 1

    # Check for unmatched mentions (minimal warning)
    mentions = MENTION_PATTERN.findall(message)
    if mentions:
        try:
            positions = load_all_positions()
            all_instances = list(positions.keys())
            sender_name = SENDER
            all_names = all_instances + [sender_name]
            unmatched = [m for m in mentions
                        if not any(name.lower().startswith(m.lower()) for name in all_names)]
            if unmatched:
                print(f"Note: @{', @'.join(unmatched)} don't match any instances - broadcasting to all", file=sys.stderr)
        except Exception:
            pass  # Don't fail on warning

    # Determine sender from injected session_id or CLI
    if session_id and not force_cli:
        # Instance context - resolve name from session_id (searches existing instances first)
        try:
            sender_name, instance_data = resolve_instance_name(session_id, get_config().tag)
        except (ValueError, Exception) as e:
            print(format_error(f"Invalid session_id: {e}"), file=sys.stderr)
            return 1

        # Initialize instance if doesn't exist (first use)
        if not instance_data:
            initialize_instance_in_position_file(sender_name, session_id)
            instance_data = load_instance_position(sender_name)

        # Check force_closed
        if instance_data.get('force_closed'):
            print(format_error(f"HCOM force stopped for this instance. To recover, delete instance file: rm ~/.hcom/instances/{sender_name}.json"), file=sys.stderr)
            return 1

        # Check enabled state
        if not instance_data.get('enabled', False):
            print(format_error("HCOM not started for this instance. To send a message first run: 'hcom start' then use hcom send"), file=sys.stderr)
            return 1

        # Send message
        if not send_message(sender_name, message):
            print(format_error("Failed to send message"), file=sys.stderr)
            return 1

        # Show unread messages
        messages = get_unread_messages(sender_name, update_position=True)
        if messages:
            max_msgs = MAX_MESSAGES_PER_DELIVERY
            formatted = format_hook_messages(messages[:max_msgs], sender_name)
            print(f"Message sent\n\n{formatted}", file=sys.stderr)
        else:
            print("Message sent", file=sys.stderr)

        return 0
    else:
        # CLI context - no session_id or force_cli=True

        # Warn if inside Claude Code but no session_id (hooks not working)
        if os.environ.get('CLAUDECODE') == '1' and not session_id and not force_cli:
            print(f"⚠️  Cannot determine alias - message sent as '{SENDER}'", file=sys.stderr)
            print("   Prompt Claude to send a hcom message instead of using bash mode (! prefix).", file=sys.stderr)


        sender_name = SENDER

        if not send_message(sender_name, message):
            print(format_error("Failed to send message"), file=sys.stderr)
            return 1

        if not quiet:
            print(f"✓ Sent from {sender_name}", file=sys.stderr)

        return 0

def send_cli(message: str, quiet: bool = False) -> int:
    """Force CLI sender (skip outbox, use config sender name)"""
    return cmd_send([message], force_cli=True, quiet=quiet)

# ==================== Hook Helpers ====================

def format_hook_messages(messages: list[dict[str, str]], instance_name: str) -> str:
    """Format messages for hook feedback"""
    if len(messages) == 1:
        msg = messages[0]
        reason = f"[new message] {msg['from']} → {instance_name}: {msg['message']}"
    else:
        parts = [f"{msg['from']} → {instance_name}: {msg['message']}" for msg in messages]
        reason = f"[{len(messages)} new messages] | {' | '.join(parts)}"

    # Only append hints to messages
    hints = get_config().hints
    if hints:
        reason = f"{reason} | [{hints}]"

    return reason

# ==================== Hook Handlers ====================

def init_hook_context(hook_data: dict[str, Any], hook_type: str | None = None) -> tuple[str, dict[str, Any], bool]:
    """
    Initialize instance context. Flow:
    1. Resolve instance name (search by session_id, generate if not found)
    2. Create instance file if fresh start in UserPromptSubmit
    3. Build updates dict
    4. Return (instance_name, updates, is_matched_resume)
    """
    session_id = hook_data.get('session_id', '')
    transcript_path = hook_data.get('transcript_path', '')
    tag = get_config().tag

    # Resolve instance name - existing_data is None for fresh starts
    instance_name, existing_data = resolve_instance_name(session_id, tag)

    # Save migrated data if we have it
    if existing_data:
        save_instance_position(instance_name, existing_data)

    # Create instance file if fresh start in UserPromptSubmit
    if existing_data is None and hook_type == 'userpromptsubmit':
        initialize_instance_in_position_file(instance_name, session_id)

    # Build updates dict
    updates: dict[str, Any] = {
        'directory': str(Path.cwd()),
        'tag': tag,
    }

    if session_id:
        updates['session_id'] = session_id

    if transcript_path:
        updates['transcript_path'] = transcript_path

    bg_env = os.environ.get('HCOM_BACKGROUND')
    if bg_env:
        updates['background'] = True
        updates['background_log_file'] = str(hcom_path(LOGS_DIR, bg_env))

    # Simple boolean: matched resume if existing_data found
    is_matched_resume = (existing_data is not None)

    return instance_name, updates, is_matched_resume

def pretooluse_decision(decision: str, reason: str) -> None:
    """Exit PreToolUse hook with permission decision"""
    output = {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": decision,
            "permissionDecisionReason": reason
        }
    }
    print(json.dumps(output, ensure_ascii=False))
    sys.exit(0)

def handle_pretooluse(hook_data: dict[str, Any], instance_name: str) -> None:
    """Handle PreToolUse hook - check force_closed, inject session_id"""
    instance_data = load_instance_position(instance_name)
    tool_name = hook_data.get('tool_name', '')
    session_id = hook_data.get('session_id', '')

    # FORCE CLOSE CHECK - deny ALL tools
    if instance_data.get('force_closed'):
        pretooluse_decision("deny", f"FORCE CLOSED: The user has force stopped HCOM for this instance (you, {instance_name}). Stop using all tools immediately and wait for user instructions.")

    # Record status for tool execution tracking (only if enabled)
    if instance_data.get('enabled', False):
        set_status(instance_name, 'tool_pending', tool_name)

    # Inject session_id into hcom commands via updatedInput
    if tool_name == 'Bash' and session_id:
        command = hook_data.get('tool_input', {}).get('command', '')

        # Match hcom commands for session_id injection and auto-approval
        matches = list(re.finditer(HCOM_COMMAND_PATTERN, command))
        if matches:
            # Inject all if chained (&&, ||, ;, |), otherwise first only (avoids quoted text in messages)
            inject_all = len(matches) > 1 and any(op in command[matches[0].end():matches[1].start()] for op in ['&&', '||', ';', '|'])
            modified_command = HCOM_COMMAND_PATTERN.sub(rf'\g<0> --_hcom_session {session_id}', command, count=0 if inject_all else 1)

            output = {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "allow",
                    "updatedInput": {
                        "command": modified_command
                    }
                }
            }
            print(json.dumps(output, ensure_ascii=False))
            sys.exit(0)



def handle_stop(hook_data: dict[str, Any], instance_name: str, updates: dict[str, Any], instance_data: dict[str, Any] | None) -> None:
    """Handle Stop hook - poll for messages and deliver"""

    try:
        updates['last_stop'] = time.time()
        timeout = get_config().timeout
        updates['wait_timeout'] = timeout
        set_status(instance_name, 'waiting')

        try:
            update_instance_position(instance_name, updates)
        except Exception as e:
            log_hook_error(f'stop:update_instance_position({instance_name})', e)

        start_time = time.time()

        try:
            first_poll = True
            last_heartbeat = start_time
            # Actual polling loop - this IS the holding pattern
            while time.time() - start_time < timeout:
                if first_poll:
                    first_poll = False

                # Reload instance data each poll iteration
                instance_data = load_instance_position(instance_name)

                # Check flag file FIRST (highest priority coordination signal)
                flag_file = get_user_input_flag_file(instance_name)
                if flag_file.exists():
                    try:
                        flag_file.unlink()
                    except (FileNotFoundError, PermissionError):
                        # Already deleted or locked, continue anyway
                        pass
                    sys.exit(0)

                # Check if session ended (SessionEnd hook fired) - exit without changing status
                if instance_data.get('session_ended'):
                    sys.exit(0)  # Don't overwrite session_ended status

                # Check if user input is pending (timestamp fallback) - exit cleanly if recent input
                last_user_input = instance_data.get('last_user_input', 0)
                if time.time() - last_user_input < 0.2:
                    sys.exit(0)  # Don't overwrite status - let current status remain

                # Check if stopped/disabled - exit cleanly
                if not instance_data.get('enabled', False):
                    sys.exit(0)  # Preserve 'stopped' status set by cmd_stop

                # Check for new messages and deliver
                if messages := get_unread_messages(instance_name, update_position=True):
                    messages_to_show = messages[:MAX_MESSAGES_PER_DELIVERY]
                    reason = format_hook_messages(messages_to_show, instance_name)
                    set_status(instance_name, 'message_delivered', messages_to_show[0]['from'])

                    output = {"decision": "block", "reason": reason}
                    print(json.dumps(output, ensure_ascii=False), file=sys.stderr)
                    sys.exit(2)

                # Update heartbeat every 0.5 seconds for staleness detection
                now = time.time()
                if now - last_heartbeat >= 0.5:
                    try:
                        update_instance_position(instance_name, {'last_stop': now})
                        last_heartbeat = now
                    except Exception as e:
                        log_hook_error(f'stop:heartbeat_update({instance_name})', e)

                time.sleep(STOP_HOOK_POLL_INTERVAL)

        except Exception as loop_e:
            # Log polling loop errors but continue to cleanup
            log_hook_error(f'stop:polling_loop({instance_name})', loop_e)

        # Timeout reached
        set_status(instance_name, 'timeout')
        sys.exit(0)

    except Exception as e:
        # Log error and exit gracefully
        log_hook_error('handle_stop', e)
        sys.exit(0)  # Preserve previous status on exception

def handle_notify(hook_data: dict[str, Any], instance_name: str, updates: dict[str, Any], instance_data: dict[str, Any] | None) -> None:
    """Handle Notification hook - track permission requests"""
    updates['notification_message'] = hook_data.get('message', '')
    update_instance_position(instance_name, updates)
    set_status(instance_name, 'blocked', hook_data.get('message', ''))

def get_user_input_flag_file(instance_name: str) -> Path:
    """Get path to user input coordination flag file"""
    return hcom_path(FLAGS_DIR, f'{instance_name}.user_input')

def wait_for_stop_exit(instance_name: str, max_wait: float = 0.2) -> int:
    """
    Wait for Stop hook to exit using flag file coordination.
    Returns wait time in ms.
    Strategy:
    1. Create flag file
    2. Wait for Stop hook to delete it (proof it exited)
    3. Fallback to timeout if Stop hook doesn't delete flag
    """
    start = time.time()
    flag_file = get_user_input_flag_file(instance_name)

    # Wait for flag file to be deleted by Stop hook
    while flag_file.exists() and time.time() - start < max_wait:
        time.sleep(0.01)

    return int((time.time() - start) * 1000)

def handle_userpromptsubmit(hook_data: dict[str, Any], instance_name: str, updates: dict[str, Any], is_matched_resume: bool, instance_data: dict[str, Any] | None) -> None:
    """Handle UserPromptSubmit hook - track when user sends messages"""
    is_enabled = instance_data.get('enabled', False) if instance_data else False
    last_stop = instance_data.get('last_stop', 0) if instance_data else 0
    alias_announced = instance_data.get('alias_announced', False) if instance_data else False

    # Session_ended prevents user receiving messages(?) so reset it.
    if is_matched_resume and instance_data and instance_data.get('session_ended'):
        update_instance_position(instance_name, {'session_ended': False})
        instance_data['session_ended'] = False  # Resume path reactivates Stop hook polling

    # Coordinate with Stop hook only if enabled AND Stop hook is active
    stop_is_active = (time.time() - last_stop) < 1.0

    if is_enabled and stop_is_active:
        # Create flag file for coordination
        flag_file = get_user_input_flag_file(instance_name)
        try:
            flag_file.touch()
        except (OSError, PermissionError):
            # Failed to create flag, fall back to timestamp-only coordination
            pass

        # Set timestamp (backup mechanism)
        updates['last_user_input'] = time.time()
        update_instance_position(instance_name, updates)

        # Wait for Stop hook to delete flag file
        wait_for_stop_exit(instance_name)

    # Build message based on what happened
    msg = None

    # Determine if this is an HCOM-launched instance
    is_hcom_launched = os.environ.get('HCOM_LAUNCHED') == '1'

    # Show bootstrap if not already announced
    if not alias_announced:
        if is_hcom_launched:
            # HCOM-launched instance - show bootstrap immediately
            msg = build_hcom_bootstrap_text(instance_name)
            update_instance_position(instance_name, {'alias_announced': True})
        else:
            # Vanilla Claude instance - check if user is about to run an hcom command
            user_prompt = hook_data.get('prompt', '')
            hcom_command_pattern = r'\bhcom\s+\w+'
            if re.search(hcom_command_pattern, user_prompt, re.IGNORECASE):
                # Bootstrap not shown yet - show it preemptively before hcom command runs
                msg = "[HCOM COMMAND DETECTED]\n\n"
                msg += build_hcom_bootstrap_text(instance_name)
                update_instance_position(instance_name, {'alias_announced': True})

    # Add resume status note if we showed bootstrap for a matched resume
    if msg and is_matched_resume:
        if is_enabled:
            msg += "\n[HCOM Session resumed. Your alias and conversation history preserved.]"
    if msg:
        output = {
            "hookSpecificOutput": {
                "hookEventName": "UserPromptSubmit",
                "additionalContext": msg
            }
        }
        print(json.dumps(output), file=sys.stdout)

def handle_sessionstart(hook_data: dict[str, Any]) -> None:
    """Handle SessionStart hook - initial msg & reads environment variables"""
    # Only show message for HCOM-launched instances
    if os.environ.get('HCOM_LAUNCHED') == '1':
        parts = f"[HCOM is started, you can send messages with the command: {build_hcom_command()} send]"
    else:
        parts = f"[You can start HCOM with the command: {build_hcom_command()} start]"

    output = {
        "hookSpecificOutput": {
            "hookEventName": "SessionStart",
            "additionalContext": parts
        }
    }

    print(json.dumps(output))

def handle_posttooluse(hook_data: dict[str, Any], instance_name: str) -> None:
    """Handle PostToolUse hook - show launch context or bootstrap"""
    command = hook_data.get('tool_input', {}).get('command', '')
    instance_data = load_instance_position(instance_name)

    # Check for help or launch commands (combined pattern)
    if re.search(r'\bhcom\s+(?:(?:help|--help|-h)\b|\d+)', command):
        if not instance_data.get('launch_context_announced', False):
            msg = build_launch_context(instance_name)
            update_instance_position(instance_name, {'launch_context_announced': True})

            output = {
                "hookSpecificOutput": {
                    "hookEventName": "PostToolUse",
                    "additionalContext": msg
                }
            }
            print(json.dumps(output, ensure_ascii=False))
        return

    # Check HCOM_COMMAND_PATTERN for bootstrap (other hcom commands)
    matches = list(re.finditer(HCOM_COMMAND_PATTERN, command))

    if not matches:
        return

    # Show bootstrap if not announced yet
    if not instance_data.get('alias_announced', False):
        msg = build_hcom_bootstrap_text(instance_name)
        update_instance_position(instance_name, {'alias_announced': True})

        output = {
            "hookSpecificOutput": {
                "hookEventName": "PostToolUse",
                "additionalContext": msg
            }
        }
        print(json.dumps(output, ensure_ascii=False))

def handle_sessionend(hook_data: dict[str, Any], instance_name: str, updates: dict[str, Any], instance_data: dict[str, Any] | None) -> None:
    """Handle SessionEnd hook - mark session as ended and set final status"""
    reason = hook_data.get('reason', 'unknown')

    # Set session_ended flag to tell Stop hook to exit
    updates['session_ended'] = True

    # Set status with reason as context (reason: clear, logout, prompt_input_exit, other)
    set_status(instance_name, 'session_ended', reason)

    try:
        update_instance_position(instance_name, updates)
    except Exception as e:
        log_hook_error(f'sessionend:update_instance_position({instance_name})', e)

def should_skip_vanilla_instance(hook_type: str, hook_data: dict) -> bool:
    """
    Returns True if hook should exit early.
    Vanilla instances (not HCOM-launched) exit early unless:
    - Enabled
    - PreToolUse (handles opt-in)
    - UserPromptSubmit with hcom command in prompt (shows preemptive bootstrap)
    """
    # PreToolUse always runs (handles toggle commands)
    # HCOM-launched instances always run
    if hook_type == 'pre' or os.environ.get('HCOM_LAUNCHED') == '1':
        return False

    session_id = hook_data.get('session_id', '')
    if not session_id:  # No session_id = can't identify instance, skip hook
        return True

    instance_name = get_display_name(session_id, get_config().tag)
    instance_file = hcom_path(INSTANCES_DIR, f'{instance_name}.json')

    if not instance_file.exists():
        # Allow UserPromptSubmit if prompt contains hcom command
        if hook_type == 'userpromptsubmit':
            user_prompt = hook_data.get('prompt', '')
            return not re.search(r'\bhcom\s+\w+', user_prompt, re.IGNORECASE)
        return True

    return False

def handle_hook(hook_type: str) -> None:
    """Unified hook handler for all HCOM hooks"""
    hook_data = json.load(sys.stdin)

    if not ensure_hcom_directories():
        log_hook_error('handle_hook', Exception('Failed to create directories'))
        sys.exit(0)

    # SessionStart is standalone - no instance files
    if hook_type == 'sessionstart':
        handle_sessionstart(hook_data)
        sys.exit(0)

    # Vanilla instance check - exit early if should skip
    if should_skip_vanilla_instance(hook_type, hook_data):
        sys.exit(0)

    # Initialize instance context (creates file if needed, reuses existing if session_id matches)
    instance_name, updates, is_matched_resume = init_hook_context(hook_data, hook_type)

    # Load instance data once (for enabled check and to pass to handlers)
    instance_data = None
    if hook_type != 'pre':
        instance_data = load_instance_position(instance_name)

        # Skip enabled check for UserPromptSubmit when bootstrap needs to be shown
        # (alias_announced=false means bootstrap hasn't been shown yet)
        skip_enabled_check = (hook_type == 'userpromptsubmit' and
                             not instance_data.get('alias_announced', False))

        if not skip_enabled_check and not instance_data.get('enabled', False):
            sys.exit(0)

    match hook_type:
        case 'pre':
            handle_pretooluse(hook_data, instance_name)
        case 'post':
            handle_posttooluse(hook_data, instance_name)
        case 'poll':
            handle_stop(hook_data, instance_name, updates, instance_data)
        case 'notify':
            handle_notify(hook_data, instance_name, updates, instance_data)
        case 'userpromptsubmit':
            handle_userpromptsubmit(hook_data, instance_name, updates, is_matched_resume, instance_data)
        case 'sessionend':
            handle_sessionend(hook_data, instance_name, updates, instance_data)

    sys.exit(0)


# ==================== Main Entry Point ====================

def main(argv: list[str] | None = None) -> int | None:
    """Main command dispatcher"""
    if argv is None:
        argv = sys.argv[1:]
    else:
        argv = argv[1:] if len(argv) > 0 and argv[0].endswith('hcom.py') else argv

    # Hook handlers only (called BY hooks, not users)
    if argv and argv[0] in ('poll', 'notify', 'pre', 'post', 'sessionstart', 'userpromptsubmit', 'sessionend'):
        handle_hook(argv[0])
        return 0

    # Ensure directories exist first (required for version check cache)
    if not ensure_hcom_directories():
        print(format_error("Failed to create HCOM directories"), file=sys.stderr)
        return 1

    # Check for updates and show message if available (once daily check, persists until upgrade)
    if msg := get_update_notice():
        print(msg, file=sys.stderr)

    # Ensure hooks current (warns but never blocks)
    ensure_hooks_current()

    # Route to commands
    try:
        if not argv or argv[0] in ('help', '--help', '-h'):
            return cmd_help()
        elif argv[0] == 'send_cli':
            if len(argv) < 2:
                print(format_error("Message required"), file=sys.stderr)
                return 1
            return send_cli(argv[1])
        elif argv[0] == 'watch':
            return cmd_watch(argv[1:])
        elif argv[0] == 'send':
            return cmd_send(argv[1:])
        elif argv[0] == 'stop':
            return cmd_stop(argv[1:])
        elif argv[0] == 'start':
            return cmd_start(argv[1:])
        elif argv[0] == 'reset':
            return cmd_reset(argv[1:])
        elif argv[0].isdigit() or argv[0] == 'claude':
            # Launch instances: hcom <1-100> [args] or hcom claude [args]
            return cmd_launch(argv)
        else:
            print(format_error(
                f"Unknown command: {argv[0]}",
                "Run 'hcom --help' for usage"
            ), file=sys.stderr)
            return 1
    except CLIError as exc:
        print(str(exc), file=sys.stderr)
        return 1

if __name__ == '__main__':
    sys.exit(main())
