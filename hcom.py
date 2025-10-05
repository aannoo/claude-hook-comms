#!/usr/bin/env python3
"""
hcom 0.3.0
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
from typing import Optional, Any, NamedTuple
from dataclasses import dataclass, asdict, field

if sys.version_info < (3, 10):
    sys.exit("Error: hcom requires Python 3.10 or higher")

# ==================== Constants ====================

IS_WINDOWS = sys.platform == 'win32'

def is_wsl():
    """Detect if running in WSL (Windows Subsystem for Linux)"""
    if platform.system() != 'Linux':
        return False
    try:
        with open('/proc/version', 'r') as f:
            return 'microsoft' in f.read().lower()
    except (FileNotFoundError, PermissionError, OSError):
        return False

def is_termux():
    """Detect if running in Termux on Android"""
    return (
        'TERMUX_VERSION' in os.environ or              # Primary: Works all versions
        'TERMUX__ROOTFS' in os.environ or              # Modern: v0.119.0+
        Path('/data/data/com.termux').exists() or     # Fallback: Path check
        'com.termux' in os.environ.get('PREFIX', '')   # Fallback: PREFIX check
    )

HCOM_ACTIVE_ENV = 'HCOM_ACTIVE'
HCOM_ACTIVE_VALUE = '1'

EXIT_SUCCESS = 0
EXIT_BLOCK = 2

ERROR_ACCESS_DENIED = 5     # Windows - Process exists but no permission
ERROR_INVALID_PARAMETER = 87  # Windows - Invalid PID or parameters

# Windows API constants
CREATE_NO_WINDOW = 0x08000000  # Prevent console window creation
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000  # Vista+ minimal access rights
PROCESS_QUERY_INFORMATION = 0x0400  # Pre-Vista process access rights TODO: is this a joke? why am i supporting pre vista? who the fuck is running claude code on vista let alone pre?! great to keep this comment here! and i will be leaving it here!

# Timing constants
FILE_RETRY_DELAY = 0.01  # 10ms delay for file lock retries
STOP_HOOK_POLL_INTERVAL = 0.1     # 100ms between stop hook polls
KILL_CHECK_INTERVAL = 0.1  # 100ms between process termination checks
MERGE_ACTIVITY_THRESHOLD = 10  # Seconds of inactivity before allowing instance merge

# Windows kernel32 cache
_windows_kernel32_cache = None

def get_windows_kernel32():
    """Get cached Windows kernel32 with function signatures configured.
    This eliminates repeated initialization in hot code paths (e.g., stop hook polling).
    """
    global _windows_kernel32_cache
    if _windows_kernel32_cache is None and IS_WINDOWS:
        import ctypes
        import ctypes.wintypes
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]

        # Set proper ctypes function signatures to avoid ERROR_INVALID_PARAMETER
        kernel32.OpenProcess.argtypes = [ctypes.wintypes.DWORD, ctypes.wintypes.BOOL, ctypes.wintypes.DWORD]
        kernel32.OpenProcess.restype = ctypes.wintypes.HANDLE
        kernel32.GetLastError.argtypes = []
        kernel32.GetLastError.restype = ctypes.wintypes.DWORD
        kernel32.CloseHandle.argtypes = [ctypes.wintypes.HANDLE]
        kernel32.CloseHandle.restype = ctypes.wintypes.BOOL
        kernel32.GetExitCodeProcess.argtypes = [ctypes.wintypes.HANDLE, ctypes.POINTER(ctypes.wintypes.DWORD)]
        kernel32.GetExitCodeProcess.restype = ctypes.wintypes.BOOL

        _windows_kernel32_cache = kernel32
    return _windows_kernel32_cache

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
    'stop_exit': ('inactive', 'stopped'),
    'timeout': ('inactive', 'timeout'),
    'killed': ('inactive', 'killed'),
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
# Critical I/O: atomic_write, save_instance_position, merge_instance_immediately
# Pattern: Try/except/return False in hooks, raise in CLI operations.

def log_hook_error(hook_name: str, error: Exception | None = None):
    """Log hook exceptions or just general logging to ~/.hcom/scripts/hooks.log for debugging"""
    import traceback
    try:
        log_file = hcom_path(SCRIPTS_DIR) / "hooks.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
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

# Type definition for configuration
@dataclass
class HcomConfig:
    terminal_command: str | None = None
    terminal_mode: str = "new_window"
    initial_prompt: str = "Say hi in chat"
    sender_name: str = "bigboss"
    sender_emoji: str = "🐳"
    cli_hints: str = ""
    wait_timeout: int = 1800  # 30mins
    max_message_size: int = 1048576  # 1MB
    max_messages_per_delivery: int = 50
    first_use_text: str = "Essential, concise messages only, say hi in hcom chat now"
    instance_hints: str = ""
    env_overrides: dict = field(default_factory=dict)
    auto_watch: bool = True  # Auto-launch watch dashboard after open

DEFAULT_CONFIG = HcomConfig()

_config = None

# Generate env var mappings from dataclass fields (except env_overrides)
HOOK_SETTINGS = {
    field: f"HCOM_{field.upper()}"
    for field in DEFAULT_CONFIG.__dataclass_fields__
    if field != 'env_overrides'
}

# Path constants
LOG_FILE = "hcom.log"
INSTANCES_DIR = "instances"
LOGS_DIR = "logs"
SCRIPTS_DIR = "scripts"
CONFIG_FILE = "config.json"
ARCHIVE_DIR = "archive"

# ==================== File System Utilities ====================

def hcom_path(*parts: str, ensure_parent: bool = False) -> Path:
    """Build path under ~/.hcom"""
    path = Path.home() / ".hcom"
    if parts:
        path = path.joinpath(*parts)
    if ensure_parent:
        path.parent.mkdir(parents=True, exist_ok=True)
    return path

def atomic_write(filepath: str | Path, content: str) -> bool:
    """Write content to file atomically to prevent corruption (now with NEW and IMPROVED (wow!) Windows retry logic (cool!!!)). Returns True on success, False on failure."""
    filepath = Path(filepath) if not isinstance(filepath, Path) else filepath

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

def read_file_with_retry(filepath: str | Path, read_func, default: Any = None, max_retries: int = 3) -> Any:
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
        instance_file = hcom_path(INSTANCES_DIR, f"{instance_name}.json", ensure_parent=True)
        return atomic_write(instance_file, json.dumps(data, indent=2))
    except (OSError, PermissionError, ValueError):
        return False

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
        # Clean up orphaned mapping files
        for f in instances_dir.glob('.launch_map_*'):
            f.unlink(missing_ok=True)
    else:
        instances_dir.mkdir(exist_ok=True)

# ==================== Configuration System ====================

def get_cached_config():
    """Get cached configuration, loading if needed"""
    global _config
    if _config is None:
        _config = _load_config_from_file()
    return _config

def _load_config_from_file() -> dict:
    """Load configuration from ~/.hcom/config.json"""
    config_path = hcom_path(CONFIG_FILE, ensure_parent=True)

    # Start with default config as dict
    config_dict = asdict(DEFAULT_CONFIG)

    try:
        if user_config := read_file_with_retry(
            config_path,
            lambda f: json.load(f),
            default=None
        ):
            # Merge user config into default config
            config_dict.update(user_config)
        elif not config_path.exists():
            # Write default config if file doesn't exist
            atomic_write(config_path, json.dumps(config_dict, indent=2))
    except (json.JSONDecodeError, UnicodeDecodeError, PermissionError):
        print("Warning: Cannot read config file, using defaults", file=sys.stderr)
        # config_dict already has defaults

    return config_dict

def get_config_value(key: str, default: Any = None) -> Any:
    """Get config value with proper precedence:
    1. Environment variable (if in HOOK_SETTINGS)
    2. Config file
    3. Default value
    """
    if key in HOOK_SETTINGS:
        env_var = HOOK_SETTINGS[key]
        if (env_value := os.environ.get(env_var)) is not None:
            # Type conversion based on key
            if key in ['wait_timeout', 'max_message_size', 'max_messages_per_delivery']:
                try:
                    return int(env_value)
                except ValueError:
                    # Invalid integer - fall through to config/default
                    pass
            elif key == 'auto_watch':
                return env_value.lower() in ('true', '1', 'yes', 'on')
            else:
                # String values - return as-is
                return env_value

    config = get_cached_config()
    return config.get(key, default)

def get_hook_command():
    """Get hook command with silent fallback
    
    Uses ${HCOM:-true} for clean paths, conditional for paths with spaces.
    Both approaches exit silently (code 0) when not launched via 'hcom open'.
    """
    python_path = sys.executable
    script_path = str(Path(__file__).resolve())
    
    if IS_WINDOWS:
        # Windows cmd.exe syntax - no parentheses so arguments append correctly
        if ' ' in python_path or ' ' in script_path:
            return f'IF "%HCOM_ACTIVE%"=="1" "{python_path}" "{script_path}"', {}
        return f'IF "%HCOM_ACTIVE%"=="1" {python_path} {script_path}', {}
    elif ' ' in python_path or ' ' in script_path:
        # Unix with spaces: use conditional check
        escaped_python = shlex.quote(python_path)
        escaped_script = shlex.quote(script_path)
        return f'[ "${{HCOM_ACTIVE}}" = "1" ] && {escaped_python} {escaped_script} || true', {}
    else:
        # Unix clean paths: use environment variable
        return '${HCOM:-true}', {}

def build_send_command(example_msg: str = '') -> str:
    """Build send command string - checks if $HCOM exists, falls back to full path"""
    msg = f" '{example_msg}'" if example_msg else ''
    if os.environ.get('HCOM'):
        return f'eval "$HCOM send{msg}"'
    python_path = shlex.quote(sys.executable)
    script_path = shlex.quote(str(Path(__file__).resolve()))
    return f'{python_path} {script_path} send{msg}'

def build_claude_env():
    """Build environment variables for Claude instances"""
    env = {HCOM_ACTIVE_ENV: HCOM_ACTIVE_VALUE}
    
    # Get config file values
    config = get_cached_config()
    
    # Pass env vars only when they differ from config file values
    for config_key, env_var in HOOK_SETTINGS.items():
        actual_value = get_config_value(config_key)  # Respects env var precedence
        config_file_value = config.get(config_key)
        
        # Only pass if different from config file (not default)
        if actual_value != config_file_value and actual_value is not None:
            env[env_var] = str(actual_value)
    
    # Still support env_overrides from config file
    env.update(config.get('env_overrides', {}))
    
    # Set HCOM only for clean paths (spaces handled differently)
    python_path = sys.executable
    script_path = str(Path(__file__).resolve())
    if ' ' not in python_path and ' ' not in script_path:
        env['HCOM'] = f'{python_path} {script_path}'
    
    return env

# ==================== Message System ====================

def validate_message(message: str) -> Optional[str]:
    """Validate message size and content. Returns error message or None if valid."""
    if not message or not message.strip():
        return format_error("Message required")

    # Reject control characters (except \n, \r, \t)
    if re.search(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\u0080-\u009F]', message):
        return format_error("Message contains control characters")

    max_size = get_config_value('max_message_size', 1048576)
    if len(message) > max_size:
        return format_error(f"Message too large (max {max_size} chars)")

    return None

def send_message(from_instance: str, message: str) -> bool:
    """Send a message to the log"""
    try:
        log_file = hcom_path(LOG_FILE, ensure_parent=True)
        
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

def should_deliver_message(msg: dict[str, str], instance_name: str, all_instance_names: Optional[list[str]] = None) -> bool:
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
    sender_name = get_config_value('sender_name', 'bigboss')
    sender_mentioned = any(sender_name.lower().startswith(mention.lower()) for mention in mentions)
    
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

def parse_open_args(args: list[str]) -> tuple[list[str], Optional[str], list[str], bool]:
    """Parse arguments for open command
    
    Returns:
        tuple: (instances, prefix, claude_args, background)
            instances: list of agent names or 'generic'
            prefix: team name prefix or None
            claude_args: additional args to pass to claude
            background: bool, True if --background or -p flag
    """
    instances = []
    prefix = None
    claude_args = []
    background = False
    
    i = 0
    while i < len(args):
        arg = args[i]
        
        if arg == '--prefix':
            if i + 1 >= len(args):
                raise ValueError(format_error('--prefix requires an argument'))
            prefix = args[i + 1]
            if '|' in prefix:
                raise ValueError(format_error('Prefix cannot contain pipe characters'))
            i += 2
        elif arg == '--claude-args':
            # Next argument contains claude args as a string
            if i + 1 >= len(args):
                raise ValueError(format_error('--claude-args requires an argument'))
            claude_args = shlex.split(args[i + 1])
            i += 2
        elif arg == '--background' or arg == '-p':
            background = True
            i += 1
        else:
            try:
                count = int(arg)
                if count < 0:
                    raise ValueError(format_error(f"Cannot launch negative instances: {count}"))
                if count > 100:
                    raise ValueError(format_error(f"Too many instances requested: {count}", "Maximum 100 instances at once"))
                instances.extend(['generic'] * count)
            except ValueError as e:
                if "Cannot launch" in str(e) or "Too many instances" in str(e):
                    raise
                # Not a number, treat as agent name
                instances.append(arg)
            i += 1
    
    if not instances:
        instances = ['generic']
    
    return instances, prefix, claude_args, background

def extract_agent_config(content: str) -> dict[str, str]:
    """Extract configuration from agent YAML frontmatter"""
    if not content.startswith('---'):
        return {}
    
    # Find YAML section between --- markers
    yaml_end = content.find('\n---', 3)
    if yaml_end < 0:
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

def get_display_name(session_id: Optional[str], prefix: Optional[str] = None) -> str:
    """Get display name for instance using session_id"""
    syls = ['ka', 'ko', 'ma', 'mo', 'na', 'no', 'ra', 'ro', 'sa', 'so', 'ta', 'to', 'va', 'vo', 'za', 'zo', 'be', 'de', 'fe', 'ge', 'le', 'me', 'ne', 're', 'se', 'te', 've', 'we', 'hi']
    # Phonetic letters (5 per syllable, matches syls order)
    phonetic = "nrlstnrlstnrlstnrlstnrlstnrlstnmlstnmlstnrlmtnrlmtnrlmsnrlmsnrlstnrlstnrlmtnrlmtnrlaynrlaynrlaynrlayaanxrtanxrtdtraxntdaxntraxnrdaynrlaynrlasnrlst"

    dir_char = (Path.cwd().name + 'x')[0].lower()

    # Use session_id directly instead of extracting UUID from transcript
    if session_id:
        hash_val = sum(ord(c) for c in session_id)
        syl_idx = hash_val % len(syls)
        syllable = syls[syl_idx]

        letters = phonetic[syl_idx * 5:(syl_idx + 1) * 5]
        letter_hash = sum(ord(c) for c in session_id[1:]) if len(session_id) > 1 else hash_val
        letter = letters[letter_hash % 5]

        # Session IDs are UUIDs like "374acbe2-978b-4882-9c0b-641890f066e1"
        hex_char = session_id[0] if session_id else 'x'
        base_name = f"{dir_char}{syllable}{letter}{hex_char}"

        # Collision detection: if taken by different session_id, use more chars
        instance_file = hcom_path(INSTANCES_DIR, f"{base_name}.json")
        if instance_file.exists():
            try:
                with open(instance_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                their_session_id = data.get('session_id', '')

                # Deterministic check: different session_id = collision
                if their_session_id and their_session_id != session_id:
                    # Use first 4 chars of session_id for collision resolution
                    base_name = f"{dir_char}{session_id[0:4]}"
                # If same session_id, it's our file - reuse the name (no collision)
                # If no session_id in file, assume it's stale/malformed - use base name

            except (json.JSONDecodeError, KeyError, ValueError, OSError):
                pass  # Malformed file - assume stale, use base name
    else:
        # session_id is required - fail gracefully
        raise ValueError("session_id required for instance naming")

    if prefix:
        return f"{prefix}-{base_name}"
    return base_name

def _remove_hcom_hooks_from_settings(settings):
    """Remove hcom hooks from settings dict"""
    if not isinstance(settings, dict) or 'hooks' not in settings:
        return
    
    if not isinstance(settings['hooks'], dict):
        return
    
    import copy
    
    # Patterns to match any hcom hook command
    # Modern hooks (patterns 1-2, 7): Match all hook types via env var or wrapper
    # - ${HCOM:-true} sessionstart/pre/stop/notify/userpromptsubmit
    # - [ "${HCOM_ACTIVE}" = "1" ] && ... hcom.py ... || true
    # - sh -c "[ ... ] && ... hcom ..."
    # Legacy hooks (patterns 3-6): Direct command invocation
    # - hcom pre/post/stop/notify/sessionstart/userpromptsubmit
    # - uvx hcom pre/post/stop/notify/sessionstart/userpromptsubmit
    # - /path/to/hcom.py pre/post/stop/notify/sessionstart/userpromptsubmit
    hcom_patterns = [
        r'\$\{?HCOM',                                # Environment variable (${HCOM:-true}) - all hook types
        r'\bHCOM_ACTIVE.*hcom\.py',                 # Conditional with full path - all hook types
        r'\bhcom\s+(pre|post|stop|notify|sessionstart|userpromptsubmit)\b',           # Direct hcom command
        r'\buvx\s+hcom\s+(pre|post|stop|notify|sessionstart|userpromptsubmit)\b',     # uvx hcom command
        r'hcom\.py["\']?\s+(pre|post|stop|notify|sessionstart|userpromptsubmit)\b',   # hcom.py with optional quote
        r'["\'][^"\']*hcom\.py["\']?\s+(pre|post|stop|notify|sessionstart|userpromptsubmit)\b(?=\s|$)',  # Quoted path
        r'sh\s+-c.*hcom',                           # Shell wrapper with hcom
    ]
    compiled_patterns = [re.compile(pattern) for pattern in hcom_patterns]

    # Check all hook types including PostToolUse for backward compatibility cleanup
    for event in ['SessionStart', 'UserPromptSubmit', 'PreToolUse', 'PostToolUse', 'Stop', 'Notification']:
        if event not in settings['hooks']:
            continue
        
        # Process each matcher
        updated_matchers = []
        for matcher in settings['hooks'][event]:
            # Fail fast on malformed settings - Claude won't run with broken settings anyway
            if not isinstance(matcher, dict):
                raise ValueError(f"Malformed settings: matcher in {event} is not a dict: {type(matcher).__name__}")
            
            # Work with a copy to avoid any potential reference issues
            matcher_copy = copy.deepcopy(matcher)
            
            # Filter out HCOM hooks from this matcher
            non_hcom_hooks = [
                hook for hook in matcher_copy.get('hooks', [])
                if not any(
                    pattern.search(hook.get('command', ''))
                    for pattern in compiled_patterns
                )
            ]
            
            # Only keep the matcher if it has non-HCOM hooks remaining
            if non_hcom_hooks:
                matcher_copy['hooks'] = non_hcom_hooks
                updated_matchers.append(matcher_copy)
            elif not matcher.get('hooks'):  # Preserve matchers that never had hooks
                updated_matchers.append(matcher_copy)
        
        # Update or remove the event
        if updated_matchers:
            settings['hooks'][event] = updated_matchers
        else:
            del settings['hooks'][event]
    

def build_env_string(env_vars, format_type="bash"):
    """Build environment variable string for bash shells"""
    if format_type == "bash_export":
        # Properly escape values for bash
        return ' '.join(f'export {k}={shlex.quote(str(v))};' for k, v in env_vars.items())
    else:
        return ' '.join(f'{k}={shlex.quote(str(v))}' for k, v in env_vars.items())


def format_error(message: str, suggestion: Optional[str] = None) -> str:
    """Format error message consistently"""
    base = f"Error: {message}"
    if suggestion:
        base += f". {suggestion}"
    return base


def has_claude_arg(claude_args, arg_names, arg_prefixes):
    """Check if argument already exists in claude_args"""
    return claude_args and any(
        arg in arg_names or arg.startswith(arg_prefixes)
        for arg in claude_args
    )

def build_claude_command(agent_content: Optional[str] = None, claude_args: Optional[list[str]] = None, initial_prompt: str = "Say hi in chat", model: Optional[str] = None, tools: Optional[str] = None) -> tuple[str, Optional[str]]:
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
        scripts_dir.mkdir(parents=True, exist_ok=True)
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
    
    if claude_args or agent_content:
        cmd_parts.append('--')
    
    # Quote initial prompt normally
    cmd_parts.append(shlex.quote(initial_prompt))
    
    return ' '.join(cmd_parts), temp_file_path

def create_bash_script(script_file, env, cwd, command_str, background=False):
    """Create a bash script for terminal launch
    Scripts provide uniform execution across all platforms/terminals.
    Cleanup behavior:
    - Normal scripts: append 'rm -f' command for self-deletion
    - Background scripts: persist until `hcom clear` housekeeping (24 hours)
    - Agent scripts: treated like background (contain 'hcom_agent_')
    """
    try:
        # Ensure parent directory exists
        script_path = Path(script_file)
        script_path.parent.mkdir(parents=True, exist_ok=True)
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

def find_bash_on_windows():
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
    local_appdata = os.environ.get('LOCALAPPDATA', '')
    if local_appdata:
        git_portable = Path(local_appdata) / 'Programs' / 'Git'
        candidates.extend([
            str(git_portable / 'usr' / 'bin' / 'bash.exe'),
            str(git_portable / 'bin' / 'bash.exe')
        ])

    # 3. PATH bash (if not WSL's launcher)
    path_bash = shutil.which('bash')
    if path_bash and not path_bash.lower().endswith(r'system32\bash.exe'):
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
def get_macos_terminal_argv():
    """Return macOS Terminal.app launch command as argv list."""
    return ['osascript', '-e', 'tell app "Terminal" to do script "bash {script}"', '-e', 'tell app "Terminal" to activate']

def get_windows_terminal_argv():
    """Return Windows terminal launcher as argv list."""
    bash_exe = find_bash_on_windows()
    if not bash_exe:
        raise Exception(format_error("Git Bash not found"))

    if shutil.which('wt'):
        return ['wt', bash_exe, '{script}']
    return ['cmd', '/c', 'start', 'Claude Code', bash_exe, '{script}']

def get_linux_terminal_argv():
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

def windows_hidden_popen(argv, *, env=None, cwd=None, stdout=None):
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

def _parse_terminal_command(template, script_file):
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

def launch_terminal(command, env, cwd=None, background=False):
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
        f'hcom_{os.getpid()}_{random.randint(1000,9999)}.sh',
        ensure_parent=True))
    create_bash_script(script_file, env, cwd, command_str, background)

    # 2) Background mode
    if background:
        logs_dir = hcom_path(LOGS_DIR)
        logs_dir.mkdir(parents=True, exist_ok=True)
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
    terminal_mode = get_config_value('terminal_mode', 'new_window')

    if terminal_mode == 'show_commands':
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

    if terminal_mode == 'same_terminal':
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

    # 4) New window mode
    custom_cmd = get_config_value('terminal_command')

    if not custom_cmd:  # No string sentinel checks
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
        terminal_getter = PLATFORM_TERMINAL_GETTERS.get(system)
        if not terminal_getter:
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

def setup_hooks():
    """Set up Claude hooks in current directory"""
    claude_dir = Path.cwd() / '.claude'
    claude_dir.mkdir(exist_ok=True)
    
    settings_path = claude_dir / 'settings.local.json'
    try:
        settings = read_file_with_retry(
            settings_path,
            lambda f: json.load(f),
            default={}
        )
    except (json.JSONDecodeError, PermissionError) as e:
        raise Exception(format_error(f"Cannot read settings: {e}"))
    
    if 'hooks' not in settings:
        settings['hooks'] = {}

    _remove_hcom_hooks_from_settings(settings)
        
    # Get the hook command template
    hook_cmd_base, _ = get_hook_command()
    
    # Get wait_timeout (needed for Stop hook)
    wait_timeout = get_config_value('wait_timeout', 1800)
    
    # Define all hooks (PostToolUse removed - causes API 400 errors)
    hook_configs = [
        ('SessionStart', '', f'{hook_cmd_base} sessionstart', None),
        ('UserPromptSubmit', '', f'{hook_cmd_base} userpromptsubmit', None),
        ('PreToolUse', 'Bash', f'{hook_cmd_base} pre', None),
        # ('PostToolUse', '.*', f'{hook_cmd_base} post', None),  # DISABLED
        ('Stop', '', f'{hook_cmd_base} stop', wait_timeout),
        ('Notification', '', f'{hook_cmd_base} notify', None),
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
    
    # Write settings atomically
    try:
        atomic_write(settings_path, json.dumps(settings, indent=2))
    except Exception as e:
        raise Exception(format_error(f"Cannot write settings: {e}"))
    
    # Quick verification
    if not verify_hooks_installed(settings_path):
        raise Exception(format_error("Hook installation failed"))
    
    return True

def verify_hooks_installed(settings_path):
    """Verify that HCOM hooks were installed correctly"""
    try:
        settings = read_file_with_retry(
            settings_path,
            lambda f: json.load(f),
            default=None
        )
        if not settings:
            return False

        # Check all hook types exist with HCOM commands (PostToolUse removed)
        hooks = settings.get('hooks', {})
        for hook_type in ['SessionStart', 'PreToolUse', 'Stop', 'Notification']:
            if not any('hcom' in str(h).lower() or 'HCOM' in str(h)
                      for h in hooks.get(hook_type, [])):
                return False

        return True
    except Exception:
        return False

def is_interactive():
    """Check if running in interactive mode"""
    return sys.stdin.isatty() and sys.stdout.isatty()

def get_archive_timestamp():
    """Get timestamp for archive files"""
    return datetime.now().strftime("%Y-%m-%d_%H%M%S")

def is_parent_alive(parent_pid=None):
    """Check if parent process is alive"""
    if parent_pid is None:
        parent_pid = os.getppid()

    # Orphan detection - PID 1 == definitively orphaned
    if parent_pid == 1:
        return False

    result = is_process_alive(parent_pid)
    return result

def is_process_alive(pid):
    """Check if a process with given PID exists - cross-platform"""
    if pid is None:
        return False

    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False

    if IS_WINDOWS:
        # Windows: Use Windows API to check process existence
        try:
            kernel32 = get_windows_kernel32()  # Use cached kernel32 instance
            if not kernel32:
                return False

            # Try limited permissions first (more likely to succeed on Vista+)
            handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
            error = kernel32.GetLastError()

            if not handle:  # Check for None or 0
                # ERROR_ACCESS_DENIED (5) means process exists but no permission
                if error == ERROR_ACCESS_DENIED:
                    return True

                # Try fallback with broader permissions for older Windows
                handle = kernel32.OpenProcess(PROCESS_QUERY_INFORMATION, False, pid)

                if not handle:  # Check for None or 0
                    return False  # Process doesn't exist or no permission at all

            # Check if process is still running (not just if handle exists)
            import ctypes.wintypes
            exit_code = ctypes.wintypes.DWORD()
            STILL_ACTIVE = 259

            if kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                kernel32.CloseHandle(handle)
                is_still_active = exit_code.value == STILL_ACTIVE
                return is_still_active

            kernel32.CloseHandle(handle)
            return False  # Couldn't get exit code
        except Exception:
            return False
    else:
        # Unix: Use os.kill with signal 0
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except Exception:
            return False

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
    log_file = hcom_path(LOG_FILE, ensure_parent=True)

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

def get_instance_status(pos_data: dict[str, Any]) -> tuple[str, str]:
    """Get current status of instance. Returns (status_type, age_string)."""
    # Returns: (display_category, formatted_age) - category for color, age for display
    now = int(time.time())

    # Check if killed
    if pos_data.get('pid') is None: #TODO: replace this later when process management stuff removed
        return "inactive", ""

    # Get last known status
    last_status = pos_data.get('last_status', '')
    last_status_time = pos_data.get('last_status_time', 0)

    if not last_status or not last_status_time:
        return "unknown", ""

    # Get display category from STATUS_INFO
    display_status, _ = STATUS_INFO.get(last_status, ('unknown', ''))

    # Check timeout
    age = now - last_status_time
    timeout = pos_data.get('wait_timeout', get_config_value('wait_timeout', 1800))
    if age > timeout:
        return "inactive", ""

    status_suffix = " (bg)" if pos_data.get('background') else ""
    return display_status, f"({format_age(age)}){status_suffix}"

def get_status_block(status_type: str) -> str:
    """Get colored status block for a status type"""
    color, symbol = STATUS_MAP.get(status_type, (BG_RED, "?"))
    text_color = FG_BLACK if color == BG_YELLOW else FG_WHITE
    return f"{text_color}{BOLD}{color} {symbol} {RESET}"

def format_message_line(msg, truncate=False):
    """Format a message for display"""
    time_obj = datetime.fromisoformat(msg['timestamp'])
    time_str = time_obj.strftime("%H:%M")
    
    sender_name = get_config_value('sender_name', 'bigboss')
    sender_emoji = get_config_value('sender_emoji', '🐳')
    
    display_name = f"{sender_emoji} {msg['from']}" if msg['from'] == sender_name else msg['from']
    
    if truncate:
        sender = display_name[:10]
        message = msg['message'][:50]
        return f"   {DIM}{time_str}{RESET} {BOLD}{sender}{RESET}: {message}"
    else:
        return f"{DIM}{time_str}{RESET} {BOLD}{display_name}{RESET}: {msg['message']}"

def show_recent_messages(messages, limit=None, truncate=False):
    """Show recent messages"""
    if limit is None:
        messages_to_show = messages
    else:
        start_idx = max(0, len(messages) - limit)
        messages_to_show = messages[start_idx:]
    
    for msg in messages_to_show:
        print(format_message_line(msg, truncate))


def get_terminal_height():
    """Get current terminal height"""
    try:
        return shutil.get_terminal_size().lines
    except (AttributeError, OSError):
        return 24

def show_recent_activity_alt_screen(limit=None):
    """Show recent messages in alt screen format with dynamic height"""
    if limit is None:
        # Calculate available height: total - header(8) - instances(varies) - footer(4) - input(3)
        available_height = get_terminal_height() - 20
        limit = max(2, available_height // 2)
    
    log_file = hcom_path(LOG_FILE)
    if log_file.exists():
        messages = parse_log_messages(log_file).messages
        show_recent_messages(messages, limit, truncate=True)

def show_instances_by_directory():
    """Show instances organized by their working directories"""
    positions = load_all_positions()
    if not positions:
        print(f"   {DIM}No Claude instances connected{RESET}")
        return
    
    if positions:
        directories = {}
        for instance_name, pos_data in positions.items():
            directory = pos_data.get("directory", "unknown")
            if directory not in directories:
                directories[directory] = []
            directories[directory].append((instance_name, pos_data))
        
        for directory, instances in directories.items():
            print(f" {directory}")
            for instance_name, pos_data in instances:
                status_type, age = get_instance_status(pos_data)
                status_block = get_status_block(status_type)

                # Format status description using STATUS_INFO and context
                last_status = pos_data.get('last_status', '')
                last_context = pos_data.get('last_status_context', '')
                _, desc_template = STATUS_INFO.get(last_status, ('unknown', ''))

                # Format description with context if template has {}
                if '{}' in desc_template and last_context:
                    status_desc = desc_template.format(last_context)
                else:
                    status_desc = desc_template

                print(f"   {FG_GREEN}->{RESET} {BOLD}{instance_name}{RESET} {status_block} {DIM}{status_desc} {age}{RESET}")
            print()
    else:
        print(f"   {DIM}Error reading instance data{RESET}")

def alt_screen_detailed_status_and_input():
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

def get_status_summary():
    """Get a one-line summary of all instance statuses"""
    positions = load_all_positions()
    if not positions:
        return f"{BG_BLUE}{BOLD}{FG_WHITE} no instances {RESET}"

    status_counts = {status: 0 for status in STATUS_MAP.keys()}

    for _, pos_data in positions.items():
        status_type, _ = get_instance_status(pos_data)
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

def update_status(s):
    """Update status line in place"""
    sys.stdout.write("\r\033[K" + s)
    sys.stdout.flush()

def log_line_with_status(message, status):
    """Print message and immediately restore status"""
    sys.stdout.write("\r\033[K" + message + "\n")
    sys.stdout.write("\033[K" + status)
    sys.stdout.flush()

def initialize_instance_in_position_file(instance_name, session_id=None):
    """Initialize instance file with required fields (idempotent). Returns True on success, False on failure."""
    try:
        data = load_instance_position(instance_name)

        defaults = {
            "pos": 0,
            "directory": str(Path.cwd()),
            "last_stop": 0,
            "session_id": session_id or "",
            "transcript_path": "",
            "notification_message": "",
            "alias_announced": False
        }

        # Add missing fields (preserve existing)
        for key, value in defaults.items():
            data.setdefault(key, value)

        return save_instance_position(instance_name, data)
    except Exception:
        return False

def update_instance_position(instance_name, update_fields):
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

def set_status(instance_name: str, status: str, context: str = ''):
    """Set instance status event with timestamp"""
    update_instance_position(instance_name, {
        'last_status': status,
        'last_status_time': int(time.time()),
        'last_status_context': context
    })
    log_hook_error(f"Set status for {instance_name} to {status} with context {context}") #TODO: change to 'log'?

def merge_instance_data(to_data, from_data):
    """Merge instance data from from_data into to_data."""
    # Use current session_id from source (overwrites previous)
    to_data['session_id'] = from_data.get('session_id', to_data.get('session_id', ''))

    # Update transient fields from source
    to_data['pid'] = os.getppid()  # Always use current PID
    to_data['transcript_path'] = from_data.get('transcript_path', to_data.get('transcript_path', ''))

    # Preserve maximum position
    to_data['pos'] = max(to_data.get('pos', 0), from_data.get('pos', 0))

    # Update directory to most recent
    to_data['directory'] = from_data.get('directory', to_data.get('directory', str(Path.cwd())))

    # Update heartbeat timestamp to most recent
    to_data['last_stop'] = max(to_data.get('last_stop', 0), from_data.get('last_stop', 0))

    # Merge new status fields - take most recent status event
    from_time = from_data.get('last_status_time', 0)
    to_time = to_data.get('last_status_time', 0)
    if from_time > to_time:
        to_data['last_status'] = from_data.get('last_status', '')
        to_data['last_status_time'] = from_time
        to_data['last_status_context'] = from_data.get('last_status_context', '')

    # Preserve background mode if set
    to_data['background'] = to_data.get('background') or from_data.get('background')
    if from_data.get('background_log_file'):
        to_data['background_log_file'] = from_data['background_log_file']

    return to_data

def terminate_process(pid, force=False):
    """Cross-platform process termination"""
    try:
        if IS_WINDOWS:
            cmd = ['taskkill', '/PID', str(pid)]
            if force:
                cmd.insert(1, '/F')
            subprocess.run(cmd, capture_output=True, check=True)
        else:
            os.kill(pid, 9 if force else 15)  # SIGKILL or SIGTERM
        return True
    except (ProcessLookupError, OSError, subprocess.CalledProcessError):
        return False  # Process already dead

def merge_instance_immediately(from_name, to_name):
    """Merge from_name into to_name with safety checks. Returns success message or error message."""
    if from_name == to_name:
        return ""

    try:
        from_data = load_instance_position(from_name)
        to_data = load_instance_position(to_name)

        # Check if target has recent activity (time-based check instead of PID)
        now = time.time()
        last_activity = max(
            to_data.get('last_stop', 0),
            to_data.get('last_status_time', 0)
        )
        time_since_activity = now - last_activity
        if time_since_activity < MERGE_ACTIVITY_THRESHOLD:
            return f"Cannot recover {to_name}: instance is active (activity {int(time_since_activity)}s ago)"

        # Merge data using helper
        to_data = merge_instance_data(to_data, from_data)

        # Save merged data - check for success
        if not save_instance_position(to_name, to_data):
            return f"Failed to save merged data for {to_name}"

        # Cleanup source file only after successful save
        try:
            hcom_path(INSTANCES_DIR, f"{from_name}.json").unlink()
        except (FileNotFoundError, PermissionError, OSError):
            pass  # Non-critical if cleanup fails

        return f"[SUCCESS] ✓ Recovered alias: {to_name}"
    except Exception:
        return f"Failed to recover alias: {to_name}"


# ==================== Command Functions ====================

def show_main_screen_header():
    """Show header for main screen"""
    sys.stdout.write("\033[2J\033[H")
    
    log_file = hcom_path(LOG_FILE)
    all_messages = []
    if log_file.exists():
        all_messages = parse_log_messages(log_file).messages

    print(f"{BOLD}HCOM{RESET} LOGS")
    print(f"{DIM}{'─'*40}{RESET}\n")
    
    return all_messages

def show_cli_hints(to_stderr=True):
    """Show CLI hints if configured"""
    cli_hints = get_config_value('cli_hints', '')
    if cli_hints:
        if to_stderr:
            print(f"\n{cli_hints}", file=sys.stderr)
        else:
            print(f"\n{cli_hints}")

def cmd_help():
    """Show help text"""
    # Basic help for interactive users
    print("""hcom - Claude Hook Comms

Usage:
  hcom open [n]                Launch n Claude instances
  hcom open <agent>            Launch named agent from .claude/agents/
  hcom open --prefix <team> n  Launch n instances with team prefix
  hcom open --background       Launch instances as background processes (-p also works)
  hcom open --claude-args "--model sonnet"  Pass claude code CLI flags
  hcom watch                   View conversation dashboard
  hcom clear                   Clear and archive conversation
  hcom cleanup                 Remove hooks from current directory
  hcom cleanup --all           Remove hooks from all tracked directories
  hcom kill [instance alias]   Kill specific instance
  hcom kill --all              Kill all running instances
  hcom help                    Show this help

Automation:
  hcom send 'msg'              Send message to all
  hcom send '@prefix msg'      Send to specific instances
  hcom watch --logs            Show conversation log
  hcom watch --status          Show status of instances
  hcom watch --wait [seconds]  Wait for new messages (default 60s)

Docs: https://raw.githubusercontent.com/aannoo/claude-hook-comms/main/README.md""")

    # Additional help for AI assistants
    if os.environ.get('CLAUDECODE') == '1' or not sys.stdin.isatty():
        print("""

=== ADDITIONAL INFO ===

CONCEPT: HCOM launches Claude Code instances in new terminal windows.
They communicate with each other via a shared conversation.
You communicate with them via hcom automation commands.

KEY UNDERSTANDING:
• Single conversation - All instances share ~/.hcom/hcom.log
• Messaging - Use 'hcom send "message"' from CLI to send messages to instances
• Instances receive messages via hooks automatically and send with 'eval $HCOM send "message"'
• hcom open is directory-specific - always cd to project directory first
• hcom watch --wait outputs existing logs, then waits for the next message, prints it, and exits. 
Times out after [seconds]
• Named agents are custom system prompts created by users/claude code.
"reviewer" named agent loads .claude/agents/reviewer.md (if it was ever created)

LAUNCH PATTERNS:
  hcom open 2 reviewer                   # 2 generic + 1 reviewer agent
  hcom open reviewer reviewer            # 2 separate reviewer instances
  hcom open --prefix api 2               # Team naming: api-hova7, api-kolec
  hcom open --claude-args "--model sonnet"  # Pass 'claude' CLI flags
  hcom open --background (or -p) then hcom kill  # Detached background process
  hcom watch --status (get sessionid) then hcom open --claude-args "--resume <sessionid>"
  HCOM_INITIAL_PROMPT="do x task" hcom open  # initial prompt to instance

@MENTION TARGETING:
  hcom send "message"           # Broadcasts to everyone
  hcom send "@api fix this"     # Targets all api-* instances (api-hova7, api-kolec)
  hcom send "@hova7 status?"    # Targets specific instance
  (Unmatched @mentions broadcast to everyone)

STATUS INDICATORS:
• ▶ active - instance is working (processing/executing)
• ▷ delivered - instance just received a message
• ◉ waiting - instance is waiting for new messages
• ■ blocked - instance is blocked by permission request (needs user approval)
• ○ inactive - instance is timed out, disconnected, etc
• ○ unknown - no status information available
              
CONFIG:
Config file (persistent): ~/.hcom/config.json

Key settings (full list in config.json):
  terminal_mode: "new_window" (default) | "same_terminal" | "show_commands"
  initial_prompt: "Say hi in chat", first_use_text: "Essential messages only..."
  instance_hints: "text", cli_hints: "text"  # Extra info for instances/CLI
  env_overrides: "custom environment variables for instances"

Temporary environment overrides for any setting (all caps & append HCOM_):
HCOM_INSTANCE_HINTS="useful info" hcom open  # applied to all messages received by instance
export HCOM_CLI_HINTS="useful info" && hcom send 'hi'  # applied to all cli commands

EXPECT: hcom instance aliases are auto-generated (5-char format: "hova7"). Check actual aliases 
with 'hcom watch --status'. Instances respond automatically in shared chat.

Run 'claude --help' to see all claude code CLI flags.""")

        show_cli_hints(to_stderr=False)
    else:
        if not IS_WINDOWS:
            print("\nFor additional info & examples: hcom --help | cat")

    return 0

def cmd_open(*args):
    """Launch Claude instances with chat enabled"""
    try:
        # Parse arguments
        instances, prefix, claude_args, background = parse_open_args(list(args))

        # Add -p flag and stream-json output for background mode if not already present
        if background and '-p' not in claude_args and '--print' not in claude_args:
            claude_args = ['-p', '--output-format', 'stream-json', '--verbose'] + (claude_args or [])
        
        terminal_mode = get_config_value('terminal_mode', 'new_window')
        
        # Fail fast for same_terminal with multiple instances
        if terminal_mode == 'same_terminal' and len(instances) > 1:
            print(format_error(
                f"same_terminal mode cannot launch {len(instances)} instances",
                "Use 'hcom open' for one generic instance or 'hcom open <agent>' for one agent"
            ), file=sys.stderr)
            return 1
        
        try:
            setup_hooks()
        except Exception as e:
            print(format_error(f"Failed to setup hooks: {e}"), file=sys.stderr)
            return 1
        
        log_file = hcom_path(LOG_FILE, ensure_parent=True)
        instances_dir = hcom_path(INSTANCES_DIR)
        instances_dir.mkdir(exist_ok=True)
        
        if not log_file.exists():
            log_file.touch()
        
        # Build environment variables for Claude instances
        base_env = build_claude_env()

        # Add prefix-specific hints if provided
        if prefix:
            base_env['HCOM_PREFIX'] = prefix
            send_cmd = build_send_command()
            hint = f"To respond to {prefix} group: {send_cmd} '@{prefix} message'"
            base_env['HCOM_INSTANCE_HINTS'] = hint
            first_use = f"You're in the {prefix} group. Use {prefix} to message: {send_cmd} '@{prefix} message'"
            base_env['HCOM_FIRST_USE_TEXT'] = first_use
        
        launched = 0
        initial_prompt = get_config_value('initial_prompt', 'Say hi in chat')
        
        for _, instance_type in enumerate(instances):
            instance_env = base_env.copy()

            # Set unique launch ID for sender detection in cmd_send()
            launch_id = f"{int(time.time())}_{random.randint(10000, 99999)}"
            instance_env['HCOM_LAUNCH_ID'] = launch_id

            # Mark background instances via environment with log filename
            if background:
                # Generate unique log filename
                log_filename = f'background_{int(time.time())}_{random.randint(1000, 9999)}.log'
                instance_env['HCOM_BACKGROUND'] = log_filename
            
            # Build claude command
            if instance_type == 'generic':
                # Generic instance - no agent content
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
        
        requested = len(instances)
        failed = requested - launched

        if launched == 0:
            print(format_error(f"No instances launched (0/{requested})"), file=sys.stderr)
            return 1

        # Show results
        if failed > 0:
            print(f"Launched {launched}/{requested} Claude instance{'s' if requested != 1 else ''} ({failed} failed)")
        else:
            print(f"Launched {launched} Claude instance{'s' if launched != 1 else ''}")

        # Auto-launch watch dashboard if configured and conditions are met
        terminal_mode = get_config_value('terminal_mode')
        auto_watch = get_config_value('auto_watch', True)

        # Only auto-watch if ALL instances launched successfully
        if terminal_mode == 'new_window' and auto_watch and failed == 0 and is_interactive():
            # Show tips first if needed
            if prefix:
                print(f"\n  • Send to {prefix} team: hcom send '@{prefix} message'")

            # Clear transition message
            print("\nOpening hcom watch...")
            time.sleep(2)  # Brief pause so user sees the message

            # Launch interactive watch dashboard in current terminal
            return cmd_watch()
        else:
            tips = [
                "Run 'hcom watch' to view/send in conversation dashboard",
            ]
            if prefix:
                tips.append(f"Send to {prefix} team: hcom send '@{prefix} message'")

            if tips:
                print("\n" + "\n".join(f"  • {tip}" for tip in tips) + "\n")

            # Show cli_hints if configured (non-interactive mode)
            if not is_interactive():
                show_cli_hints(to_stderr=False)

            return 0
        
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 1
    except Exception as e:
        print(str(e), file=sys.stderr)
        return 1

def cmd_watch(*args):
    """View conversation dashboard"""
    log_file = hcom_path(LOG_FILE)
    instances_dir = hcom_path(INSTANCES_DIR)
    
    if not log_file.exists() and not instances_dir.exists():
        print(format_error("No conversation log found", "Run 'hcom open' first"), file=sys.stderr)
        return 1
    
    # Parse arguments
    show_logs = False
    show_status = False
    wait_timeout = None
    
    i = 0
    while i < len(args):
        arg = args[i]
        if arg == '--logs':
            show_logs = True
        elif arg == '--status':
            show_status = True
        elif arg == '--wait':
            # Check if next arg is a number
            if i + 1 < len(args) and args[i + 1].isdigit():
                wait_timeout = int(args[i + 1])
                i += 1  # Skip the number
            else:
                wait_timeout = 60  # Default
        i += 1
    
    # If wait is specified, enable logs to show the messages
    if wait_timeout is not None:
        show_logs = True
    
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
            
            # If --wait, show only recent messages to prevent context bloat
            if wait_timeout is not None:
                cutoff = datetime.now() - timedelta(seconds=5)
                recent_messages = [m for m in messages if datetime.fromisoformat(m['timestamp']) > cutoff]
                
                # Status to stderr, data to stdout
                if recent_messages:
                    print(f'---Showing last 5 seconds of messages---', file=sys.stderr) #TODO: change this to recent messages and have logic like last 3 messages + all messages in last 5 seconds.
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
            
            show_cli_hints()
                    
        elif show_status:
            # Build JSON output
            positions = load_all_positions()

            instances = {}
            status_counts = {}

            for name, data in positions.items():
                status, age = get_instance_status(data)
                instances[name] = {
                    "status": status,
                    "age": age.strip() if age else "",
                    "directory": data.get("directory", "unknown"),
                    "session_id": data.get("session_id", ""),
                    "last_status": data.get("last_status", ""),
                    "last_status_time": data.get("last_status_time", 0),
                    "last_status_context": data.get("last_status_context", ""),
                    "pid": data.get("pid"),
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
            show_cli_hints()
        else:
            print("No TTY - Automation usage:", file=sys.stderr)
            print("  hcom send 'message'    Send message to chat", file=sys.stderr)
            print("  hcom watch --logs      Show message history", file=sys.stderr)
            print("  hcom watch --status    Show instance status", file=sys.stderr)
            print("  hcom watch --wait      Wait for new messages", file=sys.stderr)
            print("  Full information: hcom --help")
            
            show_cli_hints()
        
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
                    sender_name = get_config_value('sender_name', 'bigboss')
                    send_message(sender_name, message.strip())
                    print(f"{FG_GREEN}✓ Sent{RESET}")
                
                print()
                
                current_status = get_status_summary()
                update_status(f"{current_status}{status_suffix}")
            
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        sys.stdout.write("\033[?1049l\r\033[K")
        print(f"\n{DIM}[stopped]{RESET}")
        
    return 0

def cmd_clear():
    """Clear and archive conversation"""
    log_file = hcom_path(LOG_FILE, ensure_parent=True)
    instances_dir = hcom_path(INSTANCES_DIR)
    archive_folder = hcom_path(ARCHIVE_DIR)
    archive_folder.mkdir(exist_ok=True)

    # Clean up temp files from failed atomic writes
    if instances_dir.exists():
        deleted_count = sum(1 for f in instances_dir.glob('*.tmp') if f.unlink(missing_ok=True) is None)
        if deleted_count > 0:
            print(f"Cleaned up {deleted_count} temp files")

    # Clean up old script files (older than 24 hours)
    scripts_dir = hcom_path(SCRIPTS_DIR)
    if scripts_dir.exists():
        cutoff_time = time.time() - (24 * 60 * 60)  # 24 hours ago
        script_count = sum(1 for f in scripts_dir.glob('*') if f.is_file() and f.stat().st_mtime < cutoff_time and f.unlink(missing_ok=True) is None)
        if script_count > 0:
            print(f"Cleaned up {script_count} old script files")

    # Clean up old launch mapping files (older than 24 hours)
    if instances_dir.exists():
        cutoff_time = time.time() - (24 * 60 * 60)  # 24 hours ago
        mapping_count = sum(1 for f in instances_dir.glob('.launch_map_*') if f.is_file() and f.stat().st_mtime < cutoff_time and f.unlink(missing_ok=True) is None)
        if mapping_count > 0:
            print(f"Cleaned up {mapping_count} old launch mapping files")

    # Check if hcom files exist
    if not log_file.exists() and not instances_dir.exists():
        print("No hcom conversation to clear")
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
            session_archive.mkdir(exist_ok=True)
            
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
                archive_instances.mkdir(exist_ok=True)

                # Move json files only
                for f in instances_dir.glob('*.json'):
                    f.rename(archive_instances / f.name)

                # Clean up orphaned mapping files (position files are archived)
                for f in instances_dir.glob('.launch_map_*'):
                    f.unlink(missing_ok=True)

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
        print("Started fresh hcom conversation log")
        return 0
        
    except Exception as e:
        print(format_error(f"Failed to archive: {e}"), file=sys.stderr)
        return 1

def cleanup_directory_hooks(directory):
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
        settings = read_file_with_retry(
            settings_path,
            lambda f: json.load(f),
            default=None
        )
        if not settings:
            return 1, "Cannot read Claude settings"
        
        hooks_found = False
        
        # Include PostToolUse for backward compatibility cleanup
        original_hook_count = sum(len(settings.get('hooks', {}).get(event, []))
                                  for event in ['SessionStart', 'UserPromptSubmit', 'PreToolUse', 'PostToolUse', 'Stop', 'Notification'])

        _remove_hcom_hooks_from_settings(settings)

        # Check if any were removed
        new_hook_count = sum(len(settings.get('hooks', {}).get(event, []))
                             for event in ['SessionStart', 'UserPromptSubmit', 'PreToolUse', 'PostToolUse', 'Stop', 'Notification'])
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


def cmd_kill(*args):
    """Kill instances by name or all with --all"""

    instance_name = args[0] if args and args[0] != '--all' else None
    positions = load_all_positions() if not instance_name else {instance_name: load_instance_position(instance_name)}

    # Filter to instances with PIDs (any instance that's running)
    targets = [(name, data) for name, data in positions.items() if data.get('pid')]

    if not targets:
        print(f"No running process found for {instance_name}" if instance_name else "No running instances found")
        return 1 if instance_name else 0

    killed_count = 0
    for target_name, target_data in targets:
        status, age = get_instance_status(target_data)
        instance_type = "background" if target_data.get('background') else "foreground"

        pid = int(target_data['pid'])
        try:
            # Try graceful termination first
            terminate_process(pid, force=False)

            # Wait for process to exit gracefully
            for _ in range(20):
                time.sleep(KILL_CHECK_INTERVAL)
                if not is_process_alive(pid):
                    # Process terminated successfully
                    break
            else:
                # Process didn't die from graceful attempt, force kill
                terminate_process(pid, force=True)
                time.sleep(0.1)

            print(f"Killed {target_name} ({instance_type}, {status}{age}, PID {pid})")
            killed_count += 1
        except (TypeError, ValueError) as e:
            print(f"Process {pid} invalid: {e}")

        # Mark instance as killed
        update_instance_position(target_name, {'pid': None})
        set_status(target_name, 'killed')

    if not instance_name:
        print(f"Killed {killed_count} instance(s)")

    return 0

def cmd_cleanup(*args):
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
        
        if not directories:
            print("No directories found in current hcom tracking")
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

def cmd_send(message):
    """Send message to hcom"""
    # Check if hcom files exist
    log_file = hcom_path(LOG_FILE)
    instances_dir = hcom_path(INSTANCES_DIR)

    if not log_file.exists() and not instances_dir.exists():
        print(format_error("No conversation found", "Run 'hcom open' first"), file=sys.stderr)
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
            unmatched = [m for m in mentions
                        if not any(name.lower().startswith(m.lower()) for name in all_instances)]
            if unmatched:
                print(f"Note: @{', @'.join(unmatched)} don't match any instances - broadcasting to all", file=sys.stderr)
        except Exception:
            pass  # Don't fail on warning

    # Determine sender: lookup by launch_id, fallback to config
    sender_name = None
    launch_id = os.environ.get('HCOM_LAUNCH_ID')
    if launch_id:
        try:
            mapping_file = hcom_path(INSTANCES_DIR, f'.launch_map_{launch_id}')
            if mapping_file.exists():
                sender_name = mapping_file.read_text(encoding='utf-8').strip()
        except Exception:
            pass

    if not sender_name:
        sender_name = get_config_value('sender_name', 'bigboss')

    if send_message(sender_name, message):
        # For instances: check for new messages and display immediately
        if launch_id:  # Only for instances with HCOM_LAUNCH_ID
            messages = get_unread_messages(sender_name, update_position=True)
            if messages:
                max_msgs = get_config_value('max_messages_per_delivery', 50)
                messages_to_show = messages[:max_msgs]
                formatted = format_hook_messages(messages_to_show, sender_name)
                print(f"Message sent\n\n{formatted}", file=sys.stderr)
            else:
                print("Message sent", file=sys.stderr)
        else:
            # Bigboss: just confirm send
            print("Message sent", file=sys.stderr)
        
        # Show cli_hints if configured (non-interactive mode)
        if not is_interactive():
            show_cli_hints()
        
        return 0
    else:
        print(format_error("Failed to send message"), file=sys.stderr)
        return 1

def cmd_resume_merge(alias: str) -> int:
    """Resume/merge current instance into an existing instance by alias.

    INTERNAL COMMAND: Only called via '$HCOM send --resume alias' during implicit resume workflow.
    Not meant for direct CLI usage.
    """
    # Get current instance name via launch_id mapping (same mechanism as cmd_send)
    # The mapping is created by init_hook_context() when hooks run
    launch_id = os.environ.get('HCOM_LAUNCH_ID')
    if not launch_id:
        print(format_error("Not in HCOM instance context - no launch ID"), file=sys.stderr)
        return 1

    instance_name = None
    try:
        mapping_file = hcom_path(INSTANCES_DIR, f'.launch_map_{launch_id}')
        if mapping_file.exists():
            instance_name = mapping_file.read_text(encoding='utf-8').strip()
    except Exception:
        pass

    if not instance_name:
        print(format_error("Could not determine instance name"), file=sys.stderr)
        return 1

    # Sanitize alias: only allow alphanumeric, dash, underscore
    # This prevents path traversal attacks (e.g., ../../etc, /etc, etc.)
    if not re.match(r'^[A-Za-z0-9\-_]+$', alias):
        print(format_error("Invalid alias format. Use alphanumeric, dash, or underscore only"), file=sys.stderr)
        return 1

    # Attempt to merge current instance into target alias
    status = merge_instance_immediately(instance_name, alias)

    # Handle results
    if not status:
        # Empty status means names matched (from_name == to_name)
        status = f"[SUCCESS] ✓ Already using alias {alias}"

    # Print status and return
    print(status, file=sys.stderr)
    return 0 if status.startswith('[SUCCESS]') else 1

# ==================== Hook Helpers ====================

def format_hook_messages(messages, instance_name):
    """Format messages for hook feedback"""
    if len(messages) == 1:
        msg = messages[0]
        reason = f"[new message] {msg['from']} → {instance_name}: {msg['message']}"
    else:
        parts = [f"{msg['from']} → {instance_name}: {msg['message']}" for msg in messages]
        reason = f"[{len(messages)} new messages] | {' | '.join(parts)}"

    # Only append instance_hints to messages (first_use_text is handled separately)
    instance_hints = get_config_value('instance_hints', '')
    if instance_hints:
        reason = f"{reason} | [{instance_hints}]"

    return reason

# ==================== Hook Handlers ====================

def init_hook_context(hook_data, hook_type=None):
    """Initialize instance context - shared by post/stop/notify hooks"""
    import time

    session_id = hook_data.get('session_id', '')
    transcript_path = hook_data.get('transcript_path', '')
    prefix = os.environ.get('HCOM_PREFIX')

    instances_dir = hcom_path(INSTANCES_DIR)
    instance_name = None
    merged_state = None

    # Check if current session_id matches any existing instance
    # This maintains identity after resume/merge operations
    if not instance_name and session_id and instances_dir.exists():
        for instance_file in instances_dir.glob("*.json"):
            try:
                data = load_instance_position(instance_file.stem)
                if session_id == data.get('session_id'):
                    instance_name = instance_file.stem
                    merged_state = data
                    log_hook_error(f'DEBUG: Session_id {session_id[:8]} matched {instance_file.stem}, reusing that name')
                    break
            except (json.JSONDecodeError, OSError, KeyError):
                continue

    # If not found or not resuming, generate new name from session_id
    if not instance_name:
        instance_name = get_display_name(session_id, prefix)
        # DEBUG: Log name generation
        log_hook_error(f'DEBUG: Generated instance_name={instance_name} from session_id={session_id[:8] if session_id else "None"}')

    # Save launch_id → instance_name mapping for cmd_send()
    launch_id = os.environ.get('HCOM_LAUNCH_ID')
    if launch_id:
        try:
            mapping_file = hcom_path(INSTANCES_DIR, f'.launch_map_{launch_id}', ensure_parent=True)
            mapping_file.write_text(instance_name, encoding='utf-8')
            log_hook_error(f'DEBUG: FINAL - Wrote launch_map_{launch_id} → {instance_name} (session_id={session_id[:8] if session_id else "None"})')
        except Exception:
            pass  # Non-critical

    # Save migrated data if we have it
    if merged_state:
        save_instance_position(instance_name, merged_state)

    # Check if instance is brand new or pre-existing (before creation (WWJD))
    instance_file = hcom_path(INSTANCES_DIR, f'{instance_name}.json')
    is_new_instance = not instance_file.exists()

    # Skip instance creation for unmatched SessionStart resumes (prevents orphans)
    # Instance will be created in UserPromptSubmit with correct session_id
    should_create_instance = not (
        hook_type == 'sessionstart' and
            hook_data.get('source', 'startup') == 'resume' and not merged_state
    )
    if should_create_instance:
        initialize_instance_in_position_file(instance_name, session_id)
    existing_data = load_instance_position(instance_name) if should_create_instance else {}

    # Prepare updates
    updates: dict[str, Any] = {
        'directory': str(Path.cwd()),
    }

    # Update session_id (overwrites previous)
    if session_id:
        updates['session_id'] = session_id

    # Update transcript_path to current
    if transcript_path:
        updates['transcript_path'] = transcript_path

    # Always update PID to current (fixes stale PID on implicit resume)
    updates['pid'] = os.getppid()

    # Add background status if applicable
    bg_env = os.environ.get('HCOM_BACKGROUND')
    if bg_env:
        updates['background'] = True
        updates['background_log_file'] = str(hcom_path(LOGS_DIR, bg_env))

    # Return flags indicating resume state
    is_resume_match = merged_state is not None
    return instance_name, updates, existing_data, is_resume_match, is_new_instance

def handle_pretooluse(hook_data, instance_name, updates):
    """Handle PreToolUse hook - auto-approve HCOM_SEND commands when safe"""
    tool_name = hook_data.get('tool_name', '')

    # Non-HCOM_SEND tools: record status (they'll run without permission check)
    set_status(instance_name, 'tool_pending', tool_name)

    import time

    # Handle HCOM commands in Bash
    if tool_name == 'Bash':
        command = hook_data.get('tool_input', {}).get('command', '')
        script_path = str(Path(__file__).resolve())

        # === Auto-approve ALL '$HCOM send' commands (including --resume) ===
        # This includes:
        #   - $HCOM send "message" (normal messaging between instances)
        #   - $HCOM send --resume alias (resume/merge operation)
        if ('$HCOM send' in command or
            'hcom send' in command or
            (script_path in command and ' send ' in command)):
            output = {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "allow",
                    "permissionDecisionReason": "HCOM send command auto-approved"
                }
            }
            print(json.dumps(output, ensure_ascii=False))
            sys.exit(EXIT_SUCCESS)


def safe_exit_with_status(instance_name, code=EXIT_SUCCESS):
    """Safely exit stop hook with proper status tracking"""
    try:
        set_status(instance_name, 'stop_exit')
    except (OSError, PermissionError):
        pass  # Silently handle any errors
    sys.exit(code)

def handle_stop(hook_data, instance_name, updates):
    """Handle Stop hook - poll for messages and deliver"""
    import time as time_module

    parent_pid = os.getppid()
    log_hook_error(f'stop:entering_stop_hook_now_pid_{os.getpid()}')
    log_hook_error(f'stop:entering_stop_hook_now_ppid_{parent_pid}')


    try:
        entry_time = time_module.time()
        updates['last_stop'] = entry_time
        timeout = get_config_value('wait_timeout', 1800)
        updates['wait_timeout'] = timeout
        set_status(instance_name, 'waiting')

        try:
            update_instance_position(instance_name, updates)
        except Exception as e:
            log_hook_error(f'stop:update_instance_position({instance_name})', e)

        start_time = time_module.time()
        log_hook_error(f'stop:start_time_pid_{os.getpid()}')

        try:
            loop_count = 0
            # STEP 4: Actual polling loop - this IS the holding pattern
            while time_module.time() - start_time < timeout:
                if loop_count == 0:
                    time_module.sleep(0.1)  # Initial wait before first poll
                loop_count += 1

                # Check if parent is alive
                if not IS_WINDOWS and os.getppid() == 1:
                    log_hook_error(f'stop:parent_died_pid_{os.getpid()}')
                    safe_exit_with_status(instance_name, EXIT_SUCCESS)

                parent_alive = is_parent_alive(parent_pid)
                if not parent_alive:
                    log_hook_error(f'stop:parent_not_alive_pid_{os.getpid()}')
                    safe_exit_with_status(instance_name, EXIT_SUCCESS)

                # Check if user input is pending - exit cleanly if so
                user_input_signal = hcom_path(INSTANCES_DIR, f'.user_input_pending_{instance_name}')
                if user_input_signal.exists():
                    log_hook_error(f'stop:user_input_pending_exiting_pid_{os.getpid()}')
                    safe_exit_with_status(instance_name, EXIT_SUCCESS)

                # Check for new messages and deliver
                if messages := get_unread_messages(instance_name, update_position=True):
                    messages_to_show = messages[:get_config_value('max_messages_per_delivery', 50)]
                    reason = format_hook_messages(messages_to_show, instance_name)
                    set_status(instance_name, 'message_delivered', messages_to_show[0]['from'])

                    log_hook_error(f'stop:delivering_message_pid_{os.getpid()}')
                    output = {"decision": "block", "reason": reason}
                    print(json.dumps(output, ensure_ascii=False), file=sys.stderr)
                    sys.exit(EXIT_BLOCK)

                # Update heartbeat
                try:
                    update_instance_position(instance_name, {'last_stop': time_module.time()})
                    # log_hook_error(f'hb_pid_{os.getpid()}')
                except Exception as e:
                    log_hook_error(f'stop:heartbeat_update({instance_name})', e)

                time_module.sleep(STOP_HOOK_POLL_INTERVAL)

        except Exception as loop_e:
            # Log polling loop errors but continue to cleanup
            log_hook_error(f'stop:polling_loop({instance_name})', loop_e)

        # Timeout reached
        set_status(instance_name, 'timeout')

    except Exception as e:
        # Log error and exit gracefully
        log_hook_error('handle_stop', e)
        safe_exit_with_status(instance_name, EXIT_SUCCESS)

def handle_notify(hook_data, instance_name, updates):
    """Handle Notification hook - track permission requests"""
    updates['notification_message'] = hook_data.get('message', '')
    update_instance_position(instance_name, updates)
    set_status(instance_name, 'blocked', hook_data.get('message', ''))

def handle_userpromptsubmit(hook_data, instance_name, updates, is_resume_match, is_new_instance):
    """Handle UserPromptSubmit hook - track when user sends messages"""
    import time as time_module

    # Update last user input timestamp
    updates['last_user_input'] = time_module.time()
    update_instance_position(instance_name, updates)

    # Signal any polling Stop hook to exit cleanly before user input processed
    signal_file = hcom_path(INSTANCES_DIR, f'.user_input_pending_{instance_name}')
    try:
        log_hook_error(f'userpromptsubmit:signal_file_touched_pid_{os.getpid()}')
        signal_file.touch()
        time_module.sleep(0.15)  # Give Stop hook time to detect and exit
        log_hook_error(f'userpromptsubmit:signal_file_unlinked_pid_{os.getpid()}')
        signal_file.unlink()
    except (OSError, PermissionError) as e:
        log_hook_error(f'userpromptsubmit:signal_file_error', e)

    send_cmd = build_send_command('your message')
    resume_cmd = send_cmd.replace("'your message'", "--resume your_old_alias")
    # Build message based on what happened
    msg = None
    if is_resume_match:
        msg = f"[Resumed with hcom chat alias: {instance_name}. Use {send_cmd} to send messages]"
    elif is_new_instance:
        # Unmatched resume - show critical recovery message
        msg = (
            f"[HCOM RESUME DETECTED - CRITICAL ACTION REQUIRED "
            f"You MUST recover your HCOM identity to maintain conversation context "
            f"Run: {resume_cmd} "
            f"This is REQUIRED for message history and position tracking]"
        )
    else:
        # Check if we need to announce alias (normal startup)
        instance_data = load_instance_position(instance_name)
        alias_announced = instance_data.get('alias_announced', False)
        if not alias_announced:
            msg = f"[Your hcom chat alias is {instance_name}. You can at-mention others in hcom chat by their alias to DM them. To send a message use: {send_cmd}]"
            update_instance_position(instance_name, {'alias_announced': True})

    if msg:
        output = {"hookSpecificOutput": {"hookEventName": "UserPromptSubmit", "additionalContext": msg}}
        print(json.dumps(output))

def handle_sessionstart(hook_data, instance_name, updates, is_resume_match):
    """Handle SessionStart hook - deliver welcome/resume message"""
    source = hook_data.get('source', 'startup')

    log_hook_error(f'sessionstart:is_resume_match_{is_resume_match}')
    log_hook_error(f'sessionstart:instance_name_{instance_name}')
    log_hook_error(f'sessionstart:source_{source}')
    log_hook_error(f'sessionstart:updates_{updates}')
    log_hook_error(f'sessionstart:hook_data_{hook_data}')

    # Reset alias_announced flag so alias shows again on resume/clear/compact
    updates['alias_announced'] = False

    # Only update instance position if file exists (startup or matched resume)
    # For unmatched resumes, skip - UserPromptSubmit will create the file with correct session_id
    if source == 'startup' or is_resume_match:
        update_instance_position(instance_name, updates)
        set_status(instance_name, 'session_start')

    log_hook_error(f'sessionstart:instance_name_after_update_{instance_name}')

    # Build send command using helper
    send_cmd = build_send_command('your message')
    help_text = f"[Welcome! HCOM chat active. Send messages: {send_cmd}]"

    # Add subagent type if this is a named agent
    subagent_type = os.environ.get('HCOM_SUBAGENT_TYPE')
    if subagent_type:
        help_text += f" [Subagent: {subagent_type}]"

    # Add first use text only on startup
    if source == 'startup':
        first_use_text = get_config_value('first_use_text', '')
        if first_use_text:
            help_text += f" [{first_use_text}]"
    elif source == 'resume':
        if is_resume_match:
            help_text += f" [Resumed alias: {instance_name}]"
        else:
            help_text += f" [Session resumed]"

    # Add instance hints to all messages
    instance_hints = get_config_value('instance_hints', '')
    if instance_hints:
        help_text += f" [{instance_hints}]"

    # Output as additionalContext using hookSpecificOutput format
    output = {
        "hookSpecificOutput": {
            "hookEventName": "SessionStart",
            "additionalContext": help_text
        }
    }
    print(json.dumps(output))

def handle_hook(hook_type: str) -> None:
    """Unified hook handler for all HCOM hooks"""
    if os.environ.get(HCOM_ACTIVE_ENV) != HCOM_ACTIVE_VALUE:
        sys.exit(EXIT_SUCCESS)

    hook_data = json.load(sys.stdin)
    log_hook_error(f'handle_hook:hook_data_{hook_data}')

    # DEBUG: Log which hook is being called with which session_id
    session_id_short = hook_data.get('session_id', 'none')[:8] if hook_data.get('session_id') else 'none'
    log_hook_error(f'DEBUG: Hook {hook_type} called with session_id={session_id_short}')

    instance_name, updates, _, is_resume_match, is_new_instance = init_hook_context(hook_data, hook_type)

    match hook_type:
        case 'pre':
            handle_pretooluse(hook_data, instance_name, updates)
        case 'stop':
            handle_stop(hook_data, instance_name, updates)
        case 'notify':
            handle_notify(hook_data, instance_name, updates)
        case 'userpromptsubmit':
            handle_userpromptsubmit(hook_data, instance_name, updates, is_resume_match, is_new_instance)
        case 'sessionstart':
            handle_sessionstart(hook_data, instance_name, updates, is_resume_match)

    log_hook_error(f'handle_hook:instance_name_{instance_name}')


    sys.exit(EXIT_SUCCESS)


# ==================== Main Entry Point ====================

def main(argv=None):
    """Main command dispatcher"""
    if argv is None:
        argv = sys.argv
    
    if len(argv) < 2:
        return cmd_help()
    
    cmd = argv[1]
    
    match cmd:
        case 'help' | '--help':
            return cmd_help()
        case 'open':
            return cmd_open(*argv[2:])
        case 'watch':
            return cmd_watch(*argv[2:])
        case 'clear':
            return cmd_clear()
        case 'cleanup':
            return cmd_cleanup(*argv[2:])
        case 'send':
            if len(argv) < 3:
                print(format_error("Message required"), file=sys.stderr)
                return 1

            # HIDDEN COMMAND: --resume is only used internally by instances during resume workflow
            # Not meant for regular CLI usage. Primary usage:
            #   - From instances: $HCOM send "message" (instances send messages to each other)
            #   - From CLI: hcom send "message" (user/claude orchestrator sends to instances)
            if argv[2] == '--resume':
                if len(argv) < 4:
                    print(format_error("Alias required for --resume"), file=sys.stderr)
                    return 1
                return cmd_resume_merge(argv[3])

            return cmd_send(argv[2])
        case 'kill':
            return cmd_kill(*argv[2:])
        case 'stop' | 'notify' | 'pre' | 'sessionstart' | 'userpromptsubmit':
            handle_hook(cmd)
            return 0
        case _:
            print(format_error(f"Unknown command: {cmd}", "Run 'hcom help' for available commands"), file=sys.stderr)
            return 1

if __name__ == '__main__':
    sys.exit(main())
