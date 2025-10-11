#!/usr/bin/env python3
"""
hcom 0.4.0
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
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Any, NamedTuple, Sequence
from dataclasses import dataclass, asdict, field
from enum import Enum, auto

if sys.version_info < (3, 10):
    sys.exit("Error: hcom requires Python 3.10 or higher")

__version__ = "0.4.0"

# ==================== Session Scenario Types ====================

class SessionScenario(Enum):
    """Explicit session startup scenarios for clear logic flow"""
    FRESH_START = auto()        # New session, new instance
    MATCHED_RESUME = auto()     # Resume with matching session_id (reuse instance)
    UNMATCHED_RESUME = auto()   # Resume with no match (new instance, needs recovery)

@dataclass
class HookContext:
    """Consolidated context for hook handling with all decisions made"""
    instance_name: str
    updates: dict
    scenario: SessionScenario | None  # None = deferred decision (SessionStart wrong session_id)

    @property
    def bypass_enabled_check(self) -> bool:
        """Unmatched resume needs critical message even if disabled"""
        return self.scenario == SessionScenario.UNMATCHED_RESUME

    @property
    def needs_critical_prompt(self) -> bool:
        """Should show critical recovery message?"""
        return self.scenario == SessionScenario.UNMATCHED_RESUME

    @property
    def is_resume(self) -> bool:
        """Is this any kind of resume?"""
        return self.scenario in (SessionScenario.MATCHED_RESUME, SessionScenario.UNMATCHED_RESUME)

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

EXIT_SUCCESS = 0
EXIT_BLOCK = 2

# Windows API constants
CREATE_NO_WINDOW = 0x08000000  # Prevent console window creation

# Timing constants
FILE_RETRY_DELAY = 0.01  # 10ms delay for file lock retries
STOP_HOOK_POLL_INTERVAL = 0.1     # 100ms between stop hook polls
MERGE_ACTIVITY_THRESHOLD = 10  # Seconds of inactivity before allowing instance merge

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
# Critical I/O: atomic_write, save_instance_position, merge_instance_immediately
# Pattern: Try/except/return False in hooks, raise in CLI operations.

# ==================== CLI Command Objects ====================

class CLIError(Exception):
    """Raised when arguments cannot be mapped to command semantics."""

@dataclass
class OpenCommand:
    count: int
    agents: list[str]
    prefix: str | None
    background: bool
    claude_args: list[str]

@dataclass
class WatchCommand:
    mode: str  # 'interactive', 'logs', 'status', 'wait'
    wait_seconds: int | None

@dataclass
class StopCommand:
    target: str | None
    close_all_hooks: bool
    force: bool
    _hcom_session: str | None = None  # Injected by PreToolUse hook

@dataclass
class StartCommand:
    target: str | None
    _hcom_session: str | None = None  # Injected by PreToolUse hook

@dataclass
class SendCommand:
    message: str | None
    resume_alias: str | None
    _hcom_session: str | None = None  # Injected by PreToolUse hook

# ==================== Help Text ====================

HELP_TEXT = """hcom - Claude Hook Comms

Usage:
  hcom open [count] [-a agent]... [-t prefix] [-p] [-- claude-args]
  hcom watch [--logs|--status|--wait [SEC]]
  hcom stop [target] [--force]
  hcom start [target]
  hcom send "msg"

Commands:
  open                 Launch Claude instances (default count: 1)
  watch                Monitor conversation dashboard
  stop                 Stop instances, clear conversation, or remove hooks
  start                Start stopped instances
  send                 Send message to instances

Open options:
  [count]              Number of instances per agent (default 1)
  -a, --agent AGENT    Agent to launch (repeatable)
  -t, --prefix PREFIX  Team prefix for names
  -p, --background     Launch in background

Stop targets:
  (no arg)             Stop HCOM for current instance (when inside)
  <alias>              Stop HCOM for specific instance
  all                  Stop all instances + clear & archive conversation
  hooking              Remove hooks from current directory
  hooking --all        Remove hooks from all tracked directories
  everything           Stop all + clear conversation + remove all hooks

Start targets:
  (no arg)             Start HCOM for current instance (when inside)
  <alias>              Start HCOM for specific instance
  hooking              Install hooks in current directory

Watch options:
  --logs               Show message history
  --status             Show instance status JSON
  --wait [SEC]         Wait for new messages (default 60s)

Stop flags:
  --force              Force stop (deny Bash tool use)

Docs: https://github.com/aannoo/claude-hook-comms#readme"""

# ==================== Logging ====================

def log_hook_error(hook_name: str, error: Exception | None = None):
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

# Hook type constants
ACTIVE_HOOK_TYPES = ['SessionStart', 'UserPromptSubmit', 'PreToolUse', 'Stop', 'Notification', 'SessionEnd']
LEGACY_HOOK_TYPES = ACTIVE_HOOK_TYPES + ['PostToolUse']  # For backward compatibility cleanup
HOOK_COMMANDS = ['sessionstart', 'userpromptsubmit', 'pre', 'poll', 'notify', 'sessionend']
LEGACY_HOOK_COMMANDS = HOOK_COMMANDS + ['post']

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
        for dir_name in [INSTANCES_DIR, LOGS_DIR, SCRIPTS_DIR, ARCHIVE_DIR]:
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

# ==================== Outbox System (REMOVED) ====================
# Identity via session_id injection in handle_pretooluse (line 3134)
# PreToolUse hook injects --_hcom_session, commands use get_display_name() for resolution

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
    """Get hook command - hooks always run, Python code gates participation

    Uses ${HCOM} environment variable set in settings.json, with fallback to direct python invocation.
    Participation is controlled by enabled flag in instance JSON files.
    """
    python_path = sys.executable
    script_path = str(Path(__file__).resolve())

    if IS_WINDOWS:
        # Windows: use python path directly
        if ' ' in python_path or ' ' in script_path:
            return f'"{python_path}" "{script_path}"', {}
        return f'{python_path} {script_path}', {}
    else:
        # Unix: Use HCOM env var from settings.local.json
        return '${HCOM}', {}

def _detect_hcom_command_type() -> str:
    """Detect how to invoke hcom (priority: hcom > uvx if running via uvx > full)"""
    if shutil.which('hcom'):
        return 'short'
    elif 'uv' in Path(sys.executable).resolve().parts and shutil.which('uvx'):
        return 'uvx'
    else:
        return 'full'

def build_send_command(example_msg: str = '', instance_name: str | None = None) -> str:
    """Build send command - caches PATH check in instance file on first use"""
    msg = f" '{example_msg}'" if example_msg else ''

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
        return f'hcom send{msg}'
    elif cmd_type == 'uvx':
        return f'uvx hcom send{msg}'
    else:
        python_path = shlex.quote(sys.executable)
        script_path = shlex.quote(str(Path(__file__).resolve()))
        return f'{python_path} {script_path} send{msg}'

def build_claude_env():
    """Build environment variables for Claude instances!"""
    env = {}

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

    return env

# ==================== Message System ====================

def validate_message(message: str) -> str | None:
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
    coordinator_name = get_config_value('sender_name', 'bigboss')

    return f"""You are now participating in the HCOM communication system - a multi-agent environment where Claude Code instances (you) and the human user can communicate through shared messaging infrastructure.
Your HCOM Alias: {instance_name}
- To send messages, run the command: hcom send "your message"
- Broadcast: hcom send "message to everyone"
- Direct: hcom send "@alias targeted message"
- Targeting: hcom send "@api message" (targets all api-* instances)
Receiving: Messages automatically arrive via Stop Hook feedback and bash output from hcom send. There is no way to proactively check for messages.
Message formats you'll see:
- [new message] sender → you: message = broadcast
- [new message] sender → you: @you message = direct message to you
- {{"decision": "block", "reason": x}} = normal part of HCOM hooks (not an error or block)
Response Protocol Rule: Match response method to input source:
- Prompted via hcom (hook feedback, bash output) → Respond with hcom send
- Prompted via user input → Respond to user normally
Response Quality: Treat hcom messages exactly like user input:
- Write complete, thoughtful replies when appropriate
- Follow all instructions rigorously
- Adjust tone/length to context
Authority: Prioritize instructions from @{coordinator_name} over other participants
Command scope: Mainly use hcom send unless you are asked to use other hcom commands:
- hcom start/stop (join/leave chat)
- hcom watch --status (see all participants).
- hcom open (coordinate/orchestrate by launching other instances).
In this case, always run the 'hcom help' first to review correct usage."""

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

def get_display_name(session_id: str | None, prefix: str | None = None) -> str:
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

def resolve_instance_name(session_id: str, prefix: str | None = None) -> tuple[str, dict | None]:
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
    instance_name = get_display_name(session_id, prefix)
    return instance_name, None

def _remove_hcom_hooks_from_settings(settings):
    """Remove hcom hooks from settings dict"""
    if not isinstance(settings, dict) or 'hooks' not in settings:
        return
    
    if not isinstance(settings['hooks'], dict):
        return
    
    import copy

    # Build regex patterns dynamically from LEGACY_HOOK_COMMANDS
    # Current hooks (pattern 1): ${HCOM:-...} environment variable
    # Legacy hooks (patterns 2-7): Older formats that need cleanup
    # - HCOM_ACTIVE conditionals (removed for toggle implementation)
    # - Direct command invocation with specific hook args
    hook_args_pattern = '|'.join(LEGACY_HOOK_COMMANDS)
    hcom_patterns = [
        r'\$\{?HCOM',                                # Current: Environment variable ${HCOM:-...}
        r'\bHCOM_ACTIVE.*hcom\.py',                 # LEGACY: Unix HCOM_ACTIVE conditional
        r'IF\s+"%HCOM_ACTIVE%"',                    # LEGACY: Windows HCOM_ACTIVE conditional
        rf'\bhcom\s+({hook_args_pattern})\b',       # LEGACY: Direct hcom command
        rf'\buvx\s+hcom\s+({hook_args_pattern})\b', # LEGACY: uvx hcom command
        rf'hcom\.py["\']?\s+({hook_args_pattern})\b', # LEGACY: hcom.py with optional quote
        rf'["\'][^"\']*hcom\.py["\']?\s+({hook_args_pattern})\b(?=\s|$)',  # LEGACY: Quoted path
        r'sh\s+-c.*hcom',                           # LEGACY: Shell wrapper
    ]
    compiled_patterns = [re.compile(pattern) for pattern in hcom_patterns]

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

    # Remove HCOM from env section
    if 'env' in settings and isinstance(settings['env'], dict):
        settings['env'].pop('HCOM', None)
        # Clean up empty env dict
        if not settings['env']:
            del settings['env']


def build_env_string(env_vars, format_type="bash"):
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


def has_claude_arg(claude_args, arg_names, arg_prefixes):
    """Check if argument already exists in claude_args"""
    return claude_args and any(
        arg in arg_names or arg.startswith(arg_prefixes)
        for arg in claude_args
    )

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
    - Background scripts: persist until stop housekeeping (e.g., `hcom stop everything`) (24 hours)
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
def get_macos_terminal_argv():
    """Return macOS Terminal.app launch command as argv list."""
    return ['osascript', '-e', 'tell app "Terminal" to do script "bash {script}"', '-e', 'tell app "Terminal" to activate']

def get_windows_terminal_argv():
    """Return Windows terminal launcher as argv list."""
    if not (bash_exe := find_bash_on_windows()):
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

    # Define all hooks - must match ACTIVE_HOOK_TYPES
    # Format: (hook_type, matcher, command, timeout)
    hook_configs = [
        ('SessionStart', '', f'{hook_cmd_base} sessionstart', None),
        ('UserPromptSubmit', '', f'{hook_cmd_base} userpromptsubmit', None),
        ('PreToolUse', 'Bash', f'{hook_cmd_base} pre', None),
        ('Stop', '', f'{hook_cmd_base} poll', 86400),  # 24hr timeout max; internal timeout 30min default via config
        ('Notification', '', f'{hook_cmd_base} notify', None),
        ('SessionEnd', '', f'{hook_cmd_base} sessionend', None),
    ]

    # Validate hook_configs matches ACTIVE_HOOK_TYPES
    configured_types = [hook_type for hook_type, _, _, _ in hook_configs]
    if configured_types != ACTIVE_HOOK_TYPES:
        raise Exception(format_error(
            f"Hook configuration mismatch: {configured_types} != {ACTIVE_HOOK_TYPES}"
        ))

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

    python_path = sys.executable
    script_path = str(Path(__file__).resolve())
    settings['env']['HCOM'] = f'{python_path} {script_path}'

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
    """Verify that HCOM hooks were installed correctly with correct commands"""
    try:
        settings = read_file_with_retry(
            settings_path,
            lambda f: json.load(f),
            default=None
        )
        if not settings:
            return False

        # Check all hook types have correct commands
        hooks = settings.get('hooks', {})
        for hook_type, expected_cmd in zip(ACTIVE_HOOK_TYPES, HOOK_COMMANDS):
            hook_matchers = hooks.get(hook_type, [])
            if not hook_matchers:
                return False

            # Check if any matcher has the correct command
            found_correct_cmd = False
            for matcher in hook_matchers:
                for hook in matcher.get('hooks', []):
                    command = hook.get('command', '')
                    # Check for HCOM and the correct subcommand
                    if ('${HCOM}' in command or 'hcom' in command.lower()) and expected_cmd in command:
                        found_correct_cmd = True
                        break
                if found_correct_cmd:
                    break

            if not found_correct_cmd:
                return False

        # Check that HCOM env var is set
        env = settings.get('env', {})
        if 'HCOM' not in env:
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
    timeout = pos_data.get('wait_timeout', get_config_value('wait_timeout', 1800))
    if age > timeout:
        return "inactive", "", "timeout"

    # Detect stale 'waiting' status - check heartbeat, not status timestamp
    if last_status == 'waiting':
        last_stop = pos_data.get('last_stop', 0)
        heartbeat_age = now - last_stop if last_stop else 999999
        if heartbeat_age > 2:
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

def should_show_in_watch(d):
    """Show only enabled instances by default"""
    # Hide disabled instances
    if not d.get('enabled', False):
        return False

    # Hide truly ended sessions
    if d.get('session_ended'):
        return False

    # Show all other instances (including 'closed' during transition)
    return True

def show_instances_by_directory():
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

def get_status_summary():
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

        # Determine default enabled state: True for hcom-launched, False for vanilla
        is_hcom_launched = os.environ.get('HCOM_LAUNCHED') == '1'

        defaults = {
            "pos": 0,
            "enabled": is_hcom_launched,
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

def enable_instance(instance_name):
    """Enable instance - clears all stop flags and enables Stop hook polling"""
    update_instance_position(instance_name, {
        'enabled': True,
        'force_closed': False,
        'session_ended': False
    })
    set_status(instance_name, 'started')

def disable_instance(instance_name, force=False):
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

def merge_instance_data(to_data, from_data):
    """Merge instance data from from_data into to_data."""
    # Use current session_id from source (overwrites previous)
    to_data['session_id'] = from_data.get('session_id', to_data.get('session_id', ''))

    # Update transient fields from source
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

# ==================== CLI Parsing Functions ====================

def parse_count(value: str) -> int:
    """Parse and validate instance count"""
    try:
        number = int(value, 10)
    except ValueError as exc:
        raise argparse.ArgumentTypeError('Count must be an integer. Use -a/--agent for agent names.') from exc
    if number <= 0:
        raise argparse.ArgumentTypeError('Count must be positive.')
    if number > 100:
        raise argparse.ArgumentTypeError('Too many instances requested (max 100).')
    return number

def split_forwarded_args(argv: Sequence[str]) -> tuple[list[str], list[str]]:
    """Split arguments on -- separator for forwarding to claude"""
    if '--' not in argv:
        return list(argv), []
    idx = argv.index('--')
    return list(argv[:idx]), list(argv[idx + 1:])

def parse_open(namespace: argparse.Namespace, forwarded: list[str]) -> OpenCommand:
    """Parse and validate open command arguments"""
    prefix = namespace.prefix
    if prefix and '|' in prefix:
        raise CLIError('Prefix cannot contain "|" characters.')

    agents = namespace.agent or []
    count = namespace.count if namespace.count is not None else 1
    if not agents:
        agents = ['generic']

    return OpenCommand(
        count=count,
        agents=agents,
        prefix=prefix,
        background=namespace.background,
        claude_args=forwarded,
    )

def parse_watch(namespace: argparse.Namespace) -> WatchCommand:
    """Parse and validate watch command arguments"""
    wait_value = namespace.wait
    if wait_value is not None and wait_value < 0:
        raise CLIError('--wait expects a non-negative number of seconds.')

    if wait_value is not None:
        return WatchCommand(mode='wait', wait_seconds=wait_value or 60)
    if namespace.logs:
        return WatchCommand(mode='logs', wait_seconds=None)
    if namespace.status:
        return WatchCommand(mode='status', wait_seconds=None)
    return WatchCommand(mode='interactive', wait_seconds=None)

def parse_stop(namespace: argparse.Namespace) -> StopCommand:
    """Parse and validate stop command arguments"""
    target = namespace.target
    return StopCommand(
        target=target,
        close_all_hooks=namespace.all,
        force=namespace.force,
        _hcom_session=getattr(namespace, '_hcom_session', None),
    )

def parse_start(namespace: argparse.Namespace) -> StartCommand:
    """Parse and validate start command arguments"""
    return StartCommand(
        target=namespace.target,
        _hcom_session=getattr(namespace, '_hcom_session', None),
    )

def parse_send(namespace: argparse.Namespace) -> SendCommand:
    """Parse and validate send command arguments"""
    if namespace.resume and namespace.message:
        raise CLIError('Specify a resume alias or a message, not both.')
    session_id = getattr(namespace, '_hcom_session', None)
    if namespace.resume:
        return SendCommand(message=None, resume_alias=namespace.resume, _hcom_session=session_id)
    if namespace.message is None:
        raise CLIError('Message required (usage: hcom send "message").')
    return SendCommand(message=namespace.message, resume_alias=None, _hcom_session=session_id)

def build_parser() -> argparse.ArgumentParser:
    """Build argparse parser for hcom commands"""
    parser = argparse.ArgumentParser(prog='hcom', add_help=False)
    subparsers = parser.add_subparsers(dest='command', required=True)

    # Open command
    open_parser = subparsers.add_parser('open', add_help=False)
    open_parser.add_argument('count', nargs='?', type=parse_count, default=1)
    open_parser.add_argument('-a', '--agent', dest='agent', action='append')
    open_parser.add_argument('-t', '--prefix', dest='prefix')
    open_parser.add_argument('-p', '--background', action='store_true', dest='background')
    open_parser.add_argument('--help', action='store_true', dest='help_flag')
    open_parser.add_argument('-h', action='store_true', dest='help_flag_short')

    # Watch command
    watch_parser = subparsers.add_parser('watch', add_help=False)
    group = watch_parser.add_mutually_exclusive_group()
    group.add_argument('--logs', action='store_true')
    group.add_argument('--status', action='store_true')
    group.add_argument('--wait', nargs='?', const=60, type=int, metavar='SEC')
    watch_parser.add_argument('--help', action='store_true', dest='help_flag')
    watch_parser.add_argument('-h', action='store_true', dest='help_flag_short')

    # Stop command
    stop_parser = subparsers.add_parser('stop', add_help=False)
    stop_parser.add_argument('target', nargs='?')
    stop_parser.add_argument('--all', action='store_true')
    stop_parser.add_argument('--force', action='store_true')
    stop_parser.add_argument('--_hcom_session', help=argparse.SUPPRESS)
    stop_parser.add_argument('--help', action='store_true', dest='help_flag')
    stop_parser.add_argument('-h', action='store_true', dest='help_flag_short')

    # Start command
    start_parser = subparsers.add_parser('start', add_help=False)
    start_parser.add_argument('target', nargs='?')
    start_parser.add_argument('--_hcom_session', help=argparse.SUPPRESS)
    start_parser.add_argument('--help', action='store_true', dest='help_flag')
    start_parser.add_argument('-h', action='store_true', dest='help_flag_short')

    # Send command
    send_parser = subparsers.add_parser('send', add_help=False)
    send_parser.add_argument('message', nargs='?')
    send_parser.add_argument('--resume', metavar='ALIAS', help=argparse.SUPPRESS)
    send_parser.add_argument('--_hcom_session', help=argparse.SUPPRESS)
    send_parser.add_argument('--help', action='store_true', dest='help_flag')
    send_parser.add_argument('-h', action='store_true', dest='help_flag_short')

    return parser

def dispatch(namespace: argparse.Namespace, forwarded: list[str]):
    """Dispatch parsed arguments to appropriate command parser"""
    command = namespace.command
    if command == 'open':
        if getattr(namespace, 'help_flag', False) or getattr(namespace, 'help_flag_short', False):
            return cmd_help()
        return parse_open(namespace, forwarded)
    if command == 'watch':
        if getattr(namespace, 'help_flag', False) or getattr(namespace, 'help_flag_short', False):
            return cmd_help()
        return parse_watch(namespace)
    if command == 'stop':
        if getattr(namespace, 'help_flag', False) or getattr(namespace, 'help_flag_short', False):
            return cmd_help()
        return parse_stop(namespace)
    if command == 'start':
        if getattr(namespace, 'help_flag', False) or getattr(namespace, 'help_flag_short', False):
            return cmd_help()
        return parse_start(namespace)
    if command == 'send':
        if getattr(namespace, 'help_flag', False) or getattr(namespace, 'help_flag_short', False):
            return cmd_help()
        return parse_send(namespace)
    raise CLIError(f'Unsupported command: {command}')

def needs_help(args: Sequence[str]) -> bool:
    """Check if help was requested"""
    if not args:
        return True
    head = args[0]
    if head in {'help', '--help', '-h'}:
        return True
    return False

# ==================== Command Functions ====================

def cmd_help():
    """Show help text"""
    print(HELP_TEXT)

    # Additional help for AI assistants
    if os.environ.get('CLAUDECODE') == '1' or not sys.stdin.isatty():
        print("""

=== ADDITIONAL INFO ===

CONCEPT: HCOM launches Claude Code instances in new terminal windows.
They communicate with each other via a shared conversation.
You communicate with them via hcom commands.

KEY UNDERSTANDING:
• Single conversation - All instances share ~/.hcom/hcom.log
• Messaging - CLI and instances send with hcom send "message"
• Instances receive messages via hooks automatically
• hcom open is directory-specific - always cd to project directory first
• Named agents are custom system prompt files created by users/claude code beforehand.
• Named agents load from .claude/agents/<name>.md - if they have been created
• hcom watch --wait outputs last 5 seconds of messages, waits for the next message, prints it, and exits. 

LAUNCH PATTERNS:
  hcom open 2                            # 2 generic instances
  hcom open -a reviewer                  # 1 reviewer instance (agent file must already exist)
  hcom open 3 -a reviewer                # 3 reviewer instances
  hcom open -a reviewer -a tester        # 1 reviewer + 1 tester
  hcom open -t api 2                     # Team naming: api-hova7, api-kolec
  hcom open -- --model sonnet            # Pass `claude` CLI flags after --
  hcom open -p                           # Detached background (stop with: hcom stop <alias>)
  hcom open -- --resume <sessionid>      # Resume specific session
  HCOM_INITIAL_PROMPT="task" hcom open   # Set initial prompt for instance

@MENTION TARGETING:
  hcom send "message"           # Broadcasts to everyone
  hcom send "@api fix this"     # Targets all api-* instances (api-hova7, api-kolec)
  hcom send "@hova7 status?"    # Targets specific instance
  (Unmatched @mentions broadcast to everyone)

STATUS INDICATORS:
• ▶ active - processing/executing • ▷ delivered - instance just received a message
• ◉ idle - waiting for new messages • ■ blocked - permission request (needs user approval)
• ○ inactive - timed out, disconnected, etc • ○ unknown
              
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

def cmd_open(command: OpenCommand):
    """Launch Claude instances with chat enabled"""
    try:
        # Add -p flag and stream-json output for background mode if not already present
        claude_args = command.claude_args
        if command.background and '-p' not in claude_args and '--print' not in claude_args:
            claude_args = ['-p', '--output-format', 'stream-json', '--verbose'] + (claude_args or [])

        terminal_mode = get_config_value('terminal_mode', 'new_window')

        # Calculate total instances to launch
        total_instances = command.count * len(command.agents)

        # Fail fast for same_terminal with multiple instances
        if terminal_mode == 'same_terminal' and total_instances > 1:
            print(format_error(
                f"same_terminal mode cannot launch {total_instances} instances",
                "Use 'hcom open' for one generic instance or 'hcom open -a <agent>' for one agent"
            ), file=sys.stderr)
            return 1

        try:
            setup_hooks()
        except Exception as e:
            print(format_error(f"Failed to setup hooks: {e}"), file=sys.stderr)
            return 1

        log_file = hcom_path(LOG_FILE)
        instances_dir = hcom_path(INSTANCES_DIR)

        if not log_file.exists():
            log_file.touch()

        # Build environment variables for Claude instances
        base_env = build_claude_env()

        # Add prefix-specific hints if provided
        if command.prefix:
            base_env['HCOM_PREFIX'] = command.prefix
            send_cmd = build_send_command()
            hint = f"To respond to {command.prefix} group: {send_cmd} '@{command.prefix} message'"
            base_env['HCOM_INSTANCE_HINTS'] = hint
            first_use = f"You're in the {command.prefix} group. Use {command.prefix} to message: {send_cmd} '@{command.prefix} message'"
            base_env['HCOM_FIRST_USE_TEXT'] = first_use

        launched = 0
        initial_prompt = get_config_value('initial_prompt', 'Say hi in chat')

        # Launch count instances of each agent
        for agent in command.agents:
            for _ in range(command.count):
                instance_type = agent
                instance_env = base_env.copy()

                # Mark all hcom-launched instances
                instance_env['HCOM_LAUNCHED'] = '1'

                # Mark background instances via environment with log filename
                if command.background:
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
                    if command.background:
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

        # Auto-launch watch dashboard if configured and conditions are met
        terminal_mode = get_config_value('terminal_mode')
        auto_watch = get_config_value('auto_watch', True)

        # Only auto-watch if ALL instances launched successfully
        if terminal_mode == 'new_window' and auto_watch and failed == 0 and is_interactive():
            # Show tips first if needed
            if command.prefix:
                print(f"\n  • Send to {command.prefix} team: hcom send '@{command.prefix} message'")

            # Clear transition message
            print("\nOpening hcom watch...")
            time.sleep(2)  # Brief pause so user sees the message

            # Launch interactive watch dashboard in current terminal
            watch_cmd = WatchCommand(mode='interactive', wait_seconds=None)
            return cmd_watch(watch_cmd)
        else:
            tips = [
                "Run 'hcom watch' to view/send in conversation dashboard",
            ]
            if command.prefix:
                tips.append(f"Send to {command.prefix} team: hcom send '@{command.prefix} message'")

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

def cmd_watch(command: WatchCommand):
    """View conversation dashboard"""
    log_file = hcom_path(LOG_FILE)
    instances_dir = hcom_path(INSTANCES_DIR)

    if not log_file.exists() and not instances_dir.exists():
        print(format_error("No conversation log found", "Run 'hcom open' first"), file=sys.stderr)
        return 1

    # Determine mode
    show_logs = command.mode in ('logs', 'wait')
    show_status = command.mode == 'status'
    wait_timeout = command.wait_seconds

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
                    cmd_send_cli(message.strip())
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


def cmd_stop(command: StopCommand):
    """Stop instances, remove hooks, or archive - consolidated stop operations"""

    # Handle special targets
    if command.target == 'hooking':
        # hcom stop hooking [--all]
        if command.close_all_hooks:
            return cmd_cleanup('--all')
        else:
            return cmd_cleanup()

    elif command.target == 'everything':
        # hcom stop everything: stop all + archive + remove hooks
        print("Stopping HCOM for all instances, archiving conversation, and removing hooks...")

        # Stop all instances
        positions = load_all_positions()
        if positions:
            for instance_name in positions.keys():
                disable_instance(instance_name)
            print(f"Stopped HCOM for {len(positions)} instance(s)")

        # Archive conversation
        clear_result = cmd_clear()

        # Remove hooks from all directories
        cleanup_result = cmd_cleanup('--all')

        return max(clear_result, cleanup_result)

    elif command.target == 'all':
        # hcom stop all: stop all instances + archive
        positions = load_all_positions()

        if not positions:
            print("No instances found")
            # Still archive if there's conversation history
            return cmd_clear()

        stopped_count = 0
        bg_logs = []
        for instance_name, instance_data in positions.items():
            if instance_data.get('enabled', False):
                disable_instance(instance_name)
                print(f"Stopped HCOM for {instance_name}")
                stopped_count += 1

                # Track background logs
                if instance_data.get('background'):
                    log_file = instance_data.get('background_log_file', '')
                    if log_file:
                        bg_logs.append((instance_name, log_file))

        if stopped_count == 0:
            print("All instances already stopped")
        else:
            print(f"Stopped {stopped_count} instance(s)")

            # Show background logs if any
            if bg_logs:
                print("\nBackground logs:")
                for name, log_file in bg_logs:
                    print(f"  {name}: {log_file}")
                print("\nMonitor: tail -f <log_file>")
                print("Force stop: hcom stop --force all")

        # Archive conversation
        return cmd_clear()

    else:
        # hcom stop [alias] or hcom stop (self)

        # Always verify hooks when running from Claude Code (catches broken hooks regardless of path)
        if not check_and_update_hooks():
            return 1

        # Stop specific instance or self
        # Get instance name from injected session or target
        if command._hcom_session and not command.target:
            instance_name, _ = resolve_instance_name(command._hcom_session, os.environ.get('HCOM_PREFIX'))
        else:
            instance_name = command.target

        position = load_instance_position(instance_name) if instance_name else None

        if not instance_name:
            print("Error: Could not determine instance. Run inside Claude Code or specify alias. Or run hcom stop <alias> or hcom stop all")
            return 1

        if not position:
            print(f"No instance found for {instance_name}")
            return 1

        # Skip already stopped instances (unless forcing)
        if not position.get('enabled', False) and not command.force:
            print(f"HCOM already stopped for {instance_name}")
            return 0

        # Disable instance (optionally with force)
        disable_instance(instance_name, force=command.force)

        if command.force:
            print(f"⚠️  Force stopped HCOM for {instance_name}.")
            print(f"    Bash tool use is now DENIED. Instance is locked down.")
            print(f"    To restart: hcom start {instance_name}")
        else:
            print(f"Stopped HCOM for {instance_name}. Will no longer receive chat messages automatically.")

        # Show background log location if applicable
        if position.get('background'):
            log_file = position.get('background_log_file', '')
            if log_file:
                print(f"\nBackground log: {log_file}")
                print(f"Monitor: tail -f {log_file}")
                if not command.force:
                    print(f"Force stop: hcom stop --force {instance_name}")

        return 0

def cmd_start(command: StartCommand):
    """Enable HCOM participation for instances"""

    # Always verify hooks when running from Claude Code (catches broken hooks regardless of path)
    if not check_and_update_hooks():
        return 1

    # Get instance name from injected session or target
    if command._hcom_session and not command.target:
        instance_name, existing_data = resolve_instance_name(command._hcom_session, os.environ.get('HCOM_PREFIX'))

        # Create instance if it doesn't exist (opt-in for vanilla instances)
        if not existing_data:
            initialize_instance_in_position_file(instance_name, command._hcom_session)
            # Enable instance (clears all stop flags)
            enable_instance(instance_name)
            print(f"Started HCOM for this instance. Your alias is: {instance_name}")
        else:
            # Skip already started instances
            if existing_data.get('enabled', False):
                print(f"HCOM already started for {instance_name}")
                return 0

            # Re-enabling existing instance
            enable_instance(instance_name)
            print(f"Started HCOM for {instance_name}. Rejoined chat.")

        return 0

    # Handle hooking target
    if command.target == 'hooking':
        # hcom start hooking: install hooks in current directory
        if setup_hooks():
            print("HCOM hooks installed in current directory")
            print("Hooks active on next Claude Code launch in this directory")
            return 0
        else:
            return 1

    # CLI path: start specific instance
    positions = load_all_positions()

    # Handle missing target from external CLI
    if not command.target:
        print("Error: No instance specified & Not run by Claude Code\n")
        print("Run by Claude Code: 'hcom start' starts HCOM for the current instance")
        print("From anywhere: 'hcom start <alias>'\n")
        print("To launch new instances: 'hcom open'")
        return 1

    # Start specific instance
    instance_name = command.target
    position = positions.get(instance_name)

    if not position:
        print(f"Instance not found: {instance_name}")
        return 1

    # Skip already started instances
    if position.get('enabled', False):
        print(f"HCOM already started for {instance_name}")
        return 0

    # Enable instance (clears all stop flags)
    enable_instance(instance_name)

    print(f"Started HCOM for {instance_name}. Rejoined chat.")
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

def check_and_update_hooks() -> bool:
    """Verify hooks are correct when running inside Claude Code, reinstall if needed.
    Returns True if hooks are good (continue), False if hooks were updated (restart needed)."""
    if os.environ.get('CLAUDECODE') != '1':
        return True  # Not in Claude Code, continue normally

    # Check both cwd and home for hooks
    cwd_settings = Path.cwd() / '.claude' / 'settings.local.json'
    home_settings = Path.home() / '.claude' / 'settings.local.json'

    # If hooks are correctly installed, continue
    if verify_hooks_installed(cwd_settings) or verify_hooks_installed(home_settings):
        return True

    # Hooks missing or incorrect - reinstall them
    try:
        setup_hooks()
        print("Hooks updated. Restart Claude Code to use HCOM.", file=sys.stderr)
    except Exception as e:
        print(f"Failed to update hooks: {e}", file=sys.stderr)
        print("Try running: hcom open from normal terminal", file=sys.stderr)
    return False

def cmd_send(command: SendCommand, force_cli=False):
    """Send message to hcom, force cli for config sender instead of instance generated name"""

    # Always verify hooks when running from Claude Code (catches broken hooks regardless of path)
    if not check_and_update_hooks():
        return 1

    # Handle resume command - pass caller session_id
    if command.resume_alias:
        caller_session = command._hcom_session  # May be None for CLI
        return cmd_resume_merge(command.resume_alias, caller_session)

    message = command.message

    # Check message is provided
    if not message:
        print(format_error("No message provided"), file=sys.stderr)
        return 1

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
            sender_name = get_config_value('sender_name', 'bigboss')
            all_names = all_instances + [sender_name]
            unmatched = [m for m in mentions
                        if not any(name.lower().startswith(m.lower()) for name in all_names)]
            if unmatched:
                print(f"Note: @{', @'.join(unmatched)} don't match any instances - broadcasting to all", file=sys.stderr)
        except Exception:
            pass  # Don't fail on warning

    # Determine sender from injected session_id or CLI
    if command._hcom_session and not force_cli:
        # Instance context - get name from session_id
        try:
            sender_name = get_display_name(command._hcom_session)
        except (ValueError, Exception) as e:
            print(format_error(f"Invalid session_id: {e}"), file=sys.stderr)
            return 1

        instance_data = load_instance_position(sender_name)

        # Initialize instance if doesn't exist (first use)
        if not instance_data:
            initialize_instance_in_position_file(sender_name, command._hcom_session)
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
            max_msgs = get_config_value('max_messages_per_delivery', 50)
            formatted = format_hook_messages(messages[:max_msgs], sender_name)
            print(f"Message sent\n\n{formatted}", file=sys.stderr)
        else:
            print("Message sent", file=sys.stderr)

        # Show cli_hints if configured (non-interactive mode)
        if not is_interactive():
            show_cli_hints()

        return 0
    else:
        # CLI context - no session_id or force_cli=True
        sender_name = get_config_value('sender_name', 'bigboss')

        if not send_message(sender_name, message):
            print(format_error("Failed to send message"), file=sys.stderr)
            return 1

        print(f"✓ Sent from {sender_name}", file=sys.stderr)

        # Show cli_hints if configured (non-interactive mode)
        if not is_interactive():
            show_cli_hints()

        return 0

def cmd_send_cli(message):
    """Force CLI sender (skip outbox, use config sender name)"""
    command = SendCommand(message=message, resume_alias=None)
    return cmd_send(command, force_cli=True)

def cmd_resume_merge(alias: str, caller_session: str | None = None) -> int:
    """Resume/merge current instance into an existing instance by alias.
    INTERNAL COMMAND: Only called via 'hcom send --resume alias' during implicit resume workflow.
    Not meant for direct CLI usage.
    Args:
        alias: Target instance alias to merge into
        caller_session: Session ID of caller (injected by PreToolUse hook) or None for CLI
    """
    # If caller_session provided (from hcom send --resume), use it
    if caller_session:
        instance_name = get_display_name(caller_session)
    else:
        # CLI path - no session context
        print(format_error("Not in HCOM instance context"), file=sys.stderr)
        return 1

    if not instance_name:
        print(format_error("Could not determine instance name"), file=sys.stderr)
        return 1

    # Sanitize alias: must be valid instance name format
    # Base: 5 lowercase alphanumeric (e.g., hova3)
    # Prefixed: {prefix}-{5 chars} (e.g., api-hova3, cool-team-kolec)
    # This prevents path traversal attacks (e.g., ../../etc, /etc, etc.)
    if not re.match(r'^([a-zA-Z0-9_-]+-)?[a-z0-9]{5}$', alias):
        print(format_error("Invalid alias format. Must be 5-char instance name or prefix-name format"), file=sys.stderr)
        return 1

    # Attempt to merge current instance into target alias
    status = merge_instance_immediately(instance_name, alias)

    # Handle results
    if not status:
        # Empty status means names matched (from_name == to_name)
        status = f"[SUCCESS] ✓ Already using HCOM alias {alias}. Rejoined chat."
    elif status.startswith('[SUCCESS]'):
        # Merge successful - update message
        status = f"[SUCCESS] ✓ Resumed HCOM as {alias}. Rejoined chat."

    # If merge successful, enable instance (clears session_ended and stop flags)
    if status.startswith('[SUCCESS]'):
        enable_instance(alias)

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

def detect_session_scenario(
    hook_type: str | None,
    session_id: str,
    source: str,
    existing_data: dict | None
) -> SessionScenario | None:
    """
    Detect session startup scenario explicitly.

    Returns:
        SessionScenario for definitive scenarios
        None for deferred decision (SessionStart with wrong session_id)
    """
    if existing_data is not None:
        # Found existing instance with matching session_id
        return SessionScenario.MATCHED_RESUME

    if hook_type == 'sessionstart' and source == 'resume':
        # SessionStart on resume without match = wrong session_id
        # Don't know if truly unmatched yet - UserPromptSubmit will decide
        return None  # Deferred decision

    if hook_type == 'userpromptsubmit' and source == 'resume':
        # UserPromptSubmit on resume without match = definitively unmatched
        return SessionScenario.UNMATCHED_RESUME

    # Normal startup
    return SessionScenario.FRESH_START


def should_create_instance_file(scenario: SessionScenario | None, hook_type: str | None) -> bool:
    """
    Decide whether to create instance file NOW.

    Simplified: Only UserPromptSubmit creates instances.
    SessionStart just shows minimal message and tracks status.
    """
    # Only UserPromptSubmit creates instances
    if hook_type != 'userpromptsubmit':
        return False

    # Create for new scenarios only (not matched resume which already exists)
    return scenario in (SessionScenario.FRESH_START, SessionScenario.UNMATCHED_RESUME)


def init_hook_context(hook_data, hook_type=None):
    """
    Initialize instance context with explicit scenario detection.

    Flow:
    1. Resolve instance name (search by session_id, generate if not found)
    2. Detect scenario (fresh/matched/unmatched/deferred)
    3. Decide whether to create file NOW
    4. Return context with all decisions made
    """
    session_id = hook_data.get('session_id', '')
    transcript_path = hook_data.get('transcript_path', '')
    source = hook_data.get('source', 'startup')
    prefix = os.environ.get('HCOM_PREFIX')

    # Step 1: Resolve instance name
    instance_name, existing_data = resolve_instance_name(session_id, prefix)

    # Save migrated data if we have it
    if existing_data:
        save_instance_position(instance_name, existing_data)

    # Step 2: Detect scenario
    scenario = detect_session_scenario(hook_type, session_id, source, existing_data)

    # Check if instance is brand new (before creation - for bypass logic)
    instance_file = hcom_path(INSTANCES_DIR, f'{instance_name}.json')
    is_new_instance = not instance_file.exists()


    # Step 3: Decide creation
    should_create = should_create_instance_file(scenario, hook_type)


    if should_create:
        initialize_instance_in_position_file(instance_name, session_id)

    # Step 4: Build updates dict
    updates: dict[str, Any] = {
        'directory': str(Path.cwd()),
    }

    if session_id:
        updates['session_id'] = session_id

    if transcript_path:
        updates['transcript_path'] = transcript_path

    bg_env = os.environ.get('HCOM_BACKGROUND')
    if bg_env:
        updates['background'] = True
        updates['background_log_file'] = str(hcom_path(LOGS_DIR, bg_env))

    # Return compatible with existing callers
    is_resume_match = (scenario == SessionScenario.MATCHED_RESUME)


    return instance_name, updates, is_resume_match, is_new_instance

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
    sys.exit(EXIT_SUCCESS)

def handle_pretooluse(hook_data, instance_name, updates):
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

    # Inject session_id into hcom send/stop/start commands via updatedInput
    if tool_name == 'Bash' and session_id:
        command = hook_data.get('tool_input', {}).get('command', '')

        # Match: (hcom|uvx hcom|python X hcom.py|X hcom.py) (send|stop|start)
        # Handles: hcom, uvx hcom, python hcom.py, python /path/to/hcom.py, hcom.py, /path/to/hcom.py
        hcom_pattern = r'((?:uvx\s+)?hcom|(?:python3?\s+)?\S*hcom\.py)\s+(send|stop|start)\b'

        # Check if command contains any hcom invocations
        if re.search(hcom_pattern, command):
            # Inject session_id after EACH hcom command (handles chained commands)
            modified_command = re.sub(hcom_pattern, rf'\g<0> --_hcom_session {session_id}', command)

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
            sys.exit(EXIT_SUCCESS)



def handle_stop(hook_data, instance_name, updates):
    """Handle Stop hook - poll for messages and deliver"""

    try:
        entry_time = time.time()
        updates['last_stop'] = entry_time
        timeout = get_config_value('wait_timeout', 1800)
        updates['wait_timeout'] = timeout
        set_status(instance_name, 'waiting')

        try:
            update_instance_position(instance_name, updates)
        except Exception as e:
            log_hook_error(f'stop:update_instance_position({instance_name})', e)

        start_time = time.time()

        try:
            loop_count = 0
            last_heartbeat = start_time
            # Actual polling loop - this IS the holding pattern
            while time.time() - start_time < timeout:
                if loop_count == 0:
                    time.sleep(0.1)  # Initial wait before first poll
                loop_count += 1

                # Load instance data once per poll
                instance_data = load_instance_position(instance_name)

                # Check if session ended (SessionEnd hook fired) - exit without changing status
                if instance_data.get('session_ended'):
                    sys.exit(EXIT_SUCCESS)  # Don't overwrite session_ended status

                # Check if user input is pending - exit cleanly if recent input
                last_user_input = instance_data.get('last_user_input', 0)
                if time.time() - last_user_input < 0.2:
                    sys.exit(EXIT_SUCCESS)  # Don't overwrite status - let current status remain

                # Check if closed - exit cleanly
                if not instance_data.get('enabled', False):
                    sys.exit(EXIT_SUCCESS)  # Preserve 'stopped' status set by cmd_stop

                # Check for new messages and deliver
                if messages := get_unread_messages(instance_name, update_position=True):
                    messages_to_show = messages[:get_config_value('max_messages_per_delivery', 50)]
                    reason = format_hook_messages(messages_to_show, instance_name)
                    set_status(instance_name, 'message_delivered', messages_to_show[0]['from'])

                    output = {"decision": "block", "reason": reason}
                    print(json.dumps(output, ensure_ascii=False), file=sys.stderr)
                    sys.exit(EXIT_BLOCK)

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

    except Exception as e:
        # Log error and exit gracefully
        log_hook_error('handle_stop', e)
        sys.exit(EXIT_SUCCESS)  # Preserve previous status on exception

def handle_notify(hook_data, instance_name, updates):
    """Handle Notification hook - track permission requests"""
    updates['notification_message'] = hook_data.get('message', '')
    update_instance_position(instance_name, updates)
    set_status(instance_name, 'blocked', hook_data.get('message', ''))

def wait_for_stop_exit(instance_name, max_wait=0.2):
    """Wait for Stop hook to exit. Returns wait time in ms."""
    start = time.time()

    while time.time() - start < max_wait:
        time.sleep(0.01)

        data = load_instance_position(instance_name)
        last_stop_age = time.time() - data.get('last_stop', 0)

        if last_stop_age > 0.2:
            return int((time.time() - start) * 1000)

    return int((time.time() - start) * 1000)

def handle_userpromptsubmit(hook_data, instance_name, updates, is_resume_match, is_new_instance):
    """Handle UserPromptSubmit hook - track when user sends messages"""
    # Load instance data for coordination check and alias_announced
    instance_data = load_instance_position(instance_name)
    is_enabled = instance_data.get('enabled', False)
    last_stop = instance_data.get('last_stop', 0)
    alias_announced = instance_data.get('alias_announced', False)

    # Coordinate with Stop hook only if enabled AND Stop hook is active
    stop_is_active = (time.time() - last_stop) < 1.0

    if is_enabled and stop_is_active:
        updates['last_user_input'] = time.time()
        update_instance_position(instance_name, updates)
        wait_ms = wait_for_stop_exit(instance_name)

    send_cmd = build_send_command('your message', instance_name)
    resume_cmd = send_cmd.replace("'your message'", "--resume your_old_alias")

    # Build message based on what happened
    msg = None

    # Determine if this is an HCOM-launched instance
    is_hcom_launched = os.environ.get('HCOM_LAUNCHED') == '1'

    # PRIORITY 1: Handle unmatched resume FIRST (is_new_instance=True means SessionStart was skipped)
    # This must show critical recovery message regardless of HCOM_LAUNCHED status
    if is_new_instance:
        msg = build_hcom_bootstrap_text(instance_name)
        msg += (
            f"\n\n[CRITICAL: HCOM RESUME DETECTED - You MUST recover your identity to maintain conversation context. "
            f"Run this command: {resume_cmd} - This is REQUIRED for message history and position tracking]"
        )
        update_instance_position(instance_name, {'alias_announced': True})

    # PRIORITY 2: Normal startup - show bootstrap if not already announced
    elif not alias_announced:
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

    # PRIORITY 3: Add resume status note if we showed bootstrap for a matched resume
    if msg and is_resume_match and not is_new_instance:
        if is_enabled:
            msg += "\n[Session resumed. HCOM started for this instance - will receive chat messages. Your alias and conversation history preserved.]"
        else:
            msg += "\n[Session resumed. HCOM stopped for this instance - will not receive chat messages. Run 'hcom start' to rejoin chat. Your alias and conversation history preserved.]"

    if msg:
        output = {
            # "systemMessage": "HCOM enabled",
            "hookSpecificOutput": {
                "hookEventName": "UserPromptSubmit",
                "additionalContext": msg
            }
        }
        print(json.dumps(output), file=sys.stdout)
        # sys.exit(1)

def handle_sessionstart(hook_data, instance_name, updates, is_resume_match):
    """Handle SessionStart hook - minimal message, full details on first prompt"""
    source = hook_data.get('source', 'startup')

    # Update instance if it exists (matched resume only, since we don't create in SessionStart anymore)
    instance_file = hcom_path(INSTANCES_DIR, f'{instance_name}.json')
    if instance_file.exists():
        updates['alias_announced'] = False
        update_instance_position(instance_name, updates)
        set_status(instance_name, 'session_start')

    # Minimal message - no alias yet (UserPromptSubmit will show full details)
    help_text = "[HCOM active. Submit a prompt to initialize.]"

    # Add first_use_text only for hcom-launched instances on startup
    if os.environ.get('HCOM_LAUNCHED') == '1' and source == 'startup':
        first_use_text = get_config_value('first_use_text', '')
        if first_use_text:
            help_text += f" [{first_use_text}]"

    output = {
        "hookSpecificOutput": {
            "hookEventName": "SessionStart",
            "additionalContext": help_text
        }
    }

    print(json.dumps(output))

def handle_sessionend(hook_data, instance_name, updates):
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

def handle_hook(hook_type: str) -> None:
    """Unified hook handler for all HCOM hooks"""
    hook_data = json.load(sys.stdin)

    if not ensure_hcom_directories():
        log_hook_error('handle_hook', Exception('Failed to create directories'))
        sys.exit(EXIT_SUCCESS)

    session_id_short = hook_data.get('session_id', 'none')[:8] if hook_data.get('session_id') else 'none'
    source_debug = hook_data.get('source', 'NO_SOURCE')

    # Vanilla instance check (not hcom-launched)
    # Exit early if no instance file exists, except:
    # - PreToolUse (handles first send opt-in)
    # - UserPromptSubmit with hcom command (shows preemptive bootstrap)
    if hook_type != 'pre' and os.environ.get('HCOM_LAUNCHED') != '1':
        session_id = hook_data.get('session_id', '')
        if not session_id:
            sys.exit(EXIT_SUCCESS)

        instance_name = get_display_name(session_id, os.environ.get('HCOM_PREFIX'))
        instance_file = hcom_path(INSTANCES_DIR, f'{instance_name}.json')

        if not instance_file.exists():
            # Allow UserPromptSubmit through if prompt contains hcom command
            if hook_type == 'userpromptsubmit':
                user_prompt = hook_data.get('prompt', '')
                if not re.search(r'\bhcom\s+\w+', user_prompt, re.IGNORECASE):
                    sys.exit(EXIT_SUCCESS)
                # Continue - let handle_userpromptsubmit show bootstrap
            else:
                sys.exit(EXIT_SUCCESS)

    # Initialize instance context (creates file if needed, reuses existing if session_id matches)
    instance_name, updates, is_resume_match, is_new_instance = init_hook_context(hook_data, hook_type)

    # Special bypass for unmatched resume - must show critical warning even if disabled
    # is_new_instance=True in UserPromptSubmit means SessionStart didn't create it (was skipped)
    if hook_type == 'userpromptsubmit' and is_new_instance:
        handle_userpromptsubmit(hook_data, instance_name, updates, is_resume_match, is_new_instance)
        sys.exit(EXIT_SUCCESS)

    # Check enabled status (PreToolUse handles toggle, so exempt)
    if hook_type != 'pre':
        instance_data = load_instance_position(instance_name)
        if not instance_data.get('enabled', False):
            sys.exit(EXIT_SUCCESS)

    match hook_type:
        case 'pre':
            handle_pretooluse(hook_data, instance_name, updates)
        case 'poll':
            handle_stop(hook_data, instance_name, updates)
        case 'notify':
            handle_notify(hook_data, instance_name, updates)
        case 'userpromptsubmit':
            handle_userpromptsubmit(hook_data, instance_name, updates, is_resume_match, is_new_instance)
        case 'sessionstart':
            handle_sessionstart(hook_data, instance_name, updates, is_resume_match)
        case 'sessionend':
            handle_sessionend(hook_data, instance_name, updates)

    sys.exit(EXIT_SUCCESS)


# ==================== Main Entry Point ====================

def main(argv=None):
    """Main command dispatcher"""
    if argv is None:
        argv = sys.argv[1:]
    else:
        argv = argv[1:] if len(argv) > 0 and argv[0].endswith('hcom.py') else argv

    # Check for help
    if needs_help(argv):
        return cmd_help()

    # Handle hook commands (special case - no parsing needed)
    if argv and argv[0] in ('poll', 'notify', 'pre', 'sessionstart', 'userpromptsubmit', 'sessionend'):
        handle_hook(argv[0])
        return 0

    # Handle send_cli (hidden command)
    if argv and argv[0] == 'send_cli':
        if len(argv) < 2:
            print(format_error("Message required"), file=sys.stderr)
            return 1
        return cmd_send_cli(argv[1])

    # Split on -- separator for forwarding args
    hcom_args, forwarded = split_forwarded_args(argv)

    # Ensure directories exist for commands that need them
    if hcom_args and hcom_args[0] not in ('help', '--help', '-h'):
        if not ensure_hcom_directories():
            print(format_error("Failed to create HCOM directories"), file=sys.stderr)
            return 1

    # Build parser and parse arguments
    parser = build_parser()

    try:
        namespace = parser.parse_args(hcom_args)
    except SystemExit as exc:
        if exc.code != 0:
            print("Run 'hcom -h' for help", file=sys.stderr)
        return exc.code if isinstance(exc.code, int) else 1

    # Dispatch to command parsers and get typed command objects
    try:
        command_obj = dispatch(namespace, forwarded)

        # command_obj could be exit code (from help) or command dataclass
        if isinstance(command_obj, int):
            return command_obj

        # Execute the command with typed object
        if isinstance(command_obj, OpenCommand):
            return cmd_open(command_obj)
        elif isinstance(command_obj, WatchCommand):
            return cmd_watch(command_obj)
        elif isinstance(command_obj, StopCommand):
            return cmd_stop(command_obj)
        elif isinstance(command_obj, StartCommand):
            return cmd_start(command_obj)
        elif isinstance(command_obj, SendCommand):
            return cmd_send(command_obj)
        else:
            print(format_error(f"Unknown command type: {type(command_obj)}"), file=sys.stderr)
            return 1

    except CLIError as exc:
        print(str(exc), file=sys.stderr)
        return 1

if __name__ == '__main__':
    sys.exit(main())
