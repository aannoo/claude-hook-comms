#!/usr/bin/env python3
"""
hcom 0.2.1
CLI tool for launching multiple Claude Code terminals with interactive subagents, headless persistence, and real-time communication via hooks
"""

import os
import sys
import json
import tempfile
import shutil
import shlex
import re
import subprocess
import time
import select
import threading
import platform
import random
from pathlib import Path
from datetime import datetime, timedelta

# ==================== Constants ====================

IS_WINDOWS = sys.platform == 'win32'

def is_wsl():
    """Detect if running in WSL (Windows Subsystem for Linux)"""
    if platform.system() != 'Linux':
        return False
    try:
        with open('/proc/version', 'r') as f:
            return 'microsoft' in f.read().lower()
    except:
        return False

HCOM_ACTIVE_ENV = 'HCOM_ACTIVE'
HCOM_ACTIVE_VALUE = '1'

EXIT_SUCCESS = 0
EXIT_ERROR = 1
EXIT_BLOCK = 2

HOOK_DECISION_BLOCK = 'block'

ERROR_ACCESS_DENIED = 5     # Windows - Process exists but no permission
ERROR_INVALID_PARAMETER = 87  # Windows - Invalid PID or parameters
ERROR_ALREADY_EXISTS = 183   # Windows - For file/mutex creation, not process checks

# Windows API constants
DETACHED_PROCESS = 0x00000008  # CreateProcess flag for no console window
CREATE_NO_WINDOW = 0x08000000  # Prevent console window creation
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000  # Vista+ minimal access rights
PROCESS_QUERY_INFORMATION = 0x0400  # Pre-Vista process access rights TODO: is this a joke? why am i supporting pre vista? who the fuck is running claude code on vista let alone pre?!

# Timing constants
FILE_RETRY_DELAY = 0.01  # 10ms delay for file lock retries
STOP_HOOK_POLL_INTERVAL = 0.05     # 50ms between stop hook polls
KILL_CHECK_INTERVAL = 0.1  # 100ms between process termination checks

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
        kernel32 = ctypes.windll.kernel32

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

STATUS_MAP = {
    "thinking": (BG_CYAN, "◉"),
    "responding": (BG_GREEN, "▷"),
    "executing": (BG_GREEN, "▶"),
    "waiting": (BG_BLUE, "◉"),
    "blocked": (BG_YELLOW, "■"),
    "inactive": (BG_RED, "○")
}

# ==================== Windows/WSL Console Unicode ====================
import io

# Apply UTF-8 encoding for Windows and WSL
if IS_WINDOWS or is_wsl():
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    except:
        pass # Fallback if stream redirection fails

# ==================== Error Handling Strategy ====================
# Hooks: Must never raise exceptions (breaks hcom). Functions return True/False.
# CLI: Can raise exceptions for user feedback. Check return values.
# Critical I/O: atomic_write, save_instance_position, merge_instance_immediately
# Pattern: Try/except/return False in hooks, raise in CLI operations.

# ==================== Configuration ====================

DEFAULT_CONFIG = {
    "terminal_command": None,
    "terminal_mode": "new_window",
    "initial_prompt": "Say hi in chat",
    "sender_name": "bigboss",
    "sender_emoji": "🐳",
    "cli_hints": "",
    "wait_timeout": 1800, # 30mins
    "max_message_size": 1048576,  # 1MB
    "max_messages_per_delivery": 50,
    "first_use_text": "Essential, concise messages only, say hi in hcom chat now",
    "instance_hints": "",
    "env_overrides": {},
    "auto_watch": True  # Auto-launch watch dashboard after open
}

_config = None

HOOK_SETTINGS = {
    'wait_timeout': 'HCOM_WAIT_TIMEOUT',
    'max_message_size': 'HCOM_MAX_MESSAGE_SIZE',
    'max_messages_per_delivery': 'HCOM_MAX_MESSAGES_PER_DELIVERY',
    'first_use_text': 'HCOM_FIRST_USE_TEXT',
    'instance_hints': 'HCOM_INSTANCE_HINTS',
    'sender_name': 'HCOM_SENDER_NAME',
    'sender_emoji': 'HCOM_SENDER_EMOJI',
    'cli_hints': 'HCOM_CLI_HINTS',
    'terminal_mode': 'HCOM_TERMINAL_MODE',
    'terminal_command': 'HCOM_TERMINAL_COMMAND',
    'initial_prompt': 'HCOM_INITIAL_PROMPT',
    'auto_watch': 'HCOM_AUTO_WATCH'
}

# Path constants
LOG_FILE = "hcom.log"
INSTANCES_DIR = "instances"
LOGS_DIR = "logs"
CONFIG_FILE = "config.json"
ARCHIVE_DIR = "archive"

# ==================== File System Utilities ====================

def hcom_path(*parts, ensure_parent=False):
    """Build path under ~/.hcom"""
    path = Path.home() / ".hcom"
    if parts:
        path = path.joinpath(*parts)
    if ensure_parent:
        path.parent.mkdir(parents=True, exist_ok=True)
    return path

def atomic_write(filepath, content):
    """Write content to file atomically to prevent corruption (now with NEW and IMPROVED (wow!) Windows retry logic (cool!)). Returns True on success, False on failure."""
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
                    os.unlink(tmp.name)
                except:
                    pass
                return False
        except Exception:
            try: # Clean up temp file on any other error
                os.unlink(tmp.name)
            except:
                pass
            return False

def read_file_with_retry(filepath, read_func, default=None, max_retries=3):
    """Read file with retry logic for Windows file locking"""
    if not Path(filepath).exists():
        return default

    for attempt in range(max_retries):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return read_func(f)
        except PermissionError as e:
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

def get_instance_file(instance_name):
    """Get path to instance's position file"""
    return hcom_path(INSTANCES_DIR, f"{instance_name}.json")

def migrate_instance_data_v020(data, instance_name):
    """One-time migration from v0.2.0 format (remove in v0.3.0)"""
    needs_save = False

    # Convert single session_id to session_ids array
    if 'session_ids' not in data and 'session_id' in data and data['session_id']:
        data['session_ids'] = [data['session_id']]
        needs_save = True

    # Remove conversation_uuid - no longer used anywhere
    if 'conversation_uuid' in data:
        del data['conversation_uuid']
        needs_save = True

    if needs_save:
        save_instance_position(instance_name, data)

    return data

def load_instance_position(instance_name):
    """Load position data for a single instance"""
    instance_file = get_instance_file(instance_name)

    data = read_file_with_retry(
        instance_file,
        lambda f: json.load(f),
        default={}
    )

    # Apply migration if needed
    if data:
        data = migrate_instance_data_v020(data, instance_name)

    return data

def save_instance_position(instance_name, data):
    """Save position data for a single instance. Returns True on success, False on failure."""
    try:
        instance_file = hcom_path(INSTANCES_DIR, f"{instance_name}.json", ensure_parent=True)
        return atomic_write(instance_file, json.dumps(data, indent=2))
    except:
        return False

def load_all_positions():
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

def clear_all_positions():
    """Clear all instance position files"""
    instances_dir = hcom_path(INSTANCES_DIR)
    if instances_dir.exists():
        for f in instances_dir.glob('*.json'):
            f.unlink()
    else:
        instances_dir.mkdir(exist_ok=True)

# ==================== Hook Utilities ====================

HCOM_FORMAT_INSTRUCTIONS = """\n\nREQUIRED: Display HCOM 'reason' content using this format with markdown:
═════════ HCOM ═════════
***[✓ Sent] [new message] and/or [all other info in square brackets]***
**name → name:** *message content*
════════════════════════
Always show exact full content of every single message without saying why you are displaying it. This format is mandatory for system functionality"""

def emit_hook_response(reason, decision=HOOK_DECISION_BLOCK, exit_code=EXIT_BLOCK):
    """Emit standardized hook JSON response"""
    output = {"decision": decision, "reason": reason} if decision else {"reason": reason}
    print(json.dumps(output, ensure_ascii=False), file=sys.stderr)
    sys.exit(exit_code)


# ==================== Configuration System ====================

def get_cached_config():
    """Get cached configuration, loading if needed"""
    global _config
    if _config is None:
        _config = _load_config_from_file()
    return _config

def _load_config_from_file():
    """Actually load configuration from ~/.hcom/config.json"""
    import copy
    config_path = hcom_path(CONFIG_FILE, ensure_parent=True)

    config = copy.deepcopy(DEFAULT_CONFIG)

    try:
        user_config = read_file_with_retry(
            config_path,
            lambda f: json.load(f),
            default=None
        )
        if user_config:
            for key, value in user_config.items():
                if key == 'env_overrides':
                    config['env_overrides'].update(value)
                else:
                    config[key] = value
        elif not config_path.exists():
            atomic_write(config_path, json.dumps(DEFAULT_CONFIG, indent=2))
    except (json.JSONDecodeError, UnicodeDecodeError, PermissionError):
        print("Warning: Cannot read config file, using defaults", file=sys.stderr)

    return config

def get_config_value(key, default=None):
    """Get config value with proper precedence:
    1. Environment variable (if in HOOK_SETTINGS)
    2. Config file
    3. Default value
    """
    if key in HOOK_SETTINGS:
        env_var = HOOK_SETTINGS[key]
        env_value = os.environ.get(env_var)
        if env_value is not None:
            if key in ['wait_timeout', 'max_message_size', 'max_messages_per_delivery']:
                try:
                    return int(env_value)
                except ValueError:
                    pass
            elif key == 'auto_watch': # Convert string to boolean
                return env_value.lower() in ('true', '1', 'yes', 'on')
            else:
                return env_value

    config = get_cached_config()
    return config.get(key, default)

def get_hook_command():
    """Get hook command with silent fallback
    
    Uses ${HCOM:-true} for clean paths, conditional for paths with spaces.
    Both approaches exit silently (code 0) when not launched via 'hcom open'.
    """
    python_path = sys.executable
    script_path = os.path.abspath(__file__)
    
    if IS_WINDOWS:
        # Windows cmd.exe syntax - no parentheses so arguments append correctly
        if ' ' in python_path or ' ' in script_path:
            return f'IF "%HCOM_ACTIVE%"=="1" "{python_path}" "{script_path}"', {}
        return f'IF "%HCOM_ACTIVE%"=="1" {python_path} {script_path}', {}
    elif ' ' in python_path or ' ' in script_path:
        # Unix with spaces: use conditional check
        escaped_python = shell_quote(python_path)
        escaped_script = shell_quote(script_path)
        return f'[ "${{HCOM_ACTIVE}}" = "1" ] && {escaped_python} {escaped_script} || true', {}
    else:
        # Unix clean paths: use environment variable
        return '${HCOM:-true}', {}

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
    script_path = os.path.abspath(__file__)
    if ' ' not in python_path and ' ' not in script_path:
        env['HCOM'] = f'{python_path} {script_path}'
    
    return env

# ==================== Message System ====================

def validate_message(message):
    """Validate message size and content"""
    if not message or not message.strip():
        return format_error("Message required")

    # Reject control characters (except \n, \r, \t)
    if re.search(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\u0080-\u009F]', message):
        return format_error("Message contains control characters")

    max_size = get_config_value('max_message_size', 1048576)
    if len(message) > max_size:
        return format_error(f"Message too large (max {max_size} chars)")

    return None

def send_message(from_instance, message):
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

def should_deliver_message(msg, instance_name, all_instance_names=None):
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

# ==================== Parsing and Helper Functions ====================

def parse_open_args(args):
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
                raise ValueError(format_error('Team name cannot contain pipe characters'))
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

def extract_agent_config(content):
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
    model_match = re.search(r'^model:\s*(.+)$', yaml_section, re.MULTILINE)
    if model_match:
        value = model_match.group(1).strip()
        if value and value.lower() != 'inherit':
            config['model'] = value
    
    # Extract tools field
    tools_match = re.search(r'^tools:\s*(.+)$', yaml_section, re.MULTILINE)
    if tools_match:
        value = tools_match.group(1).strip()
        if value:
            config['tools'] = value.replace(', ', ',')
    
    return config

def resolve_agent(name):
    """Resolve agent file by name
    
    Looks for agent files in:
    1. .claude/agents/{name}.md (local)
    2. ~/.claude/agents/{name}.md (global)
    
    Returns tuple: (content after stripping YAML frontmatter, config dict)
    """
    for base_path in [Path.cwd(), Path.home()]:
        agent_path = base_path / '.claude/agents' / f'{name}.md'
        if agent_path.exists():
            content = read_file_with_retry(
                agent_path,
                lambda f: f.read(),
                default=None
            )
            if content is None:
                continue  # Skip to next base_path if read failed
            config = extract_agent_config(content)
            stripped = strip_frontmatter(content)
            if not stripped.strip():
                raise ValueError(format_error(f"Agent '{name}' has empty content", 'Check the agent file contains a system prompt'))
            return stripped, config
    
    raise FileNotFoundError(format_error(f'Agent not found: {name}', 'Check available agents or create the agent file'))

def strip_frontmatter(content):
    """Strip YAML frontmatter from agent file"""
    if content.startswith('---'):
        # Find the closing --- on its own line
        lines = content.splitlines()
        for i, line in enumerate(lines[1:], 1):
            if line.strip() == '---':
                return '\n'.join(lines[i+1:]).strip()
    return content

def get_display_name(session_id, prefix=None):
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

        # Collision detection: if taken by another PID, use more session_id chars
        instance_file = hcom_path(INSTANCES_DIR, f"{base_name}.json")
        if instance_file.exists():
            try:
                with open(instance_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                their_pid = data.get('pid')
                our_pid = os.getppid()
                # Only consider it a collision if they have a PID and it's different
                if their_pid and their_pid != our_pid:
                    # Use first 4 chars of session_id for collision resolution
                    base_name = f"{dir_char}{session_id[0:4]}"
            except:
                pass
    else:
        # Fallback to PID-based naming if no session_id
        pid_suffix = os.getppid() % 10000
        base_name = f"{dir_char}{pid_suffix}claude"

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
    # - $HCOM post/stop/notify
    # - ${HCOM:-...} post/stop/notify
    # - [ "${HCOM_ACTIVE}" = "1" ] && ... hcom.py ... || true
    # - hcom post/stop/notify  
    # - uvx hcom post/stop/notify
    # - /path/to/hcom.py post/stop/notify
    # - sh -c "[ ... ] && ... hcom ..."
    # - "/path with spaces/python" "/path with spaces/hcom.py" post/stop/notify
    # - '/path/to/python' '/path/to/hcom.py' post/stop/notify
    # Note: Modern hooks use either ${HCOM:-true} (pattern 1) or the HCOM_ACTIVE conditional
    # with full paths (pattern 2), both of which match all hook types including pre/sessionstart.
    # The (post|stop|notify) patterns (3-6) are for older direct command formats that didn't
    # include pre/sessionstart hooks.
    hcom_patterns = [
        r'\$\{?HCOM',                                # Environment variable (${HCOM:-true}) - all hook types
        r'\bHCOM_ACTIVE.*hcom\.py',                 # Conditional with full path - all hook types
        r'\bhcom\s+(post|stop|notify)\b',           # Direct hcom command (older format)
        r'\buvx\s+hcom\s+(post|stop|notify)\b',     # uvx hcom command (older format)
        r'hcom\.py["\']?\s+(post|stop|notify)\b',   # hcom.py with optional quote (older format)
        r'["\'][^"\']*hcom\.py["\']?\s+(post|stop|notify)\b(?=\s|$)',  # Quoted path (older format)
        r'sh\s+-c.*hcom',                           # Shell wrapper with hcom
    ]
    compiled_patterns = [re.compile(pattern) for pattern in hcom_patterns]
    
    for event in ['SessionStart', 'PreToolUse', 'PostToolUse', 'Stop', 'Notification']:
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


def format_error(message, suggestion=None):
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

def build_claude_command(agent_content=None, claude_args=None, initial_prompt="Say hi in chat", model=None, tools=None):
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
        temp_file = tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', suffix='.txt', delete=False,
                                              prefix='hcom_agent_', dir=tempfile.gettempdir())
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
    cmd_parts.append(shell_quote(initial_prompt))
    
    return ' '.join(cmd_parts), temp_file_path

def escape_for_platform(text, platform_type):
    """Centralized escaping for different platforms"""
    if platform_type == 'applescript':
        # AppleScript escaping for text within double quotes
        # We need to escape backslashes first, then other special chars
        return (text.replace('\\', '\\\\')
                   .replace('"', '\\"')  # Escape double quotes
                   .replace('\n', '\\n')  # Escape newlines
                   .replace('\r', '\\r')  # Escape carriage returns  
                   .replace('\t', '\\t'))  # Escape tabs
    else:  # POSIX/bash
        return shlex.quote(text)

def shell_quote(text):
    """Cross-platform shell argument quoting

    Note: On Windows with Git Bash, subprocess.Popen(shell=True) uses bash, not cmd.exe
    """
    # Always use shlex.quote for proper bash escaping
    # Git Bash on Windows uses bash as the shell
    return shlex.quote(text)

def create_bash_script(script_file, env, cwd, command_str, background=False):
    """Create a bash script for terminal launch"""
    try:
        # Ensure parent directory exists
        script_path = Path(script_file)
        script_path.parent.mkdir(parents=True, exist_ok=True)
    except (OSError, IOError) as e:
        raise Exception(f"Cannot create script directory: {e}")

    with open(script_file, 'w', encoding='utf-8') as f:
        f.write('#!/bin/bash\n')
        f.write('echo "Starting Claude Code..."\n')
        f.write(build_env_string(env, "bash_export") + '\n')
        if cwd:
            f.write(f'cd {shlex.quote(cwd)}\n')

        # On Windows, let bash resolve claude from PATH (Windows paths don't work in bash)
        if platform.system() != 'Windows':
            claude_path = shutil.which('claude')
            if claude_path:
                command_str = command_str.replace('claude ', f'{claude_path} ', 1)

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
                os.path.join(base, 'Git', 'usr', 'bin', 'bash.exe'),  # usr/bin is more common
                os.path.join(base, 'Git', 'bin', 'bash.exe')
            ])

    # 2. Portable Git installation
    local_appdata = os.environ.get('LOCALAPPDATA', '')
    if local_appdata:
        git_portable = os.path.join(local_appdata, 'Programs', 'Git')
        candidates.extend([
            os.path.join(git_portable, 'usr', 'bin', 'bash.exe'),
            os.path.join(git_portable, 'bin', 'bash.exe')
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
        if bash and os.path.exists(bash):
            return bash

    return None

def schedule_file_cleanup(files, delay=5):
    """Schedule cleanup of temporary files after delay"""
    if not files:
        return

    def cleanup():
        time.sleep(delay)
        for file_path in files:
            try:
                os.unlink(file_path)
            except (OSError, FileNotFoundError):
                pass  # File already deleted or inaccessible

    thread = threading.Thread(target=cleanup, daemon=True)
    thread.start()

def launch_terminal(command, env, config=None, cwd=None, background=False):
    """Launch terminal with command
    Args:
        command: Either a string command or list of command parts
        env: Environment variables to set
        config: Configuration dict
        cwd: Working directory
        background: Launch as background process
    """

    if config is None:
        config = get_cached_config()

    env_vars = os.environ.copy()
    env_vars.update(env)

    # Command should now always be a string from build_claude_command
    command_str = command
    
    # Background mode implementation
    if background:
        # Create log file for background instance
        logs_dir = hcom_path(LOGS_DIR)
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_file = logs_dir / env['HCOM_BACKGROUND']

        # Launch detached process
        try:
            with open(log_file, 'w', encoding='utf-8') as log_handle:
                if IS_WINDOWS:
                    # Windows: Use bash script approach for proper $(cat ...) support
                    bash_exe = find_bash_on_windows()
                    if not bash_exe:
                        raise Exception("Git Bash not found")

                    # Create script file for background process
                    script_file = str(hcom_path('scripts',
                        f'background_{os.getpid()}_{random.randint(1000,9999)}.sh',
                        ensure_parent=True))

                    create_bash_script(script_file, env, cwd, command_str, background=True)

                    # Windows requires STARTUPINFO to hide console window
                    startupinfo = subprocess.STARTUPINFO()
                    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                    startupinfo.wShowWindow = subprocess.SW_HIDE

                    # Use bash.exe directly with script, avoiding shell=True and cmd.exe
                    process = subprocess.Popen(
                        [bash_exe, script_file],
                        env=env_vars,
                        cwd=cwd,
                        stdin=subprocess.DEVNULL,
                        stdout=log_handle,
                        stderr=subprocess.STDOUT,
                        startupinfo=startupinfo,
                        creationflags=CREATE_NO_WINDOW  # Belt and suspenders approach
                    )
                else:
                    # Unix: use start_new_session for proper detachment
                    process = subprocess.Popen(
                        command_str,
                        shell=True,
                        env=env_vars,
                        cwd=cwd,
                        stdin=subprocess.DEVNULL,
                        stdout=log_handle,
                        stderr=subprocess.STDOUT,
                        start_new_session=True  # detach from terminal session
                    )
        except OSError as e:
            print(format_error(f"Failed to launch background instance: {e}"), file=sys.stderr)
            return None
        
        # Check for immediate failures
        time.sleep(0.2)
        if process.poll() is not None:
            # Process already exited
            error_output = read_file_with_retry(
                log_file,
                lambda f: f.read()[:1000],
                default=""
            )
            print(format_error("Background instance failed immediately"), file=sys.stderr)
            if error_output:
                print(f"  Output: {error_output}", file=sys.stderr)
            return None
        
        return str(log_file)

    terminal_mode = get_config_value('terminal_mode', 'new_window')

    if terminal_mode == 'show_commands':
        env_str = build_env_string(env)
        print(f"{env_str} {command_str}")
        return True

    elif terminal_mode == 'same_terminal':
        print(f"Launching Claude in current terminal...")
        if IS_WINDOWS:
            # Windows: Use bash directly to support $(cat ...) syntax
            bash_exe = find_bash_on_windows()
            if not bash_exe:
                print(format_error("Git Bash not found"), file=sys.stderr)
                return False
            # Run with bash -c to execute in current terminal
            result = subprocess.run([bash_exe, '-c', command_str], env=env_vars, cwd=cwd)
        else:
            # Unix/Linux/Mac: shell=True works fine
            result = subprocess.run(command_str, shell=True, env=env_vars, cwd=cwd)

        return result.returncode == 0
    
    system = platform.system()

    custom_cmd = get_config_value('terminal_command')

    # Check for macOS Terminal.app 1024 char TTY limit
    if not custom_cmd and system == 'Darwin':
        full_cmd = build_env_string(env, "bash_export") + command_str
        if cwd:
            full_cmd = f'cd {shlex.quote(cwd)}; {full_cmd}'
        if len(full_cmd) > 1000:  # Switch before 1024 limit
            # Force script path by setting custom_cmd
            custom_cmd = 'osascript -e \'tell app "Terminal" to do script "{script}"\''

    # Windows also needs script approach - treat it like custom terminal
    if system == 'Windows' and not custom_cmd:
        # Windows always uses script approach with bash
        bash_exe = find_bash_on_windows()
        if not bash_exe:
            raise Exception(format_error("Git Bash not found"))
        # Set up to use script approach below
        if shutil.which('wt'):
            # Windows Terminal available
            custom_cmd = f'wt {bash_exe} {{script}}'
        else:
            # Use cmd.exe with start command for visible window
            custom_cmd = f'cmd /c start "Claude Code" {bash_exe} {{script}}'

    if custom_cmd and custom_cmd != 'None' and custom_cmd != 'null':
        # Check for {script} placeholder - the reliable way to handle complex commands
        if '{script}' in custom_cmd:
            # Create temp script file
            # Always use .sh extension for bash scripts
            script_ext = '.sh'
            # Use ~/.hcom/scripts/ instead of /tmp to avoid noexec issues
            script_file = str(hcom_path('scripts',
                f'launch_{os.getpid()}_{random.randint(1000,9999)}{script_ext}',
                ensure_parent=True))

            # Create the bash script using helper
            create_bash_script(script_file, env, cwd, command_str, background)

            # Replace {script} with the script path
            final_cmd = custom_cmd.replace('{script}', shlex.quote(script_file))

            # Windows needs special flags
            if system == 'Windows':
                # Use Popen for non-blocking launch on Windows
                # Use shell=True for Windows to handle complex commands properly
                subprocess.Popen(final_cmd, shell=True)
            else:
                # TODO: Test if macOS/Linux will still work with Popen for parallel launches with custom terminals like windows
                result = subprocess.run(final_cmd, shell=True, capture_output=True)

            # Schedule cleanup for all scripts (since we don't wait for completion)
            schedule_file_cleanup([script_file])

            return True

        # No {script} placeholder found
        else:
            print(format_error("Custom terminal command must use {script} placeholder",
                             "Example: open -n -a kitty.app --args bash \"{script}\"'"),
                  file=sys.stderr)
            return False
    
    if system == 'Darwin':  # macOS
        env_setup = build_env_string(env, "bash_export")
        # Include cd command if cwd is specified
        if cwd:
            full_cmd = f'cd {shlex.quote(cwd)}; {env_setup} {command_str}'
        else:
            full_cmd = f'{env_setup} {command_str}'
        
        # Escape the command for AppleScript double-quoted string
        escaped = escape_for_platform(full_cmd, 'applescript')
        
        script = f'tell app "Terminal" to do script "{escaped}"'
        subprocess.run(['osascript', '-e', script])
        return True
    
    elif system == 'Linux':
        # Try Linux terminals first (works for both regular Linux and WSLg)
        terminals = [
            ('gnome-terminal', ['gnome-terminal', '--', 'bash', '-c']),
            ('konsole', ['konsole', '-e', 'bash', '-c']),
            ('xterm', ['xterm', '-e', 'bash', '-c'])
        ]

        for term_name, term_cmd in terminals:
            if shutil.which(term_name):
                env_cmd = build_env_string(env)
                # Include cd command if cwd is specified
                if cwd:
                    full_cmd = f'cd "{cwd}"; {env_cmd} {command_str}; exec bash'
                else:
                    full_cmd = f'{env_cmd} {command_str}; exec bash'
                subprocess.run(term_cmd + [full_cmd])
                return True

        # No Linux terminals found - check if WSL and can use Windows Terminal as fallback
        if is_wsl() and shutil.which('cmd.exe'):
            # WSL fallback: Use Windows Terminal through cmd.exe
            script_file = str(hcom_path('scripts',
                f'wsl_launch_{os.getpid()}_{random.randint(1000,9999)}.sh',
                ensure_parent=True))

            create_bash_script(script_file, env, cwd, command_str, background=False)

            # Use Windows Terminal if available, otherwise cmd.exe with start
            if shutil.which('wt.exe'):
                subprocess.run(['cmd.exe', '/c', 'start', 'wt.exe', 'bash', script_file])
            else:
                subprocess.run(['cmd.exe', '/c', 'start', 'bash', script_file])

            # Schedule cleanup
            schedule_file_cleanup([script_file], delay=5)
            return True

        raise Exception(format_error("No supported terminal emulator found", "Install gnome-terminal, konsole, or xterm"))
        
    else:
        # Windows is now handled by the custom_cmd logic above
        raise Exception(format_error(f"Unsupported platform: {system}", "Supported platforms: macOS, Linux, Windows"))

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
    
    # Define all hooks
    hook_configs = [
        ('SessionStart', '', f'{hook_cmd_base} sessionstart', None),
        ('PreToolUse', 'Bash', f'{hook_cmd_base} pre', None),
        ('PostToolUse', '.*', f'{hook_cmd_base} post', None),
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

        # Check all hook types exist with HCOM commands
        hooks = settings.get('hooks', {})
        for hook_type in ['SessionStart', 'PreToolUse', 'PostToolUse', 'Stop', 'Notification']:
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
    except (TypeError, ValueError) as e:
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
        except Exception as e:
            return False
    else:
        # Unix: Use os.kill with signal 0
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError as e:
            return False
        except Exception as e:
            return False

def parse_log_messages(log_file, start_pos=0, return_end_pos=False):
    """Parse messages from log file
    Args:
        log_file: Path to log file
        start_pos: Position to start reading from
        return_end_pos: If True, return tuple (messages, end_position)
    Returns:
        list of messages, or (messages, end_pos) if return_end_pos=True
    """
    if not log_file.exists():
        return ([], start_pos) if return_end_pos else []

    def read_messages(f):
        f.seek(start_pos)
        content = f.read()
        end_pos = f.tell()  # Capture actual end position

        if not content.strip():
            return ([], end_pos)

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

        return (messages, end_pos)

    result = read_file_with_retry(
        log_file,
        read_messages,
        default=([], start_pos)
    )

    return result if return_end_pos else result[0]

def get_new_messages(instance_name):
    """Get new messages for instance with @-mention filtering"""
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
    all_messages, new_pos = parse_log_messages(log_file, last_pos, return_end_pos=True)

    # Filter messages:
    # 1. Exclude own messages
    # 2. Apply @-mention filtering
    all_instance_names = list(positions.keys())
    messages = []
    for msg in all_messages:
        if msg['from'] != instance_name:
            if should_deliver_message(msg, instance_name, all_instance_names):
                messages.append(msg)

    # Update position to what was actually processed
    update_instance_position(instance_name, {'pos': new_pos})

    return messages

def format_age(seconds):
    """Format time ago in human readable form"""
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        return f"{int(seconds/60)}m"
    else:
        return f"{int(seconds/3600)}h"

def get_transcript_status(transcript_path):
    """Parse transcript to determine current Claude state"""
    if not transcript_path or not os.path.exists(transcript_path):
        return "inactive", "", "", 0

    def read_status(f):
        # Windows file buffering fix: read entire file to get current content
        if IS_WINDOWS:
            # Seek to beginning and read all content to bypass Windows file caching
            f.seek(0)
            all_content = f.read()
            all_lines = all_content.strip().split('\n')
            lines = all_lines[-5:] if len(all_lines) >= 5 else all_lines
        else:
            lines = f.readlines()[-5:]

        for i, line in enumerate(reversed(lines)):
            try:
                entry = json.loads(line)
                timestamp = datetime.fromisoformat(entry['timestamp']).timestamp()
                age = int(time.time() - timestamp)
                entry_type = entry.get('type', '')

                if entry['type'] == 'system':
                    content = entry.get('content', '')
                    if 'Running' in content:
                        tool_name = content.split('Running ')[1].split('[')[0].strip()
                        return "executing", f"({format_age(age)})", tool_name, timestamp

                elif entry['type'] == 'assistant':
                    content = entry.get('content', [])
                    has_tool_use = any('tool_use' in str(item) for item in content)
                    if has_tool_use:
                        return "executing", f"({format_age(age)})", "tool", timestamp
                    else:
                        return "responding", f"({format_age(age)})", "", timestamp

                elif entry['type'] == 'user':
                    return "thinking", f"({format_age(age)})", "", timestamp
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                continue

        return "inactive", "", "", 0

    try:
        result = read_file_with_retry(
            transcript_path,
            read_status,
            default=("inactive", "", "", 0)
        )
        return result
    except Exception:
        return "inactive", "", "", 0

def get_instance_status(pos_data):
    """Get current status of instance"""
    now = int(time.time())
    wait_timeout = pos_data.get('wait_timeout', get_config_value('wait_timeout', 1800))

    # Check if process is still alive. pid: null means killed
    # All real instances should have a PID (set by update_instance_with_pid)
    if 'pid' in pos_data:
        pid = pos_data['pid']
        if pid is None:
            # Explicitly null = was killed
            return "inactive", ""
        if not is_process_alive(pid):
            # On Windows, PID checks can fail during process transitions
            # Let timeout logic handle this using activity timestamps
            wait_timeout = 30 if IS_WINDOWS else wait_timeout  # Shorter timeout when PID dead

    last_permission = pos_data.get("last_permission_request", 0)
    last_stop = pos_data.get("last_stop", 0)
    last_tool = pos_data.get("last_tool", 0)

    transcript_timestamp = 0
    transcript_status = "inactive"

    transcript_path = pos_data.get("transcript_path", "")
    if transcript_path:
        status, _, _, transcript_timestamp = get_transcript_status(transcript_path)
        transcript_status = status

    # Calculate last actual activity (excluding heartbeat)
    last_activity = max(last_permission, last_tool, transcript_timestamp)

    # Check timeout based on actual activity
    if last_activity > 0 and (now - last_activity) > wait_timeout:
        return "inactive", ""

    # Now determine current status including heartbeat
    events = [
        (last_permission, "blocked"),
        (last_stop, "waiting"),
        (last_tool, "inactive"),
        (transcript_timestamp, transcript_status)
    ]

    recent_events = [(ts, status) for ts, status in events if ts > 0]
    if not recent_events:
        return "inactive", ""

    most_recent_time, most_recent_status = max(recent_events)
    age = now - most_recent_time

    status_suffix = " (bg)" if pos_data.get('background') else ""
    final_result = (most_recent_status, f"({format_age(age)}){status_suffix}")

    return final_result

def get_status_block(status_type):
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
        messages = parse_log_messages(log_file)
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
                last_tool = pos_data.get("last_tool", 0)
                last_tool_name = pos_data.get("last_tool_name", "unknown")
                last_tool_str = datetime.fromtimestamp(last_tool).strftime("%H:%M:%S") if last_tool else "unknown"
                
                # Get session IDs (already migrated to array format)
                session_ids = pos_data.get("session_ids", [])
                sid = session_ids[-1] if session_ids else ""  # Show most recent session
                session_info = f" | {sid}" if sid else ""
                
                print(f"   {FG_GREEN}->{RESET} {BOLD}{instance_name}{RESET} {status_block} {DIM}{status_type} {age}- used {last_tool_name} at {last_tool_str}{session_info}{RESET}")
            print()
    else:
        print(f"   {DIM}Error reading instance data{RESET}")

def alt_screen_detailed_status_and_input():
    """Show detailed status in alt screen and get user input"""
    sys.stdout.write("\033[?1049h\033[2J\033[H")
    
    try:
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"{BOLD} HCOM DETAILED STATUS{RESET}")
        print(f"{BOLD}{'=' * 70}{RESET}")
        print(f"{FG_CYAN} HCOM: GLOBAL CHAT{RESET}")
        print(f"{DIM} LOG FILE: {hcom_path(LOG_FILE)}{RESET}")
        print(f"{DIM} UPDATED: {timestamp}{RESET}")
        print(f"{BOLD}{'-' * 70}{RESET}")
        print()
        
        show_instances_by_directory()
        
        print()
        print(f"{BOLD} RECENT ACTIVITY:{RESET}")
        
        show_recent_activity_alt_screen()
        
        print()
        print(f"{BOLD}{'-' * 70}{RESET}")
        print(f"{FG_GREEN} Type message and press Enter to send (empty to cancel):{RESET}")
        message = input(f"{FG_CYAN} > {RESET}")
        
        print(f"{BOLD}{'=' * 70}{RESET}")
        
    finally:
        sys.stdout.write("\033[?1049l")
    
    return message

def get_status_summary():
    """Get a one-line summary of all instance statuses"""
    positions = load_all_positions()
    if not positions:
        return f"{BG_BLUE}{BOLD}{FG_WHITE} no instances {RESET}"

    status_counts = {"thinking": 0, "responding": 0, "executing": 0, "waiting": 0, "blocked": 0, "inactive": 0}

    for instance_name, pos_data in positions.items():
        status_type, _ = get_instance_status(pos_data)
        if status_type in status_counts:
            status_counts[status_type] += 1

    parts = []
    status_order = ["thinking", "responding", "executing", "waiting", "blocked", "inactive"]

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
            "last_tool": 0,
            "last_tool_name": "unknown",
            "last_stop": 0,
            "last_permission_request": 0,
            "session_ids": [session_id] if session_id else [],
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

def merge_instance_data(to_data, from_data):
    """Merge instance data from from_data into to_data."""
    # Merge session_ids arrays with deduplication
    to_sessions = to_data.get('session_ids', [])
    from_sessions = from_data.get('session_ids', [])
    to_data['session_ids'] = list(dict.fromkeys(to_sessions + from_sessions))

    # Update transient fields from source
    to_data['pid'] = os.getppid()  # Always use current PID
    to_data['transcript_path'] = from_data.get('transcript_path', to_data.get('transcript_path', ''))

    # Preserve maximum position
    to_data['pos'] = max(to_data.get('pos', 0), from_data.get('pos', 0))

    # Update directory to most recent
    to_data['directory'] = from_data.get('directory', to_data.get('directory', str(Path.cwd())))

    # Update last activity timestamps to most recent
    to_data['last_tool'] = max(to_data.get('last_tool', 0), from_data.get('last_tool', 0))
    to_data['last_tool_name'] = from_data.get('last_tool_name', to_data.get('last_tool_name', 'unknown'))
    to_data['last_stop'] = max(to_data.get('last_stop', 0), from_data.get('last_stop', 0))
    to_data['last_permission_request'] = max(
        to_data.get('last_permission_request', 0),
        from_data.get('last_permission_request', 0)
    )

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

        # Check if target is active
        if to_data.get('pid'):
            if is_process_alive(to_data['pid']):
                return f"Cannot recover {to_name}: instance is active"
            # Process is dead, safe to merge

        # Merge data using helper
        to_data = merge_instance_data(to_data, from_data)

        # Save merged data - check for success
        if not save_instance_position(to_name, to_data):
            return f"Failed to save merged data for {to_name}"

        # Cleanup source file only after successful save
        try:
            hcom_path(INSTANCES_DIR, f"{from_name}.json").unlink()
        except:
            pass  # Non-critical if cleanup fails

        return f"[SUCCESS] ✓ Recovered: {from_name} → {to_name}"
    except Exception:
        return f"Failed to merge {from_name} into {to_name}"


# ==================== Command Functions ====================

def show_main_screen_header():
    """Show header for main screen"""
    sys.stdout.write("\033[2J\033[H")
    
    log_file = hcom_path(LOG_FILE)
    all_messages = []
    if log_file.exists():
        all_messages = parse_log_messages(log_file)
    message_count = len(all_messages)
    
    print(f"\n{BOLD}{'='*50}{RESET}")
    print(f"  {FG_CYAN}HCOM: global chat{RESET}")
    
    status_line = get_status_summary()
    print(f"  {BOLD}INSTANCES:{RESET} {status_line}")
    print(f"  {DIM}LOGS: {log_file} ({message_count} messages){RESET}")
    print(f"{BOLD}{'='*50}{RESET}\n")
    
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

CONCEPT: HCOM creates multi-agent collaboration by launching multiple Claude Code 
instances in separate terminals that share a group chat.

KEY UNDERSTANDING:
• Single conversation - All instances share ~/.hcom/hcom.log
• CLI usage - Use 'hcom send' for messaging. Internal instances know to use 'echo HCOM_SEND:'
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
• ◉ thinking, ▷ responding, ▶ executing - instance is working
• ◉ waiting - instance is waiting for new messages
• ■ blocked - instance is blocked by permission request (needs user approval)
• ○ inactive - instance is timed out, disconnected, etc
              
CONFIG:
Config file (persistent): ~/.hcom/config.json

Key settings (full list in config.json):
  terminal_mode: "new_window" (default) | "same_terminal" | "show_commands"
  initial_prompt: "Say hi in chat", first_use_text: "Essential messages only..."
  instance_hints: "text", cli_hints: "text"  # Extra info for instances/CLI
  env_overrides: "custom environment variables for instances"

Temporary environment overrides for any setting (all caps & append HCOM_):
HCOM_INSTANCE_HINTS="useful info" hcom open  # applied to all messages recieved by instance
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

        # Extract resume sessionId if present
        resume_session_id = None
        if claude_args:
            for i, arg in enumerate(claude_args):
                if arg in ['--resume', '-r'] and i + 1 < len(claude_args):
                    resume_session_id = claude_args[i + 1]
                    break

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

        # Pass resume sessionId to hooks (only for first instance if multiple)
        # This avoids conflicts when resuming with -n > 1
        if resume_session_id:
            if len(instances) > 1:
                print(f"Warning: --resume with {len(instances)} instances will only resume the first instance", file=sys.stderr)
            # Will be added to first instance env only

        # Add prefix-specific hints if provided
        if prefix:
            base_env['HCOM_PREFIX'] = prefix
            hint = f"To respond to {prefix} group: echo 'HCOM_SEND:@{prefix} message'"
            base_env['HCOM_INSTANCE_HINTS'] = hint

            first_use = f"You're in the {prefix} group. Use {prefix} to message: echo HCOM_SEND:@{prefix} message."
            base_env['HCOM_FIRST_USE_TEXT'] = first_use
        
        launched = 0
        initial_prompt = get_config_value('initial_prompt', 'Say hi in chat')
        
        temp_files_to_cleanup = []
        
        for idx, instance_type in enumerate(instances):
            instance_env = base_env.copy()

            # Add resume sessionId only to first instance when multiple instances
            if resume_session_id and idx == 0:
                instance_env['HCOM_RESUME_SESSION_ID'] = resume_session_id
            
            # Mark background instances via environment with log filename
            if background:
                # Generate unique log filename
                log_filename = f'background_{int(time.time())}_{random.randint(1000, 9999)}.log'
                instance_env['HCOM_BACKGROUND'] = log_filename
            
            # Build claude command
            if instance_type == 'generic':
                # Generic instance - no agent content
                claude_cmd, temp_file = build_claude_command(
                    agent_content=None,
                    claude_args=claude_args,
                    initial_prompt=initial_prompt
                )
            else:
                # Agent instance
                try:
                    agent_content, agent_config = resolve_agent(instance_type)
                    # Prepend agent instance awareness to system prompt
                    agent_prefix = f"You are an instance of {instance_type}. Do not start a subagent with {instance_type} unless explicitly asked.\n\n"
                    agent_content = agent_prefix + agent_content
                    # Use agent's model and tools if specified and not overridden in claude_args
                    agent_model = agent_config.get('model')
                    agent_tools = agent_config.get('tools')
                    claude_cmd, temp_file = build_claude_command(
                        agent_content=agent_content,
                        claude_args=claude_args,
                        initial_prompt=initial_prompt,
                        model=agent_model,
                        tools=agent_tools
                    )
                    if temp_file:
                        temp_files_to_cleanup.append(temp_file)
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
                    launch_terminal(claude_cmd, instance_env, cwd=os.getcwd())
                    launched += 1
            except Exception as e:
                print(format_error(f"Failed to launch terminal: {e}"), file=sys.stderr)
        
        # Clean up temp files after a delay (let terminals read them first)
        if temp_files_to_cleanup:
            schedule_file_cleanup(temp_files_to_cleanup)
                
        if launched == 0:
            print(format_error("No instances launched"), file=sys.stderr)
            return 1
            
        # Success message
        print(f"Launched {launched} Claude instance{'s' if launched != 1 else ''}")

        # Auto-launch watch dashboard if configured and conditions are met
        terminal_mode = get_config_value('terminal_mode')
        auto_watch = get_config_value('auto_watch', True)

        if terminal_mode == 'new_window' and auto_watch and launched > 0 and is_interactive():
            # Show tips first if needed
            if prefix:
                print(f"\n  • Send to {prefix} team: hcom send '@{prefix} message'")

            # Launch interactive watch dashboard in current terminal
            return cmd_watch()
        else:
            tips = [
                "Run 'hcom watch' to view/send in conversation dashboard",
            ]
            if prefix:
                tips.append(f"Send to {prefix} team: hcom send '@{prefix} message'")

            if tips:
                print("\n" + "\n".join(f"  • {tip}" for tip in tips))

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
        print(format_error("No conversation found", "Run 'hcom open' first"), file=sys.stderr)
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
                messages = parse_log_messages(log_file)
            else:
                last_pos = 0
                messages = []
            
            # If --wait, show only recent messages to prevent context bloat
            if wait_timeout is not None:
                cutoff = datetime.now() - timedelta(seconds=5)
                recent_messages = [m for m in messages if datetime.fromisoformat(m['timestamp']) > cutoff]
                
                # Status to stderr, data to stdout
                if recent_messages:
                    print(f'---Showing last 5 seconds of messages---', file=sys.stderr)
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
                            new_messages = parse_log_messages(log_file, last_pos)
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
                    "session_ids": data.get("session_ids", []),
                    "last_tool": data.get("last_tool_name", "unknown"),
                    "last_tool_time": data.get("last_tool", 0),
                    "pid": data.get("pid"),
                    "background": bool(data.get("background"))
                }
                status_counts[status] = status_counts.get(status, 0) + 1

            # Get recent messages
            messages = []
            if log_file.exists():
                all_messages = parse_log_messages(log_file)
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
    print(f"\n{DIM}{'─'*10} [watching for new messages] {'─'*10}{RESET}")
    
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
                    new_messages = parse_log_messages(log_file, last_pos)
                    # Use the last known status for consistency
                    status_line_text = f"{last_status}{status_suffix}"
                    for msg in new_messages:
                        log_line_with_status(format_message_line(msg), status_line_text)
                    last_pos = current_size
            
            # Check for keyboard input
            ready_for_input = False
            if IS_WINDOWS:
                import msvcrt
                if msvcrt.kbhit():
                    msvcrt.getch()
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
                print(f"\n{DIM}{'─'*10} [watching for new messages] {'─'*10}{RESET}")
                
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
        
        original_hook_count = sum(len(settings.get('hooks', {}).get(event, []))
                                  for event in ['SessionStart', 'PreToolUse', 'PostToolUse', 'Stop', 'Notification'])
        
        _remove_hcom_hooks_from_settings(settings)
        
        # Check if any were removed
        new_hook_count = sum(len(settings.get('hooks', {}).get(event, []))
                             for event in ['SessionStart', 'PreToolUse', 'PostToolUse', 'Stop', 'Notification'])
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
    
    # Send message
    sender_name = get_config_value('sender_name', 'bigboss')
    
    if send_message(sender_name, message):
        print("Message sent", file=sys.stderr)
        
        # Show cli_hints if configured (non-interactive mode)
        if not is_interactive():
            show_cli_hints()
        
        return 0
    else:
        print(format_error("Failed to send message"), file=sys.stderr)
        return 1

# ==================== Hook Functions ====================

def format_hook_messages(messages, instance_name):
    """Format messages for hook feedback"""
    if len(messages) == 1:
        msg = messages[0]
        reason = f"[new message] {msg['from']} → {instance_name}: {msg['message']}"
    else:
        parts = [f"{msg['from']} → {instance_name}: {msg['message']}" for msg in messages]
        reason = f"[{len(messages)} new messages] | " + " | ".join(parts)

    # Check alias announcement
    instance_data = load_instance_position(instance_name)
    if not instance_data.get('alias_announced', False) and not instance_name.endswith('claude'):
        reason = f"{reason} | [Alias assigned: {instance_name}] <Your hcom chat alias is {instance_name}. You can at-mention others in hcom chat by their alias to DM them. (alias1 → alias2 means alias1 sent the message to the entire group, if there is an at symbol in the message then it is targeted)>"
        update_instance_position(instance_name, {'alias_announced': True})

    # Only append instance_hints to messages (first_use_text is handled separately)
    instance_hints = get_config_value('instance_hints', '')
    if instance_hints:
        reason = f"{reason} | [{instance_hints}]"

    return reason

def get_pending_tools(transcript_path, max_lines=100):
    """Parse transcript to find tool_use IDs without matching tool_results.
    Returns count of pending tools."""
    if not transcript_path or not os.path.exists(transcript_path):
        return 0

    tool_uses = set()
    tool_results = set()

    try:
        # Read last N lines efficiently
        with open(transcript_path, 'rb') as f:
            # Seek to end and read backwards
            f.seek(0, 2)  # Go to end
            file_size = f.tell()
            read_size = min(file_size, max_lines * 500)  # Assume ~500 bytes per line
            f.seek(max(0, file_size - read_size))
            recent_content = f.read().decode('utf-8', errors='ignore')

        # Parse line by line (handle both Unix \n and Windows \r\n)
        for line in recent_content.splitlines():
            if not line.strip():
                continue
            try:
                data = json.loads(line)

                # Check for tool_use blocks in assistant messages
                if data.get('type') == 'assistant':
                    content = data.get('message', {}).get('content', [])
                    if isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict) and item.get('type') == 'tool_use':
                                tool_id = item.get('id')
                                if tool_id:
                                    tool_uses.add(tool_id)

                # Check for tool_results in user messages
                elif data.get('type') == 'user':
                    content = data.get('message', {}).get('content', [])
                    if isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict) and item.get('type') == 'tool_result':
                                tool_id = item.get('tool_use_id')
                                if tool_id:
                                    tool_results.add(tool_id)
            except Exception as e:
                continue

        # Return count of pending tools
        pending = tool_uses - tool_results
        return len(pending)
    except Exception as e:
        return 0  # On any error, assume no pending tools

# ==================== Hook Handlers ====================

def init_hook_context(hook_data):
    """Initialize instance context - shared by post/stop/notify hooks"""
    session_id = hook_data.get('session_id', '')
    transcript_path = hook_data.get('transcript_path', '')
    prefix = os.environ.get('HCOM_PREFIX')

    # Check if this is a resume operation
    resume_session_id = os.environ.get('HCOM_RESUME_SESSION_ID')
    instances_dir = hcom_path(INSTANCES_DIR)
    instance_name = None
    merged_state = None

    # First, try to find existing instance by resume sessionId
    if resume_session_id and instances_dir.exists():
        for instance_file in instances_dir.glob("*.json"):
            try:
                data = load_instance_position(instance_file.stem)
                # Check if resume_session_id matches any in the session_ids array
                old_session_ids = data.get('session_ids', [])
                if resume_session_id in old_session_ids:
                    # Found the instance! Keep the same name
                    instance_name = instance_file.stem
                    merged_state = data
                    # Append new session_id to array, update transcript_path to current
                    if session_id and session_id not in old_session_ids:
                        merged_state.setdefault('session_ids', old_session_ids).append(session_id)
                    if transcript_path:
                        merged_state['transcript_path'] = transcript_path
                    break
            except:
                continue

    # Check if current session exists in any instance's session_ids array
    # This maintains identity after implicit HCOM_RESUME
    if not instance_name and session_id and instances_dir.exists():
        for instance_file in instances_dir.glob("*.json"):
            try:
                data = load_instance_position(instance_file.stem)
                if session_id in data.get('session_ids', []):
                    instance_name = instance_file.stem
                    merged_state = data
                    break
            except:
                continue

    # If not found or not resuming, generate new name from session_id
    if not instance_name:
        instance_name = get_display_name(session_id, prefix)

    # PID deduplication: Clean up any stale instance files with same PID
    # Always run to clean up temp instances even after implicit resume
    parent_pid = os.getppid()
    if instances_dir.exists():
        for instance_file in instances_dir.glob("*.json"):
            if instance_file.stem != instance_name:  # Skip current instance
                try:
                    data = load_instance_position(instance_file.stem)
                    if data.get('pid') == parent_pid:
                        # Found duplicate with same PID - merge and delete
                        if not merged_state:
                            merged_state = data
                        else:
                            # Merge useful fields from duplicate
                            merged_state = merge_instance_data(merged_state, data)
                        instance_file.unlink()  # Delete the duplicate file
                        # Don't break - could have multiple duplicates with same PID
                except:
                    continue

    # Save migrated data if we have it
    if merged_state:
        save_instance_position(instance_name, merged_state)

    initialize_instance_in_position_file(instance_name, session_id)
    existing_data = load_instance_position(instance_name)

    # Prepare updates - use array for session_ids, single field for transcript_path
    updates = {
        'directory': str(Path.cwd()),
    }

    # Update session_ids array if we have a new session_id
    if session_id:
        current_session_ids = existing_data.get('session_ids', [])
        if session_id not in current_session_ids:
            current_session_ids.append(session_id)
            updates['session_ids'] = current_session_ids

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

    return instance_name, updates, existing_data

def extract_hcom_command(command, prefix='HCOM_SEND'):
    """Extract command payload with quote stripping"""
    marker = f'{prefix}:'
    if marker not in command:
        return None

    parts = command.split(marker, 1)
    if len(parts) <= 1:
        return None

    payload = parts[1].strip()

    # Complex quote stripping logic (preserves exact behavior)
    if len(payload) >= 2 and \
       ((payload[0] == '"' and payload[-1] == '"') or \
        (payload[0] == "'" and payload[-1] == "'")):
        payload = payload[1:-1]
    elif payload and payload[-1] in '"\'':
        payload = payload[:-1]

    return payload if payload else None

def _sanitize_alias(alias):
    """Sanitize extracted alias: strip quotes/backticks, stop at first invalid char/whitespace."""
    alias = alias.strip()
    # Strip wrapping quotes/backticks iteratively
    for _ in range(3):
        if len(alias) >= 2 and alias[0] == alias[-1] and alias[0] in ['"', "'", '`']:
            alias = alias[1:-1].strip()
        elif alias and alias[-1] in ['"', "'", '`']:
            alias = alias[:-1].strip()
        else:
            break
    # Stop at first whitespace or invalid char
    alias = re.split(r'[^A-Za-z0-9\-_]', alias)[0]
    return alias

def extract_resume_alias(command):
    """Extract resume alias safely.
    Priority:
    1) HCOM_SEND payload that starts with RESUME:alias
    2) Bare HCOM_RESUME:alias (only when not embedded in HCOM_SEND payload)
    """
    # 1) Prefer explicit HCOM_SEND payload
    payload = extract_hcom_command(command)
    if payload:
        cand = payload.strip()
        if cand.startswith('RESUME:'):
            alias_raw = cand.split(':', 1)[1].strip()
            alias = _sanitize_alias(alias_raw)
            return alias or None
        # If payload contains text like "HCOM_RESUME:alias" but not at start,
        # ignore to prevent alias hijack from normal messages

    # 2) Fallback: bare HCOM_RESUME when not using HCOM_SEND
    alias_raw = extract_hcom_command(command, 'HCOM_RESUME')
    if alias_raw:
        alias = _sanitize_alias(alias_raw)
        return alias or None
    return None

def compute_decision_for_visibility(transcript_path):
    """Compute hook decision based on pending tools to prevent API 400 errors."""
    pending_tools = get_pending_tools(transcript_path)
    decision = None if pending_tools > 0 else HOOK_DECISION_BLOCK

    return decision

def emit_resume_feedback(status, instance_name, transcript_path):
    """Emit formatted resume feedback with appropriate visibility."""
    # Build formatted feedback based on success/failure
    if status.startswith("[SUCCESS]"):
        reason = f"[{status}]{HCOM_FORMAT_INSTRUCTIONS}"
    else:
        reason = f"[⚠️ {status} - your alias is: {instance_name}]{HCOM_FORMAT_INSTRUCTIONS}"

    # Compute decision based on pending tools
    decision = compute_decision_for_visibility(transcript_path)

    # Emit response
    emit_hook_response(reason, decision=decision)

def handle_pretooluse(hook_data):
    """Handle PreToolUse hook - auto-approve HCOM_SEND commands when safe"""
    # Check if this is an HCOM_SEND command that needs auto-approval
    tool_name = hook_data.get('tool_name', '')
    if tool_name == 'Bash':
        command = hook_data.get('tool_input', {}).get('command', '')
        if 'HCOM_SEND:' in command or extract_resume_alias(command):
            # Check if other tools are pending - prevent API 400 errors
            transcript_path = hook_data.get('transcript_path', '')
            # Subtract 1 because the current tool is already in transcript but not actually pending
            pending_count = max(0, get_pending_tools(transcript_path) - 1)

            if pending_count > 0:
                # Deny execution to prevent injecting content between tool_use/tool_result
                output = {
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "permissionDecision": "deny",
                        "permissionDecisionReason": f"Waiting - {pending_count} tool(s) still executing. Try again in a moment."
                    }
                }
            else:
                # Safe to proceed
                output = {
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "permissionDecision": "allow",
                        "permissionDecisionReason": "HCOM_SEND command auto-approved"
                    }
                }
            print(json.dumps(output, ensure_ascii=False))
            sys.exit(EXIT_SUCCESS)

def handle_posttooluse(hook_data, instance_name, updates):
    """Handle PostToolUse hook - extract and deliver messages"""
    updates['last_tool'] = int(time.time())
    updates['last_tool_name'] = hook_data.get('tool_name', 'unknown')
    update_instance_position(instance_name, updates)

    # Check for HCOM_SEND in Bash commands
    sent_reason = None
    if hook_data.get('tool_name') == 'Bash':
        command = hook_data.get('tool_input', {}).get('command', '')

        # Check for RESUME command first (safe extraction)
        alias = extract_resume_alias(command)
        if alias:
            status = merge_instance_immediately(instance_name, alias)

            # If names match, find and merge any duplicate with same PID
            if not status and instance_name == alias:
                instances_dir = hcom_path(INSTANCES_DIR)
                parent_pid = os.getppid()
                if instances_dir.exists():
                    for instance_file in instances_dir.glob("*.json"):
                        if instance_file.stem != instance_name:
                            try:
                                data = load_instance_position(instance_file.stem)
                                if data.get('pid') == parent_pid:
                                    # Found duplicate - merge it
                                    status = merge_instance_immediately(instance_file.stem, instance_name)
                                    if status:
                                        status = f"[SUCCESS] ✓ Merged duplicate: {instance_file.stem} → {instance_name}"
                                    break
                            except:
                                continue

                if not status:
                    status = f"[SUCCESS] ✓ Already using alias {alias}"
            elif not status:
                status = f"[WARNING] ⚠️ Merge failed: {instance_name} → {alias}"

            if status:
                transcript_path = hook_data.get('transcript_path', '')
                emit_resume_feedback(status, instance_name, transcript_path)
            return  # Don't process RESUME as regular message

        # Normal message handling
        message = extract_hcom_command(command)  # defaults to HCOM_SEND
        if message:
            error = validate_message(message)
            if error:
                emit_hook_response(f"❌ {error}")
            send_message(instance_name, message)
            sent_reason = "[✓ Sent]"

    # Check for pending tools in transcript
    transcript_path = hook_data.get('transcript_path', '')
    pending_count = get_pending_tools(transcript_path)

    # Build response if needed
    response_reason = None

    # Only deliver messages when all tools are complete (pending_count == 0)
    if pending_count == 0:
        messages = get_new_messages(instance_name)
        if messages:
            messages = messages[:get_config_value('max_messages_per_delivery', 50)]
            reason = format_hook_messages(messages, instance_name)
            response_reason = f"{sent_reason} | {reason}" if sent_reason else reason
        elif sent_reason:
            response_reason = sent_reason
    elif sent_reason:
        # Tools still pending - acknowledge HCOM_SEND without disrupting tool batching
        response_reason = sent_reason

    # Emit response with formatting if we have anything to say
    if response_reason:
        response_reason += HCOM_FORMAT_INSTRUCTIONS
        # CRITICAL: decision=None when tools are pending to prevent API 400 errors
        decision = compute_decision_for_visibility(transcript_path)
        emit_hook_response(response_reason, decision=decision)

def handle_stop(instance_name, updates):
    """Handle Stop hook - poll for messages"""
    updates['last_stop'] = time.time()
    timeout = get_config_value('wait_timeout', 1800)
    updates['wait_timeout'] = timeout

    # Try to update position, but continue on Windows file locking errors
    try:
        update_instance_position(instance_name, updates)
    except Exception as e:
        # Silently handle initial file locking error and continue
        pass

    parent_pid = os.getppid()
    start_time = time.time()

    try:
        loop_count = 0
        while time.time() - start_time < timeout:
            loop_count += 1
            current_time = time.time()

            # Unix/Mac: Check if orphaned (reparented to PID 1)
            if not IS_WINDOWS and os.getppid() == 1:
                sys.exit(EXIT_SUCCESS)

            # All platforms: Check if parent is alive
            parent_alive = is_parent_alive(parent_pid)

            if not parent_alive:
                sys.exit(EXIT_SUCCESS)

            # Check for pending tools before delivering messages
            transcript_path = updates.get('transcript_path', '')
            pending_count = get_pending_tools(transcript_path)

            # Only deliver messages when no tools are pending
            if pending_count == 0:
                messages = get_new_messages(instance_name)
                if messages:
                    messages_to_show = messages[:get_config_value('max_messages_per_delivery', 50)]
                    reason = format_hook_messages(messages_to_show, instance_name)
                    emit_hook_response(reason)  # Normal visible delivery

            # Update position to keep instance marked as alive
            stop_update_time = time.time()
            try:
                update_instance_position(instance_name, {'last_stop': stop_update_time})
            except Exception as e:
                # Silently handle file locking exceptions on Windows and continue polling
                pass

            time.sleep(STOP_HOOK_POLL_INTERVAL)

    except Exception as e:
        # Exit with code 0 on unexpected exceptions (fail safe)
        sys.exit(EXIT_SUCCESS)

def handle_notify(hook_data, instance_name, updates):
    """Handle Notification hook - track permission requests"""
    updates['last_permission_request'] = int(time.time())
    updates['notification_message'] = hook_data.get('message', '')
    update_instance_position(instance_name, updates)

def handle_sessionstart(hook_data, instance_name, updates):
    """Handle SessionStart hook - deliver welcome/resume message"""
    source = hook_data.get('source', 'startup')

    # Reset alias_announced flag so alias shows again on resume/clear/compact
    updates['alias_announced'] = False

    # Always show base help text
    help_text = "[Welcome! HCOM chat active. Send messages: echo 'HCOM_SEND:your message']"

    # Add first use text only on startup
    if source == 'startup':
        first_use_text = get_config_value('first_use_text', '')
        if first_use_text:
            help_text += f" [{first_use_text}]"
    elif source == 'resume':
        if not os.environ.get('HCOM_RESUME_SESSION_ID'):
            # Implicit resume - prompt for alias recovery
            help_text += f" [⚠️ Resume detected - temp: {instance_name}. If you had a previous HCOM alias, run: echo \"HCOM_RESUME:your_alias\"]"
        else:
            help_text += " [Resuming session - you should have the same hcom alias as before]"

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

    # Update instance position
    update_instance_position(instance_name, updates)

def handle_hook(hook_type):
    """Unified hook handler for all HCOM hooks"""
    if os.environ.get(HCOM_ACTIVE_ENV) != HCOM_ACTIVE_VALUE:
        sys.exit(EXIT_SUCCESS)

    try:
        hook_data = json.load(sys.stdin)

        # Route to specific handler with only needed parameters
        if hook_type == 'pre':
            # PreToolUse only needs hook_data
            handle_pretooluse(hook_data)
        else:
            # Other hooks need context initialization
            instance_name, updates, _ = init_hook_context(hook_data)

            if hook_type == 'post':
                handle_posttooluse(hook_data, instance_name, updates)
            elif hook_type == 'stop':
                # Stop hook doesn't use hook_data
                handle_stop(instance_name, updates)
            elif hook_type == 'notify':
                handle_notify(hook_data, instance_name, updates)
            elif hook_type == 'sessionstart':
                handle_sessionstart(hook_data, instance_name, updates)

    except Exception:
        pass

    sys.exit(EXIT_SUCCESS)


# ==================== Main Entry Point ====================

def main(argv=None):
    """Main command dispatcher"""
    if argv is None:
        argv = sys.argv
    
    if len(argv) < 2:
        return cmd_help()
    
    cmd = argv[1]
    
    # Main commands
    if cmd == 'help' or cmd == '--help':
        return cmd_help()
    elif cmd == 'open':
        return cmd_open(*argv[2:])
    elif cmd == 'watch':
        return cmd_watch(*argv[2:])
    elif cmd == 'clear':
        return cmd_clear()
    elif cmd == 'cleanup':
        return cmd_cleanup(*argv[2:])
    elif cmd == 'send':
        if len(argv) < 3:
            print(format_error("Message required"), file=sys.stderr)
            return 1
        return cmd_send(argv[2])
    elif cmd == 'kill':
        return cmd_kill(*argv[2:])
    
    # Hook commands
    elif cmd in ['post', 'stop', 'notify', 'pre', 'sessionstart']:
        handle_hook(cmd)
        return 0
    
    # Unknown command
    else:
        print(format_error(f"Unknown command: {cmd}", "Run 'hcom help' for available commands"), file=sys.stderr)
        return 1

if __name__ == '__main__':
    sys.exit(main())
