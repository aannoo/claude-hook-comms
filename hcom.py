#!/usr/bin/env python3
"""
hcom - Claude Hook Comms
Lightweight CLI tool for real-time communication between Claude Code subagents using hooks
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

HCOM_ACTIVE_ENV = 'HCOM_ACTIVE'
HCOM_ACTIVE_VALUE = '1'

EXIT_SUCCESS = 0
EXIT_ERROR = 1
EXIT_BLOCK = 2

HOOK_DECISION_BLOCK = 'block'

MENTION_PATTERN = re.compile(r'(?<![a-zA-Z0-9._-])@(\w+)')
TIMESTAMP_SPLIT_PATTERN = re.compile(r'\n(?=\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+\|)')

RESET = "\033[0m"
DIM = "\033[2m"
BOLD = "\033[1m"
FG_BLUE = "\033[34m"
FG_GREEN = "\033[32m"
FG_CYAN = "\033[36m"
FG_RED = "\033[31m"
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

# ==================== Configuration ====================

DEFAULT_CONFIG = {
    "terminal_command": None,
    "terminal_mode": "new_window",
    "initial_prompt": "Say hi in chat",
    "sender_name": "bigboss",
    "sender_emoji": "🐳",
    "cli_hints": "",
    "wait_timeout": 1800,
    "max_message_size": 4096,
    "max_messages_per_delivery": 50,
    "first_use_text": "Essential, concise messages only, say hi in hcom chat now",
    "instance_hints": "",
    "env_overrides": {}
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
    'initial_prompt': 'HCOM_INITIAL_PROMPT'
}

# ==================== File System Utilities ====================

def get_hcom_dir():
    """Get the hcom directory in user's home"""
    return Path.home() / ".hcom"

def ensure_hcom_dir():
    """Create the hcom directory if it doesn't exist"""
    hcom_dir = get_hcom_dir()
    hcom_dir.mkdir(exist_ok=True)
    return hcom_dir

def atomic_write(filepath, content):
    """Write content to file atomically to prevent corruption"""
    filepath = Path(filepath)
    with tempfile.NamedTemporaryFile(mode='w', delete=False, dir=filepath.parent, suffix='.tmp') as tmp:
        tmp.write(content)
        tmp.flush()
        os.fsync(tmp.fileno())
    
    os.replace(tmp.name, filepath)

def get_instance_file(instance_name):
    """Get path to instance's position file"""
    return get_hcom_dir() / "instances" / f"{instance_name}.json"

def load_instance_position(instance_name):
    """Load position data for a single instance"""
    instance_file = get_instance_file(instance_name)
    if instance_file.exists():
        try:
            with open(instance_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            pass
    return {}

def save_instance_position(instance_name, data):
    """Save position data for a single instance - no locking needed"""
    instance_file = get_instance_file(instance_name)
    instance_file.parent.mkdir(exist_ok=True)
    atomic_write(instance_file, json.dumps(data, indent=2))

def load_all_positions():
    """Load positions from all instance files"""
    instances_dir = get_hcom_dir() / "instances"
    if not instances_dir.exists():
        return {}
    
    positions = {}
    for instance_file in instances_dir.glob("*.json"):
        try:
            instance_name = instance_file.stem
            with open(instance_file, 'r') as f:
                positions[instance_name] = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            continue
    return positions

def clear_all_positions():
    """Clear all instance position files"""
    instances_dir = get_hcom_dir() / "instances"
    if instances_dir.exists():
        shutil.rmtree(instances_dir)
    instances_dir.mkdir(exist_ok=True)

# ==================== Configuration System ====================

def get_cached_config():
    """Get cached configuration, loading if needed"""
    global _config
    if _config is None:
        _config = _load_config_from_file()
    return _config

def _load_config_from_file():
    """Actually load configuration from ~/.hcom/config.json"""
    ensure_hcom_dir()
    config_path = get_hcom_dir() / 'config.json'
    
    config = DEFAULT_CONFIG.copy()
    config['env_overrides'] = DEFAULT_CONFIG['env_overrides'].copy()
    
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                
                for key, value in user_config.items():
                    if key == 'env_overrides':
                        config['env_overrides'].update(value)
                    else:
                        config[key] = value
                    
        except (json.JSONDecodeError, UnicodeDecodeError, PermissionError):
            print(format_warning("Cannot read config file, using defaults"), file=sys.stderr)
    else:
        atomic_write(config_path, json.dumps(DEFAULT_CONFIG, indent=2))
    
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
    
    if ' ' in python_path or ' ' in script_path:
        # Paths with spaces: use conditional check
        escaped_python = shlex.quote(python_path)
        escaped_script = shlex.quote(script_path)
        return f'[ "${{HCOM_ACTIVE}}" = "1" ] && {escaped_python} {escaped_script} || true', {}
    else:
        # Clean paths: use environment variable
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
    
    max_size = get_config_value('max_message_size', 4096)
    if len(message) > max_size:
        return format_error(f"Message too large (max {max_size} chars)")
    
    return None

def require_args(min_count, usage_msg, extra_msg=""):
    """Check argument count and exit with usage if insufficient"""
    if len(sys.argv) < min_count:
        print(f"Usage: {usage_msg}")
        if extra_msg:
            print(extra_msg)
        sys.exit(1)

def load_positions(pos_file=None):
    """Load all positions - redirects to load_all_positions()"""
    # pos_file parameter kept for compatibility but ignored
    return load_all_positions()

def send_message(from_instance, message):
    """Send a message to the log"""
    try:
        ensure_hcom_dir()
        log_file = get_hcom_dir() / "hcom.log"
        pos_file = get_hcom_dir() / "hcom.json"
        
        escaped_message = message.replace('|', '\\|')
        escaped_from = from_instance.replace('|', '\\|')
        
        timestamp = datetime.now().isoformat()
        line = f"{timestamp}|{escaped_from}|{escaped_message}\n"
        
        with open(log_file, 'a') as f:
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
    for base_path in [Path('.'), Path.home()]:
        agent_path = base_path / '.claude/agents' / f'{name}.md'
        if agent_path.exists():
            content = agent_path.read_text()
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
        lines = content.split('\n')
        for i, line in enumerate(lines[1:], 1):
            if line.strip() == '---':
                return '\n'.join(lines[i+1:]).strip()
    return content

def get_display_name(transcript_path, prefix=None):
    """Get display name for instance"""
    syls = ['ka', 'ko', 'ma', 'mo', 'na', 'no', 'ra', 'ro', 'sa', 'so', 'ta', 'to', 'va', 'vo', 'za', 'zo', 'be', 'de', 'fe', 'ge', 'le', 'me', 'ne', 're', 'se', 'te', 've', 'we', 'hi']
    dir_name = Path.cwd().name
    dir_chars = (dir_name + 'xx')[:2].lower()  # Pad short names to ensure 2 chars
    
    conversation_uuid = get_conversation_uuid(transcript_path)
    
    if conversation_uuid:
        hash_val = sum(ord(c) for c in conversation_uuid)
        uuid_char = conversation_uuid[0]
        base_name = f"{dir_chars}{syls[hash_val % len(syls)]}{uuid_char}"
    else:
        pid_suffix = os.getppid() % 10000  # 4 digits max
        base_name = f"{dir_chars}{pid_suffix}claude"
    
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
    hcom_patterns = [
        r'\$\{?HCOM',                                # Environment variable (with or without braces)
        r'\bHCOM_ACTIVE.*hcom\.py',                 # Conditional with HCOM_ACTIVE check
        r'\bhcom\s+(post|stop|notify)\b',           # Direct hcom command
        r'\buvx\s+hcom\s+(post|stop|notify)\b',     # uvx hcom command
        r'hcom\.py["\']?\s+(post|stop|notify)\b',   # hcom.py with optional quote
        r'["\'][^"\']*hcom\.py["\']?\s+(post|stop|notify)\b(?=\s|$)',  # Quoted path with hcom.py (more precise)
        r'sh\s+-c.*hcom',                           # Shell wrapper with hcom
    ]
    compiled_patterns = [re.compile(pattern) for pattern in hcom_patterns]
    
    for event in ['PreToolUse', 'PostToolUse', 'Stop', 'Notification']:
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
    """Build environment variable string for different shells"""
    if format_type == "bash_export":
        # Properly escape values for bash
        return ' '.join(f'export {k}={shlex.quote(str(v))};' for k, v in env_vars.items())
    elif format_type == "powershell":
        # PowerShell environment variable syntax
        items = []
        for k, v in env_vars.items():
            escaped_value = str(v).replace('"', '`"')
            items.append(f'$env:{k}="{escaped_value}"')
        return ' ; '.join(items)
    else:
        return ' '.join(f'{k}={shlex.quote(str(v))}' for k, v in env_vars.items())

def format_error(message, suggestion=None):
    """Format error message consistently"""
    base = f"Error: {message}"
    if suggestion:
        base += f". {suggestion}"
    return base

def format_warning(message):
    """Format warning message consistently"""
    return f"Warning: {message}"

def build_claude_command(agent_content=None, claude_args=None, initial_prompt="Say hi in chat", model=None, tools=None):
    """Build Claude command with proper argument handling
    
    Returns tuple: (command_string, temp_file_path_or_none)
    For agent content, writes to temp file and uses cat to read it.
    """
    cmd_parts = ['claude']
    temp_file_path = None
    
    # Add model if specified and not already in claude_args
    if model:
        # Check if model already specified in args (more concise)
        has_model = claude_args and any(
            arg in ['--model', '-m'] or 
            arg.startswith(('--model=', '-m=')) 
            for arg in claude_args
        )
        if not has_model:
            cmd_parts.extend(['--model', model])
    
    # Add allowed tools if specified and not already in claude_args
    if tools:
        has_tools = claude_args and any(
            arg in ['--allowedTools', '--allowed-tools'] or
            arg.startswith(('--allowedTools=', '--allowed-tools='))
            for arg in claude_args
        )
        if not has_tools:
            cmd_parts.extend(['--allowedTools', tools])
    
    if claude_args:
        for arg in claude_args:
            cmd_parts.append(shlex.quote(arg))
    
    if agent_content:
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, 
                                              prefix='hcom_agent_', dir=tempfile.gettempdir())
        temp_file.write(agent_content)
        temp_file.close()
        temp_file_path = temp_file.name
        
        if claude_args and any(arg in claude_args for arg in ['-p', '--print']):
            flag = '--system-prompt'
        else:
            flag = '--append-system-prompt'
        
        cmd_parts.append(flag)
        if sys.platform == 'win32':
            # PowerShell handles paths differently, quote with single quotes
            escaped_path = temp_file_path.replace("'", "''")
            cmd_parts.append(f"\"$(Get-Content '{escaped_path}' -Raw)\"")
        else:
            cmd_parts.append(f'"$(cat {shlex.quote(temp_file_path)})"')
    
    if claude_args or agent_content:
        cmd_parts.append('--')
    
    # Quote initial prompt normally
    cmd_parts.append(shlex.quote(initial_prompt))
    
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
    elif platform_type == 'powershell':
        # PowerShell escaping - use backticks for special chars
        # Quote paths with spaces for PowerShell
        escaped = text.replace('`', '``').replace('"', '`"').replace('$', '`$')
        if ' ' in text and not (text.startswith('"') and text.endswith('"')):
            return f'"{escaped}"'
        return escaped
    else:  # POSIX/bash
        return shlex.quote(text)

def safe_command_substitution(template, **substitutions):
    """Safely substitute values into command templates with automatic quoting"""
    result = template
    for key, value in substitutions.items():
        placeholder = f'{{{key}}}'
        if placeholder in result:
            # Auto-quote substitutions unless already quoted
            if key == 'env':
                # env_str is already properly quoted
                quoted_value = str(value)
            else:
                quoted_value = shlex.quote(str(value))
            result = result.replace(placeholder, quoted_value)
    return result

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
        logs_dir = get_hcom_dir() / 'logs'
        logs_dir.mkdir(exist_ok=True)
        log_file = logs_dir / env['HCOM_BACKGROUND']
        
        # Launch detached process
        try:
            with open(log_file, 'w') as log_handle:
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
            print(f"Error: Failed to launch background instance: {e}", file=sys.stderr)
            return None
        
        # Check for immediate failures
        time.sleep(0.2)
        if process.poll() is not None:
            # Process already exited
            with open(log_file, 'r') as f:
                error_output = f.read()[:1000]
            print(f"Error: Background instance failed immediately", file=sys.stderr)
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
        result = subprocess.run(command_str, shell=True, env=env_vars, cwd=cwd)
        return result.returncode == 0
    
    system = platform.system()
    
    custom_cmd = get_config_value('terminal_command')
    if custom_cmd and custom_cmd != 'None' and custom_cmd != 'null':
        # Replace placeholders
        env_str = build_env_string(env)
        working_dir = cwd or os.getcwd()
        
        final_cmd = safe_command_substitution(
            custom_cmd,
            cmd=command_str,
            env=env_str,  # Already quoted
            cwd=working_dir
        )
        
        result = subprocess.run(final_cmd, shell=True, capture_output=True)
        if result.returncode != 0:
            raise subprocess.CalledProcessError(result.returncode, final_cmd, result.stderr)
        return True
    
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
        
        raise Exception(format_error("No supported terminal emulator found", "Install gnome-terminal, konsole, or xterm"))
        
    elif system == 'Windows':
        # Windows Terminal with PowerShell
        env_setup = build_env_string(env, "powershell")
        # Include cd command if cwd is specified
        if cwd:
            full_cmd = f'cd "{cwd}" ; {env_setup} ; {command_str}'
        else:
            full_cmd = f'{env_setup} ; {command_str}'
        
        try:
            # Try Windows Terminal with PowerShell
            subprocess.run(['wt', 'powershell', '-NoExit', '-Command', full_cmd])
        except FileNotFoundError:
            # Fallback to PowerShell directly
            subprocess.run(['powershell', '-NoExit', '-Command', full_cmd])
        return True
    
    else:
        raise Exception(format_error(f"Unsupported platform: {system}", "Supported platforms: macOS, Linux, Windows"))

def setup_hooks():
    """Set up Claude hooks in current directory"""
    claude_dir = Path.cwd() / '.claude'
    claude_dir.mkdir(exist_ok=True)
    
    settings_path = claude_dir / 'settings.local.json'
    settings = {}
    
    if settings_path.exists():
        try:
            with open(settings_path, 'r') as f:
                settings = json.load(f)
        except (json.JSONDecodeError, PermissionError) as e:
            raise Exception(format_error(f"Cannot read settings: {e}"))
    
    if 'hooks' not in settings:
        settings['hooks'] = {}
    if 'permissions' not in settings:
        settings['permissions'] = {}
    if 'allow' not in settings['permissions']:
        settings['permissions']['allow'] = []
    
    _remove_hcom_hooks_from_settings(settings)
    
    if 'hooks' not in settings:
        settings['hooks'] = {}
        
    # Get the hook command template
    hook_cmd_base, _ = get_hook_command()
    
    # Add PreToolUse hook for auto-approving HCOM_SEND commands
    if 'PreToolUse' not in settings['hooks']:
        settings['hooks']['PreToolUse'] = []
    
    settings['hooks']['PreToolUse'].append({
        'matcher': 'Bash',
        'hooks': [{
            'type': 'command',
            'command': f'{hook_cmd_base} pre'
        }]
    })
    
    # Add PostToolUse hook
    if 'PostToolUse' not in settings['hooks']:
        settings['hooks']['PostToolUse'] = []
    
    settings['hooks']['PostToolUse'].append({
        'matcher': '.*',
        'hooks': [{
            'type': 'command',
            'command': f'{hook_cmd_base} post'
        }]
    })
    
    # Add Stop hook
    if 'Stop' not in settings['hooks']:
        settings['hooks']['Stop'] = []
    
    wait_timeout = get_config_value('wait_timeout', 1800)
    
    settings['hooks']['Stop'].append({
        'matcher': '',
        'hooks': [{
            'type': 'command',
            'command': f'{hook_cmd_base} stop',
            'timeout': wait_timeout
        }]
    })
    
    # Add Notification hook
    if 'Notification' not in settings['hooks']:
        settings['hooks']['Notification'] = []
    
    settings['hooks']['Notification'].append({
        'matcher': '',
        'hooks': [{
            'type': 'command',
            'command': f'{hook_cmd_base} notify'
        }]
    })
    
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
        with open(settings_path, 'r') as f:
            settings = json.load(f)
        
        # Check all 4 hook types exist with HCOM commands
        hooks = settings.get('hooks', {})
        for hook_type in ['PreToolUse', 'PostToolUse', 'Stop', 'Notification']:
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

def get_conversation_uuid(transcript_path):
    """Get conversation UUID from transcript
    
    For resumed sessions, the first line may be a summary with a different leafUuid.
    We need to find the first user entry which contains the stable conversation UUID.
    """
    try:
        if not transcript_path or not os.path.exists(transcript_path):
            return None
        
        # First, try to find the UUID from the first user entry
        with open(transcript_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    # Look for first user entry with a UUID - this is the stable identifier
                    if entry.get('type') == 'user' and entry.get('uuid'):
                        return entry.get('uuid')
                except json.JSONDecodeError:
                    continue
        
        # Fallback: If no user entry found, try the first line (original behavior)
        with open(transcript_path, 'r') as f:
            first_line = f.readline().strip()
            if first_line:
                entry = json.loads(first_line)
                # Try both 'uuid' and 'leafUuid' fields
                return entry.get('uuid') or entry.get('leafUuid')
    except Exception:
        pass
    return None

def is_parent_alive(parent_pid=None):
    """Check if parent process is alive"""
    if parent_pid is None:
        parent_pid = os.getppid()
    
    if IS_WINDOWS:
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.OpenProcess(0x0400, False, parent_pid)
            if handle == 0:
                return False
            kernel32.CloseHandle(handle)
            return True
        except Exception:
            return True
    else:
        try:
            os.kill(parent_pid, 0)
            return True
        except ProcessLookupError:
            return False
        except Exception:
            return True

def parse_log_messages(log_file, start_pos=0):
    """Parse messages from log file"""
    log_file = Path(log_file)
    if not log_file.exists():
        return []
    
    messages = []
    with open(log_file, 'r') as f:
        f.seek(start_pos)
        content = f.read()
        
        if not content.strip():
            return []
            
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
    
    return messages

def get_new_messages(instance_name):
    """Get new messages for instance with @-mention filtering"""
    ensure_hcom_dir()
    log_file = get_hcom_dir() / "hcom.log"
    
    if not log_file.exists():
        return []
    
    positions = load_all_positions()
    
    # Get last position for this instance
    last_pos = 0
    if instance_name in positions:
        pos_data = positions.get(instance_name, {})
        last_pos = pos_data.get('pos', 0) if isinstance(pos_data, dict) else pos_data
    
    all_messages = parse_log_messages(log_file, last_pos)
    
    # Filter messages:
    # 1. Exclude own messages
    # 2. Apply @-mention filtering
    all_instance_names = list(positions.keys())
    messages = []
    for msg in all_messages:
        if msg['from'] != instance_name:
            if should_deliver_message(msg, instance_name, all_instance_names):
                messages.append(msg)
    
    # Update position to end of file
    with open(log_file, 'r') as f:
        f.seek(0, 2)  # Seek to end
        new_pos = f.tell()
    
    # Update position directly
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
    try:
        if not transcript_path or not os.path.exists(transcript_path):
            return "inactive", "", "", 0
            
        with open(transcript_path, 'r') as f:
            lines = f.readlines()[-5:]
        
        for line in reversed(lines):
            entry = json.loads(line)
            timestamp = datetime.fromisoformat(entry['timestamp']).timestamp()
            age = int(time.time() - timestamp)
            
            if entry['type'] == 'system':
                content = entry.get('content', '')
                if 'Running' in content:
                    tool_name = content.split('Running ')[1].split('[')[0].strip()
                    return "executing", f"({format_age(age)})", tool_name, timestamp
            
            elif entry['type'] == 'assistant':
                content = entry.get('content', [])
                if any('tool_use' in str(item) for item in content):
                    return "executing", f"({format_age(age)})", "tool", timestamp
                else:
                    return "responding", f"({format_age(age)})", "", timestamp
            
            elif entry['type'] == 'user':
                return "thinking", f"({format_age(age)})", "", timestamp
        
        return "inactive", "", "", 0
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
        try:
            os.kill(int(pid), 0)  # Check process existence
        except (ProcessLookupError, TypeError, ValueError):
            return "inactive", ""
    
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
    
    # Add background indicator
    status_suffix = " (bg)" if pos_data.get('background') else ""
    return most_recent_status, f"({format_age(age)}){status_suffix}"

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
    
    log_file = get_hcom_dir() / 'hcom.log'
    if log_file.exists():
        messages = parse_log_messages(log_file)
        show_recent_messages(messages, limit, truncate=True)

def show_instances_status():
    """Show status of all instances"""
    positions = load_all_positions()
    if not positions:
        print(f"   {DIM}No Claude instances connected{RESET}")
        return
        
    print("Instances in hcom:")
    for instance_name, pos_data in positions.items():
        status_type, age = get_instance_status(pos_data)
        status_block = get_status_block(status_type)
        directory = pos_data.get("directory", "unknown")
        print(f"  {BOLD}{instance_name}{RESET}  {status_block} {DIM}{status_type} {age}{RESET}     {directory}")

def show_instances_by_directory():
    """Show instances organized by their working directories"""
    positions = load_all_positions()
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
                
                sid = pos_data.get("session_id", "")
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
        print(f"{DIM} LOG FILE: {get_hcom_dir() / 'hcom.log'}{RESET}")
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
    
    for _, pos_data in positions.items():
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
            parts.append(f"{text_color}{BOLD}{color} {count} {symbol} {RESET}")
    
    if parts:
        return "".join(parts)
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

def initialize_instance_in_position_file(instance_name, conversation_uuid=None):
    """Initialize an instance in the position file with all required fields"""
    instance_file = get_instance_file(instance_name)
    if not instance_file.exists():
        save_instance_position(instance_name, {
            "pos": 0,
            "directory": str(Path.cwd()),
            "conversation_uuid": conversation_uuid or "unknown",
            "last_tool": 0,
            "last_tool_name": "unknown",
            "last_stop": 0,
            "last_permission_request": 0,
            "transcript_path": "",
            "session_id": "",
            "help_shown": False,
            "notification_message": ""
        })

def migrate_instance_name_if_needed(instance_name, conversation_uuid, transcript_path):
    """Migrate instance name from fallback to UUID-based if needed"""
    if instance_name.endswith("claude") and conversation_uuid:
        new_instance = get_display_name(transcript_path)
        if new_instance != instance_name and not new_instance.endswith("claude"):
            # Migration of data only happens if old name exists
            old_file = get_instance_file(instance_name)
            if old_file.exists():
                data = load_instance_position(instance_name)
                data["conversation_uuid"] = conversation_uuid
                save_instance_position(new_instance, data)
                old_file.unlink()
            return new_instance
    return instance_name

def update_instance_position(instance_name, update_fields):
    """Update instance position - direct file write, no locking needed"""
    data = load_instance_position(instance_name)
    data.update(update_fields)
    save_instance_position(instance_name, data)

def get_first_use_text(instance_name):
    """Get first-use help text if not shown yet
    
    Returns:
        str: Welcome text if not shown before, empty string otherwise
    """
    positions = load_all_positions()
    
    instance_data = positions.get(instance_name, {})
    if not instance_data.get('help_shown', False):
        # Mark help as shown
        update_instance_position(instance_name, {'help_shown': True})
        
        # Get values using unified config system
        first_use_text = get_config_value('first_use_text', '')
        instance_hints = get_config_value('instance_hints', '')
        
        help_text = f"[Welcome! hcom chat active. Your alias: {instance_name}. " \
                   f"Send messages: echo 'HCOM_SEND:your message']"
        if first_use_text:
            help_text += f" [{first_use_text}]"
        if instance_hints:
            help_text += f" [{instance_hints}]"
        
        return help_text
    return ""

def check_and_show_first_use_help(instance_name):
    """Check and show first-use help if needed (for Stop hook)"""
    help_text = get_first_use_text(instance_name)
    if help_text:
        output = {"decision": HOOK_DECISION_BLOCK, "reason": help_text}
        print(json.dumps(output, ensure_ascii=False), file=sys.stderr)
        sys.exit(EXIT_BLOCK)

# ==================== Command Functions ====================

def show_main_screen_header():
    """Show header for main screen"""
    sys.stdout.write("\033[2J\033[H")
    
    log_file = get_hcom_dir() / 'hcom.log'
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
    
    # Additional help for AI assistants when running in non-interactive mode
    if not sys.stdin.isatty():
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
Config file (persistent s): ~/.hcom/config.json

Key settings (full list in config.json):
  terminal_mode: "new_window" (default) | "same_terminal" | "show_commands"
  initial_prompt: "Say hi in chat", first_use_text: "Essential messages only..."
  instance_hints: "text", cli_hints: "text"  # Extra info for instances/CLI

Temporary environment overrides for any setting (all caps & append HCOM_):
HCOM_INSTANCE_HINTS="useful info" hcom open  # applied to all messages recieved by instance
export HCOM_CLI_HINTS="useful info" && hcom send 'hi'  # applied to all cli commands

EXPECT: hcom instance aliases are auto-generated (5-char format: "hova7"). Check actual aliases 
with 'hcom watch --status'. Instances respond automatically in shared chat.""")
        
        show_cli_hints(to_stderr=False)
    
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
        
        ensure_hcom_dir()
        log_file = get_hcom_dir() / 'hcom.log'
        instances_dir = get_hcom_dir() / 'instances'
        
        if not log_file.exists():
            log_file.touch()
        if not instances_dir.exists():
            instances_dir.mkdir(exist_ok=True)
        
        # Build environment variables for Claude instances
        base_env = build_claude_env()
        
        # Add prefix-specific hints if provided
        if prefix:
            hint = f"To respond to {prefix} group: echo 'HCOM_SEND:@{prefix} message'"
            base_env['HCOM_INSTANCE_HINTS'] = hint
            
            first_use = f"You're in the {prefix} group. Use {prefix} to message: echo HCOM_SEND:@{prefix} message."
            base_env['HCOM_FIRST_USE_TEXT'] = first_use
        
        launched = 0
        initial_prompt = get_config_value('initial_prompt', 'Say hi in chat')
        
        temp_files_to_cleanup = []
        
        for instance_type in instances:
            instance_env = base_env.copy()
            
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
                print(f"Error: Failed to launch terminal: {e}", file=sys.stderr)
        
        # Clean up temp files after a delay (let terminals read them first)
        if temp_files_to_cleanup:
            def cleanup_temp_files():
                time.sleep(5)  # Give terminals time to read the files
                for temp_file in temp_files_to_cleanup:
                    try:
                        os.unlink(temp_file)
                    except:
                        pass
            
            cleanup_thread = threading.Thread(target=cleanup_temp_files)
            cleanup_thread.daemon = True
            cleanup_thread.start()
                
        if launched == 0:
            print(format_error("No instances launched"), file=sys.stderr)
            return 1
            
        # Success message
        print(f"Launched {launched} Claude instance{'s' if launched != 1 else ''}")
        
        tips = [
            "Run 'hcom watch' to view/send in conversation dashboard",
        ]
        if prefix:
            tips.append(f"Send to {prefix} team: hcom send '@{prefix} message'")
        
        print("\n" + "\n".join(f"  • {tip}" for tip in tips))
        
        # Show cli_hints if configured (non-interactive mode)
        if not is_interactive():
            cli_hints = get_config_value('cli_hints', '')
            if cli_hints:
                print(f"\n{cli_hints}") 
        
        return 0
        
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

def cmd_watch(*args):
    """View conversation dashboard"""
    log_file = get_hcom_dir() / 'hcom.log'
    instances_dir = get_hcom_dir() / 'instances'
    
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
                    if log_file.exists() and log_file.stat().st_size > last_pos:
                        # Capture new position BEFORE parsing (atomic)
                        new_pos = log_file.stat().st_size
                        new_messages = parse_log_messages(log_file, last_pos)
                        if new_messages:
                            for msg in new_messages:
                                print(f"[{msg['timestamp']}] {msg['from']}: {msg['message']}")
                            last_pos = new_pos  # Update only after successful processing
                            return 0  # Success - got new messages
                        last_pos = new_pos  # Update even if no messages (file grew but no complete messages yet)
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
            # Status information to stdout for parsing
            print("HCOM STATUS")
            print("INSTANCES:")
            show_instances_status()
            print("\nRECENT ACTIVITY:")
            show_recent_activity_alt_screen()
            print(f"\nLOG FILE: {log_file}")
            
            show_cli_hints()
        else:
            # No TTY - show automation usage to stderr
            print("Automation usage:", file=sys.stderr)
            print("  hcom send 'message'    Send message to group", file=sys.stderr)
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
            if now - last_status_update > 2.0:
                current_status = get_status_summary()
                
                # Only redraw if status text changed
                if current_status != last_status:
                    update_status(f"{current_status}{status_suffix}")
                    last_status = current_status
                
                last_status_update = now
            
            if log_file.exists() and log_file.stat().st_size > last_pos:
                new_messages = parse_log_messages(log_file, last_pos)
                # Use the last known status for consistency
                status_line_text = f"{last_status}{status_suffix}"
                for msg in new_messages:
                    log_line_with_status(format_message_line(msg), status_line_text)
                last_pos = log_file.stat().st_size
            
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
    ensure_hcom_dir()
    log_file = get_hcom_dir() / 'hcom.log'
    instances_dir = get_hcom_dir() / 'instances'
    archive_folder = get_hcom_dir() / 'archive'
    
    # Check if hcom files exist
    if not log_file.exists() and not instances_dir.exists():
        print("No hcom conversation to clear")
        return 0
    
    # Generate archive timestamp
    timestamp = get_archive_timestamp()
    
    # Ensure archive folder exists
    archive_folder.mkdir(exist_ok=True)
    
    # Archive existing files if they have content
    archived = False
    
    try:
        # Create session archive folder if we have content to archive
        has_log = log_file.exists() and log_file.stat().st_size > 0
        has_instances = instances_dir.exists() and any(instances_dir.glob('*.json'))
        
        if has_log or has_instances:
            # Create session archive folder with timestamp
            session_archive = archive_folder / f'session-{timestamp}'
            session_archive.mkdir(exist_ok=True)
            
            # Archive log file
            if has_log:
                archive_log = session_archive / 'hcom.log'
                log_file.rename(archive_log)
                archived = True
            elif log_file.exists():
                log_file.unlink()
            
            # Archive instances
            if has_instances:
                archive_instances = session_archive / 'instances'
                archive_instances.mkdir(exist_ok=True)
                
                # Move json files only (background logs stay in logs folder)
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
        with open(settings_path, 'r') as f:
            settings = json.load(f)
        
        # Check if any hcom hooks exist
        hooks_found = False
        
        original_hook_count = sum(len(settings.get('hooks', {}).get(event, [])) 
                                  for event in ['PreToolUse', 'PostToolUse', 'Stop', 'Notification'])
        
        _remove_hcom_hooks_from_settings(settings)
        
        # Check if any were removed
        new_hook_count = sum(len(settings.get('hooks', {}).get(event, [])) 
                             for event in ['PreToolUse', 'PostToolUse', 'Stop', 'Notification'])
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
    import signal
    
    instance_name = args[0] if args and args[0] != '--all' else None
    positions = load_all_positions() if not instance_name else {instance_name: load_instance_position(instance_name)}
    
    # Filter to instances with PIDs (any instance that's running)
    targets = [(name, data) for name, data in positions.items() if data.get('pid')]
    
    if not targets:
        print(f"No running process found for {instance_name}" if instance_name else "No running instances found")
        return 1 if instance_name else 0
    
    killed_count = 0
    for target_name, target_data in targets:
        # Get status and type info before killing
        status, age = get_instance_status(target_data)
        instance_type = "background" if target_data.get('background') else "foreground"
        
        pid = int(target_data['pid'])
        try:
            # Send SIGTERM for graceful shutdown
            os.kill(pid, signal.SIGTERM)
            
            # Wait up to 2 seconds for process to terminate
            for _ in range(20):
                time.sleep(0.1)
                try:
                    os.kill(pid, 0)  # Check if process still exists
                except ProcessLookupError:
                    # Process terminated successfully
                    break
            else:
                # Process didn't die from SIGTERM, force kill
                try:
                    os.kill(pid, signal.SIGKILL)
                    time.sleep(0.1)  # Brief wait for SIGKILL to take effect
                except ProcessLookupError:
                    pass  # Already dead
            
            print(f"Killed {target_name} ({instance_type}, {status}{age}, PID {pid})")
            killed_count += 1
        except (ProcessLookupError, TypeError, ValueError) as e:
            print(f"Process {pid} already dead or invalid: {e}")
        
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
            print(format_warning(f"Could not read current instances: {e}"))
        
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
            
            # Check if settings file exists
            settings_path = Path(directory) / '.claude' / 'settings.local.json'
            if not settings_path.exists():
                print("  No Claude settings found")
                already_clean += 1
                continue
            
            exit_code, message = cleanup_directory_hooks(Path(directory))
            if exit_code == 0:
                if "No hcom hooks found" in message:
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
    log_file = get_hcom_dir() / 'hcom.log'
    instances_dir = get_hcom_dir() / 'instances'
    
    if not log_file.exists() and not instances_dir.exists():
        print(format_error("No conversation found", "Run 'hcom open' first"), file=sys.stderr)
        return 1
    
    # Validate message
    error = validate_message(message)
    if error:
        print(f"Error: {error}", file=sys.stderr)
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

def update_instance_with_pid(updates, existing_data):
    """Add PID to updates if not already set in existing data"""
    if not existing_data.get('pid'):
        updates['pid'] = os.getppid()  # Parent PID is the Claude process
    return updates

def format_hook_messages(messages, instance_name):
    """Format messages for hook feedback"""
    if len(messages) == 1:
        msg = messages[0]
        reason = f"[new message] {msg['from']} → {instance_name}: {msg['message']}"
    else:
        parts = [f"{msg['from']} → {instance_name}: {msg['message']}" for msg in messages]
        reason = f"[{len(messages)} new messages] | " + " | ".join(parts)
    
    # Only append instance_hints to messages (first_use_text is handled separately)
    instance_hints = get_config_value('instance_hints', '')
    if instance_hints:
        reason = f"{reason} | [{instance_hints}]"
    
    return reason

def handle_hook_post():
    """Handle PostToolUse hook"""
    # Check if active
    if os.environ.get(HCOM_ACTIVE_ENV) != HCOM_ACTIVE_VALUE:
        sys.exit(EXIT_SUCCESS)
    
    try:
        # Read JSON input
        hook_data = json.load(sys.stdin)
        transcript_path = hook_data.get('transcript_path', '')
        instance_name = get_display_name(transcript_path) if transcript_path else f"{Path.cwd().name[:2].lower()}claude"
        conversation_uuid = get_conversation_uuid(transcript_path)
        
        # Migrate instance name if needed (from fallback to UUID-based)
        instance_name = migrate_instance_name_if_needed(instance_name, conversation_uuid, transcript_path)
        
        initialize_instance_in_position_file(instance_name, conversation_uuid)
        
        # Get existing position data to check if already set
        existing_data = load_instance_position(instance_name)
        
        updates = {
                'last_tool': int(time.time()),
                'last_tool_name': hook_data.get('tool_name', 'unknown'),
                'session_id': hook_data.get('session_id', ''),
                'transcript_path': transcript_path,
                'conversation_uuid': conversation_uuid or 'unknown',
                'directory': str(Path.cwd()),
            }
        
        # Update PID if not already set
        update_instance_with_pid(updates, existing_data)
        
        # Check if this is a background instance and save log file
        bg_env = os.environ.get('HCOM_BACKGROUND')
        if bg_env:
            updates['background'] = True
            updates['background_log_file'] = str(get_hcom_dir() / 'logs' / bg_env)
        
        update_instance_position(instance_name, updates)
        
        # Check for first-use help (prepend to first activity)
        welcome_text = get_first_use_text(instance_name)
        welcome_prefix = f"{welcome_text} | " if welcome_text else ""
        
        # Check for HCOM_SEND in Bash commands  
        sent_reason = None
        if hook_data.get('tool_name') == 'Bash':
            command = hook_data.get('tool_input', {}).get('command', '')
            if 'HCOM_SEND:' in command:
                # Extract message after HCOM_SEND:
                parts = command.split('HCOM_SEND:', 1)
                if len(parts) > 1:
                    remainder = parts[1]
                    
                    # The message might be in the format:
                    # - message"        (from echo "HCOM_SEND:message")
                    # - message'        (from echo 'HCOM_SEND:message')
                    # - message         (from echo HCOM_SEND:message)
                    # - "message"       (from echo HCOM_SEND:"message")
                    
                    message = remainder.strip()
                    
                    # If it starts and ends with matching quotes, remove them
                    if len(message) >= 2 and \
                       ((message[0] == '"' and message[-1] == '"') or \
                        (message[0] == "'" and message[-1] == "'")):
                        message = message[1:-1]
                    # If it ends with a quote but doesn't start with one, 
                    # it's likely from echo "HCOM_SEND:message" format
                    elif message and message[-1] in '"\'':
                        message = message[:-1]
                    
                    if message:
                        error = validate_message(message)
                        if error:
                            output = {"reason": f"❌ {error}"}
                            print(json.dumps(output, ensure_ascii=False), file=sys.stderr)
                            sys.exit(EXIT_BLOCK)
                        
                        send_message(instance_name, message)
                        sent_reason = "[✓ Sent]"
        
        messages = get_new_messages(instance_name)
        
        # Limit messages to max_messages_per_delivery
        if messages:
            max_messages = get_config_value('max_messages_per_delivery', 50)
            messages = messages[:max_messages]
        
        if messages and sent_reason:
            # Both sent and received
            reason = f"{sent_reason} | {format_hook_messages(messages, instance_name)}"
            if welcome_prefix:
                reason = f"{welcome_prefix}{reason}"
            output = {"decision": HOOK_DECISION_BLOCK, "reason": reason}
            print(json.dumps(output, ensure_ascii=False), file=sys.stderr)
            sys.exit(EXIT_BLOCK)
        elif messages:
            # Just received
            reason = format_hook_messages(messages, instance_name)
            if welcome_prefix:
                reason = f"{welcome_prefix}{reason}"
            output = {"decision": HOOK_DECISION_BLOCK, "reason": reason}
            print(json.dumps(output, ensure_ascii=False), file=sys.stderr)
            sys.exit(EXIT_BLOCK)
        elif sent_reason:
            # Just sent
            reason = sent_reason
            if welcome_prefix:
                reason = f"{welcome_prefix}{reason}"
            output = {"reason": reason}
            print(json.dumps(output, ensure_ascii=False), file=sys.stderr)
            sys.exit(EXIT_BLOCK)
        elif welcome_prefix:
            # No activity but need to show welcome
            output = {"decision": HOOK_DECISION_BLOCK, "reason": welcome_prefix.rstrip(' | ')}
            print(json.dumps(output, ensure_ascii=False), file=sys.stderr)
            sys.exit(EXIT_BLOCK)
    
    except Exception:
        pass
    
    sys.exit(EXIT_SUCCESS)

def handle_hook_stop():
    """Handle Stop hook"""
    # Check if active
    if os.environ.get(HCOM_ACTIVE_ENV) != HCOM_ACTIVE_VALUE:
        sys.exit(EXIT_SUCCESS)
    
    try:
        # Read hook input
        hook_data = json.load(sys.stdin)
        transcript_path = hook_data.get('transcript_path', '')
        instance_name = get_display_name(transcript_path) if transcript_path else f"{Path.cwd().name[:2].lower()}claude"
        conversation_uuid = get_conversation_uuid(transcript_path)
        
        # Migrate instance name if needed (from fallback to UUID-based)
        instance_name = migrate_instance_name_if_needed(instance_name, conversation_uuid, transcript_path)
        
        # Initialize instance if needed
        initialize_instance_in_position_file(instance_name, conversation_uuid)
        
        # Get the timeout this hook will use
        timeout = get_config_value('wait_timeout', 1800)
        
        # Get existing position data to check if PID is already set
        existing_data = load_instance_position(instance_name)
        
        # Update instance as waiting and store the timeout
        updates = {
            'last_stop': time.time(),
            'wait_timeout': timeout,  # Store this instance's timeout
            'session_id': hook_data.get('session_id', ''),
            'transcript_path': transcript_path,
            'conversation_uuid': conversation_uuid or 'unknown',
            'directory': str(Path.cwd()),
        }
        
        # Update PID if not already set
        update_instance_with_pid(updates, existing_data)
        
        # Check if this is a background instance and save log file
        bg_env = os.environ.get('HCOM_BACKGROUND')
        if bg_env:
            updates['background'] = True
            updates['background_log_file'] = str(get_hcom_dir() / 'logs' / bg_env)
        
        update_instance_position(instance_name, updates)
        
        parent_pid = os.getppid()
        
        # Check for first-use help
        check_and_show_first_use_help(instance_name)
        
        # Simple polling loop with parent check
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            # Check if parent is alive
            if not is_parent_alive(parent_pid):
                sys.exit(EXIT_SUCCESS)
            
            # Check for new messages
            messages = get_new_messages(instance_name)
            
            if messages:
                # Deliver messages
                max_messages = get_config_value('max_messages_per_delivery', 50)
                messages_to_show = messages[:max_messages]
                
                reason = format_hook_messages(messages_to_show, instance_name)
                output = {"decision": HOOK_DECISION_BLOCK, "reason": reason}
                print(json.dumps(output, ensure_ascii=False), file=sys.stderr)
                sys.exit(EXIT_BLOCK)
            
            # Update heartbeat
            update_instance_position(instance_name, {
                'last_stop': time.time()
            })
            
            time.sleep(0.1)
            
    except Exception:
        pass
    
    sys.exit(EXIT_SUCCESS)

def handle_hook_notification():
    """Handle Notification hook"""
    # Check if active
    if os.environ.get(HCOM_ACTIVE_ENV) != HCOM_ACTIVE_VALUE:
        sys.exit(EXIT_SUCCESS)
    
    try:
        # Read hook input
        hook_data = json.load(sys.stdin)
        transcript_path = hook_data.get('transcript_path', '')
        instance_name = get_display_name(transcript_path) if transcript_path else f"{Path.cwd().name[:2].lower()}claude"
        conversation_uuid = get_conversation_uuid(transcript_path)
        
        # Initialize instance if needed
        initialize_instance_in_position_file(instance_name, conversation_uuid)
        
        # Get existing position data to check if PID is already set
        existing_data = load_instance_position(instance_name)
        
        # Update permission request timestamp
        updates = {
            'last_permission_request': int(time.time()),
            'notification_message': hook_data.get('message', ''),
            'session_id': hook_data.get('session_id', ''),
            'transcript_path': transcript_path,
            'conversation_uuid': conversation_uuid or 'unknown',
            'directory': str(Path.cwd()),
        }
        
        # Update PID if not already set
        update_instance_with_pid(updates, existing_data)
        
        # Check if this is a background instance and save log file
        bg_env = os.environ.get('HCOM_BACKGROUND')
        if bg_env:
            updates['background'] = True
            updates['background_log_file'] = str(get_hcom_dir() / 'logs' / bg_env)
        
        update_instance_position(instance_name, updates)
        
        check_and_show_first_use_help(instance_name)
                    
    except Exception:
        pass
    
    sys.exit(EXIT_SUCCESS)

def handle_hook_pretooluse():
    """Handle PreToolUse hook - auto-approve HCOM_SEND commands"""
    # Check if active
    if os.environ.get(HCOM_ACTIVE_ENV) != HCOM_ACTIVE_VALUE:
        sys.exit(EXIT_SUCCESS)
    
    try:
        # Read hook input
        hook_data = json.load(sys.stdin)
        tool_name = hook_data.get('tool_name', '')
        tool_input = hook_data.get('tool_input', {})
        
        # Only process Bash commands
        if tool_name == 'Bash':
            command = tool_input.get('command', '')
            
            # Check if it's an HCOM_SEND command
            if 'HCOM_SEND:' in command:
                # Auto-approve HCOM_SEND commands
                output = {
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "permissionDecision": "allow",
                        "permissionDecisionReason": "HCOM_SEND command auto-approved"
                    }
                }
                print(json.dumps(output, ensure_ascii=False))
                sys.exit(EXIT_SUCCESS)
    
    except Exception:
        pass
    
    # Let normal permission flow continue for non-HCOM_SEND commands
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
    elif cmd == 'post':
        handle_hook_post()
        return 0
    elif cmd == 'stop':
        handle_hook_stop()
        return 0
    elif cmd == 'notify':
        handle_hook_notification()
        return 0
    elif cmd == 'pre':
        handle_hook_pretooluse()
        return 0
    
    
    # Unknown command
    else:
        print(format_error(f"Unknown command: {cmd}", "Run 'hcom help' for available commands"), file=sys.stderr)
        return 1

if __name__ == '__main__':
    sys.exit(main())
