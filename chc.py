#!/usr/bin/env python3

import os
import sys
import json
import datetime
import tempfile
import re
import time
import select
import shutil
from pathlib import Path

# ANSI color codes
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

# Configurable constants
WAIT_TIMEOUT = int(os.environ.get('CHC_WAIT_TIMEOUT', '600'))  # How long Stop hook waits (seconds)
MAX_MESSAGE_SIZE = int(os.environ.get('CHC_MAX_MESSAGE_SIZE', '4096'))  # Message size limit
SENDER_NAME = os.environ.get('CHC_SENDER_NAME', 'bigboss')  # CLI sender name
MAX_MESSAGES_PER_DELIVERY = int(os.environ.get('CHC_MAX_MESSAGES_PER_DELIVERY', '20'))  # 0 for unlimited
FIRST_USE_TEXT = os.environ.get('CHC_FIRST_USE_TEXT', 'Essential messages only. Say hi in chc chat')  # Group rules shown on first use
SENDER_EMOJI = os.environ.get('CHC_SENDER_EMOJI', '🐳')  # Emoji for CLI sender (no space)
CLI_HINTS = os.environ.get('CHC_CLI_HINTS', '')  # Appended to CLI outputs (chc send, setup, etc)
INSTANCE_HINTS = os.environ.get('CHC_INSTANCE_HINTS', '')  # Appended to instance messages


def require_args(min_count, usage_msg, extra_msg=""):
    """Check argument count and exit with usage if insufficient"""
    if len(sys.argv) < min_count:
        print(f"Usage: {usage_msg}")
        if extra_msg:
            print(extra_msg)
        sys.exit(1)

def is_interactive():
    """Check if running in interactive mode (TTY)"""
    return sys.stdin.isatty()

def get_conversation_uuid(transcript_path):
    """Extract conversation UUID from transcript file (first message UUID)"""
    try:
        if not transcript_path or not os.path.exists(transcript_path):
            return None
        
        with open(transcript_path, 'r') as f:
            first_line = f.readline().strip()
            if first_line:
                entry = json.loads(first_line)
                return entry.get('uuid')
    except Exception:
        return None

def get_display_name(transcript_path):
    syls = ['ka', 'ko', 'ma', 'mo', 'na', 'no', 'ra', 'ro', 'sa', 'so', 'ta', 'to', 'va', 'vo', 'za', 'zo', 'be', 'de', 'fe', 'ge', 'le', 'me', 'ne', 're', 'se', 'te', 've', 'we', 'hi']
    dir_chars = Path.cwd().name[:2].lower()
    
    conversation_uuid = get_conversation_uuid(transcript_path)
    
    if conversation_uuid:
        hash_val = sum(ord(c) for c in conversation_uuid)
        uuid_char = conversation_uuid[0]
        return f"{dir_chars}{syls[hash_val % len(syls)]}{uuid_char}"
    else:
        return f"{dir_chars}claude"

def load_positions(pos_file):
    """Load positions from file with error handling"""
    positions = {}
    if os.path.exists(pos_file):
        try:
            with open(pos_file, 'r') as f:
                positions = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError): 
            pass
    return positions

def initialize_instance_in_position_file(group, instance_name, conversation_uuid=None):
    """Initialize an instance in the position file with directory and conversation info"""
    pos_file = get_pos_file(group)
    positions = load_positions(pos_file)
    
    if instance_name not in positions:
        positions[instance_name] = {
            "pos": 0,
            "directory": str(Path.cwd()),
            "conversation_uuid": conversation_uuid or "unknown",
            "last_tool": 0,
            "last_tool_name": "unknown",
            "last_stop": 0,
            "last_permission_request": 0
        }
        atomic_write(pos_file, json.dumps(positions, indent=2))

def get_chc_dir():
    """Get the CHC directory in user's home"""
    return Path.home() / ".chc"

def get_log_file(group):
    """Get the log file path for a group"""
    return get_chc_dir() / f"{group}.log"

def get_pos_file(group):
    """Get the position file path for a group"""
    return get_chc_dir() / f"{group}.json"


def format_age(seconds):
    """Format time ago in human readable form"""
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        return f"{int(seconds/60)}m"
    else:
        return f"{int(seconds/3600)}h"

def get_transcript_status(transcript_path):
    """Parse transcript to determine current Claude state (thinking/responding/executing only)"""
    try:
        with open(transcript_path, 'r') as f:
            lines = f.readlines()[-5:]
        
        for line in reversed(lines):
            entry = json.loads(line)
            timestamp = datetime.datetime.fromisoformat(entry['timestamp']).timestamp()
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
    """Simple status detection: most recent timestamp wins"""
    now = int(time.time())
    
    last_permission = pos_data.get("last_permission_request", 0)
    last_stop = pos_data.get("last_stop", 0)
    last_tool = pos_data.get("last_tool", 0)
    
    transcript_timestamp = 0
    transcript_status = "inactive"
    
    transcript_path = pos_data.get("transcript_path", "")
    if transcript_path:
        status, _, _, transcript_timestamp = get_transcript_status(transcript_path)
        transcript_status = status
    
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
    
    if age > WAIT_TIMEOUT:
        return "inactive", ""
        
    return most_recent_status, f"({format_age(age)})"

def get_status_block(status_type):
    """Get colored status block for a status type"""
    status_map = {
        "thinking": (BG_CYAN, "◉"),
        "responding": (BG_GREEN, "▷"),
        "executing": (BG_GREEN, "▶"),
        "waiting": (BG_BLUE, "◉"),
        "blocked": (BG_YELLOW, "■"),  # Need permission from human user
        "inactive": (BG_RED, "○")
    }
    
    color, symbol = status_map.get(status_type, (BG_RED, "?"))
    text_color = FG_BLACK if color == BG_YELLOW else FG_WHITE
    return f"{text_color}{BOLD}{color} {symbol} {RESET}"

def log_line(s):
    """Print a message line and move to next line"""
    sys.stdout.write("\r\033[K" + s + "\n")
    sys.stdout.flush()

def update_status(s):
    """Update status line in place"""
    sys.stdout.write("\r\033[K" + s)
    sys.stdout.flush()

def log_line_with_status(message, status):
    """Print message and immediately restore status"""
    sys.stdout.write("\r\033[K" + message + "\n")
    sys.stdout.write("\033[K" + status)
    sys.stdout.flush()

def ensure_chc_dir():
    """Create the CHC directory if it doesn't exist"""
    chc_dir = get_chc_dir()
    chc_dir.mkdir(exist_ok=True)
    return chc_dir

def validate_group_name(group):
    """Validate group name is safe for filesystem use"""
    if not re.match(r'^[a-zA-Z0-9_-]+$', group):
        raise ValueError("Group name must be alphanumeric with _ or -")
    return group

def validate_message(message):
    """Validate message size is reasonable"""
    if len(message) > MAX_MESSAGE_SIZE:
        raise ValueError(f"Message too long (max {MAX_MESSAGE_SIZE} chars)")
    return message

def validate_settings_json(settings_path):
    """Validate settings.json structure"""
    try:
        with open(settings_path, 'r') as f:
            json.load(f)
        return True
    except (json.JSONDecodeError, FileNotFoundError):
        return False

def remove_chc_hooks(settings):
    """Remove existing CHC hooks to prevent duplicates"""
    if 'hooks' not in settings:
        return
    
    for event in ['PostToolUse', 'Stop', 'Notification']:
        if event in settings['hooks']:
            settings['hooks'][event] = [
                matcher for matcher in settings['hooks'][event]
                if not any('chc' in hook.get('command', '') and ('post' in hook.get('command', '') or 'stop' in hook.get('command', '') or 'notify' in hook.get('command', ''))
                          for hook in matcher.get('hooks', []))
            ]

def atomic_write(filepath, content):
    """Write content to file atomically to prevent corruption"""
    filepath = Path(filepath)
    with tempfile.NamedTemporaryFile(mode='w', delete=False, dir=filepath.parent, suffix='.tmp') as tmp:
        tmp.write(content)
        tmp.flush()
        os.fsync(tmp.fileno())
    
    os.replace(tmp.name, filepath)

def update_instance_position(group, instance, update_fields):
    """Shared position update logic - eliminates duplication across hooks"""
    pos_file = get_pos_file(group)
    positions = load_positions(pos_file)
    
    pos_data = positions.get(instance, {})
    pos_data.update(update_fields)
    positions[instance] = pos_data
    atomic_write(pos_file, json.dumps(positions, indent=2))

def should_show_help(group, instance):
    """Check if first-use help should be shown for this instance"""
    pos_file = get_pos_file(group)
    positions = load_positions(pos_file)
    pos_data = positions.get(instance, {})
    
    return not pos_data.get("help_shown", False)

def mark_help_shown(group, instance):
    """Mark that help has been shown for this instance"""
    pos_file = get_pos_file(group)
    positions = load_positions(pos_file)
    
    pos_data = positions.get(instance, {})
    pos_data["help_shown"] = True
    positions[instance] = pos_data
    atomic_write(pos_file, json.dumps(positions, indent=2))

def check_and_show_first_use_help(group, instance):
    """Check and show first-use help if needed - shared across all hooks"""
    if group and instance and should_show_help(group, instance):
        mark_help_shown(group, instance)
        help_text = f'Welcome! CHC group chat active. Your alias is: {instance}. Send messages: echo "CHC_SEND:your message". {FIRST_USE_TEXT} {INSTANCE_HINTS}'.strip()
            
        print(help_text, file=sys.stderr)
        sys.exit(2)



def parse_hook_input():
    """Parse hook input JSON and extract common fields"""
    hook_data = json.load(sys.stdin)
    transcript_path = hook_data.get('transcript_path', '')
    instance = get_display_name(transcript_path)
    conversation_uuid = get_conversation_uuid(transcript_path)
    group = os.environ.get('CHC_GROUP')
    return hook_data, conversation_uuid, instance, group, transcript_path

def format_status_block(bg_color, fg_color, text):
    """Format status block with colors"""
    return f"{bg_color}{BOLD}{fg_color} {text} {RESET}"

def send_message(group, from_instance, message):
    """Send a message to the group log"""
    ensure_chc_dir()
    log_file = get_log_file(group)
    
    escaped_message = message.replace('|', '\\|')
    escaped_from = from_instance.replace('|', '\\|')
    
    timestamp = datetime.datetime.now().isoformat()
    line = f"{timestamp}|{escaped_from}|{escaped_message}\n"
    
    with open(log_file, 'a') as f:
        f.write(line)
        f.flush()

    return True

def parse_log_messages(log_file, start_pos=0):
    """Parse log file messages from a given position"""
    if not os.path.exists(log_file):
        return []
    
    messages = []
    with open(log_file, 'r') as f:
        f.seek(start_pos)
        content = f.read()
        
        if content.strip():
            message_entries = re.split(r'\n(?=\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+\|)', content.strip())
            
            for entry in message_entries:
                if entry and '|' in entry:
                    # Split only on first two | to handle messages containing |
                    parts = entry.split('|', 2)
                    if len(parts) == 3:
                        timestamp, from_instance, message = parts
                        unescaped_from = from_instance.replace('\\|', '|')
                        unescaped_message = message.replace('\\|', '|')
                        messages.append({
                            'timestamp': timestamp,
                            'from': unescaped_from,
                            'message': unescaped_message
                        })
    
    return messages

def get_new_messages(group, instance_name):
    """Get ALL new messages for this instance in the group"""
    ensure_chc_dir()
    log_file = get_log_file(group)
    pos_file = get_pos_file(group)
    
    if not os.path.exists(log_file):
        return []
    
    last_pos = 0
    positions = load_positions(pos_file)
    if positions:
        pos_data = positions.get(instance_name, {})
        last_pos = pos_data.get("pos", 0)
    else:
        last_pos = 0
    
    all_messages = parse_log_messages(log_file, last_pos)
    
    messages = []
    with open(log_file, 'r') as f:
        f.seek(last_pos)
        f.read()
        new_pos = f.tell()
    
    for msg in all_messages:
        if msg['from'] != instance_name:
            messages.append(msg)
    
    pos_data = positions.get(instance_name, {})
    pos_data["pos"] = new_pos
    positions[instance_name] = pos_data
    atomic_write(pos_file, json.dumps(positions, indent=2))
    
    # Limit messages if configured (keep latest messages) - after position update
    if MAX_MESSAGES_PER_DELIVERY > 0 and len(messages) > MAX_MESSAGES_PER_DELIVERY:
        messages = messages[-MAX_MESSAGES_PER_DELIVERY:]
    
    return messages

def get_groups_with_activity():
    """Get groups ordered by recent activity with status info"""
    chc_dir = get_chc_dir()
    if not chc_dir.exists():
        return []
    
    groups_data = []
    now = int(time.time())
    
    for pos_file in chc_dir.glob("*.json"):
        positions = load_positions(pos_file)
        
        if not positions:
            setup_time = int(pos_file.stat().st_mtime)
            age_str = format_age(now - setup_time)
            groups_data.append({
                'name': pos_file.stem,
                'instances': 0,
                'last_activity': setup_time,
                'age': age_str
            })
            continue
            
        last_activity = max((p.get("last_stop", 0) for p in positions.values()), default=0)
        age_str = format_age(now - last_activity) if last_activity else "never"
        
        groups_data.append({
            'name': pos_file.stem,
            'instances': len(positions),
            'last_activity': last_activity,
            'age': age_str
        })
    
    return sorted(groups_data, key=lambda g: g['last_activity'])

def show_groups_and_select(interactive=True):
    """Enhanced group listing and optional selection"""
    groups_data = get_groups_with_activity()
    
    if not groups_data:
        print("No groups found. Use 'chc setup' to create one.")
        return None
    
    print("========================")
    print("CLAUDE HOOK COMMS")
    print("========================")
    print()
    print("Groups:")
    
    for i, g in enumerate(groups_data, 1):
        if interactive:
            # Reverse numbering: 1 = most recent (last in list)
            num = len(groups_data) - i + 1
            print(f"  {num}. {g['name']} ({g['instances']} instances, {g['age']} ago)")
        else:
            print(f"  {g['name']} ({g['instances']} instances, {g['age']} ago)")
    
    if not interactive:
        return None
    
    if len(groups_data) == 1:
        print(f"\nUsing group: {groups_data[0]['name']}")
        return groups_data[0]['name']
     
    while True:
        try:
            choice = input(f"\nSelect group - type number [1-{len(groups_data)}] or name (Enter to cancel): ")
            if choice.isdigit():
                reverse_num = int(choice)
                if 1 <= reverse_num <= len(groups_data):
                    idx = len(groups_data) - reverse_num
                    return groups_data[idx]['name']
            elif choice in [g['name'] for g in groups_data]:
                return choice
            elif choice.strip() == "":
                return None
        except (KeyboardInterrupt, EOFError):
            return None
        print("Enter a number or group name")


def get_status_string(group):
    """Get current status string for all instances"""
    pos_file = get_pos_file(group)
    if not os.path.exists(pos_file):
        return format_status_block(BG_BLUE, FG_WHITE, "no instances")
    
    positions = load_positions(pos_file)
    if not positions:
        return format_status_block(BG_BLUE, FG_WHITE, "no instances")
    
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
            color, symbol = {
                "thinking": (BG_CYAN, "◉"),
                "responding": (BG_GREEN, "▷"),
                "executing": (BG_GREEN, "▶"),
                "waiting": (BG_BLUE, "◉"),
                "blocked": (BG_YELLOW, "■"),
                "inactive": (BG_RED, "○")
            }[status_type]
            text_color = FG_BLACK if color == BG_YELLOW else FG_WHITE
            parts.append(f"{text_color}{BOLD}{color} {count} {symbol} {RESET}")
    
    if parts:
        return "".join(parts)
    else:
        return format_status_block(BG_BLUE, FG_WHITE, "no instances")

def show_recent_messages(group, limit=None, truncate=False):
    """Unified message display - used by both main and alt screen"""
    all_messages = parse_log_messages(get_log_file(group))
    
    if limit is None:
        messages_to_show = all_messages
    else:
        start_idx = max(0, len(all_messages) - limit)
        messages_to_show = all_messages[start_idx:]
    
    for msg in messages_to_show:
        time_obj = datetime.datetime.fromisoformat(msg['timestamp'])
        time_str = time_obj.strftime("%H:%M")
        display_name = f"{SENDER_EMOJI} {msg['from']}" if msg['from'] == SENDER_NAME else msg['from']
        
        if truncate:
            sender = display_name[:10]
            message = msg['message'][:50]
            print(f"   {DIM}{time_str}{RESET} {BOLD}{sender}{RESET}: {message}")
        else:
            print(f"{DIM}{time_str}{RESET} {BOLD}{display_name}{RESET}: {msg['message']}")

def show_main_screen_header(group):
    """Show header for main screen"""
    sys.stdout.write("\033[2J\033[H")
    
    all_messages = parse_log_messages(get_log_file(group))
    message_count = len(all_messages)
    
    print(f"\n{BOLD}{'='*50}{RESET}")
    print(f"  {FG_CYAN}GROUP: {group}{RESET}")
    
    status_line = get_status_string(group)
    print(f"  {BOLD}INSTANCES:{RESET} {status_line}")
    print(f"  {DIM}LOGS: {get_log_file(group)} ({message_count} messages){RESET}")
    print(f"{BOLD}{'='*50}{RESET}\n")

def show_logs_simple(group):
    """New simplified logs viewer with alt-screen details"""
    last_pos = 0
    status_suffix = " ⏎ chat"


    show_main_screen_header(group)
    show_recent_messages(group, limit=5)
    print(f"\n{DIM}{'─'*10} [watching for new messages] {'─'*10}{RESET}")
    
    log_file = get_log_file(group)
    if log_file.exists():
        last_pos = log_file.stat().st_size
    
    # Print newline to ensure status starts on its own line
    print()
    
    # Show initial status
    current_status = get_status_string(group)
    update_status(f"{current_status}{status_suffix}")
    last_status_update = time.time()
    
    # Track last status to avoid unnecessary redraws
    last_status = current_status
    
    try:
        while True:
            now = time.time()
            if now - last_status_update > 2.0:
                current_status = get_status_string(group)
                
                # Only redraw if status text changed
                if current_status != last_status:
                    update_status(f"{current_status}{status_suffix}")
                    last_status = current_status
                
                last_status_update = now
            
            if log_file.exists() and log_file.stat().st_size > last_pos:
                new_messages = parse_log_messages(log_file, last_pos)
                # Use the last known status for consistency
                status_line = f"{last_status}{status_suffix}"
                for msg in new_messages:
                    time_obj = datetime.datetime.fromisoformat(msg['timestamp'])
                    time_str = time_obj.strftime("%H:%M")
                    display_name = f"{SENDER_EMOJI} {msg['from']}" if msg['from'] == SENDER_NAME else msg['from']
                    log_line_with_status(f"{DIM}{time_str}{RESET} {BOLD}{display_name}{RESET}: {msg['message']}", status_line)
                last_pos = log_file.stat().st_size
            
            ready_for_input = False
            if sys.platform == 'win32':
                import msvcrt
                if msvcrt.kbhit():
                    msvcrt.getch()
                    ready_for_input = True
            else:
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    sys.stdin.readline()
                    ready_for_input = True
            
            if ready_for_input:
                # Clear status line before alt screen
                sys.stdout.write("\r\033[K")
                
                message = alt_screen_detailed_status_and_input(group)
                
                # Clear screen and redraw after returning from alt screen
                show_main_screen_header(group)
                show_recent_messages(group)
                print(f"\n{DIM}{'─'*10} [watching for new messages] {'─'*10}{RESET}")
                if log_file.exists():
                    last_pos = log_file.stat().st_size
                
                if message and message.strip():
                    send_message(group, SENDER_NAME, message.strip())
                    print(f"{FG_GREEN}✓ Sent{RESET}")
                
                # Ensure we're on a new line before resuming status updates
                print()
                
                current_status = get_status_string(group)
                update_status(f"{current_status}{status_suffix}")
            
    except KeyboardInterrupt:
        pass
    finally:
        # Ensure clean terminal state
        sys.stdout.write("\033[?1049l\r\033[K")
        print(f"\n{DIM}[stopped]{RESET}")

def show_logs(group=None):
    """Smart logs - auto-detects what you want"""
    if group is None:
        show_groups_and_select(interactive=False)
        return
    
    log_file = get_log_file(group)
    if not log_file.exists():
        print(f"No messages in '{group}' yet")
        show_groups_and_select(interactive=False)
        return
    
    # If terminal, show new interactive view, if piped, output raw
    is_tty = sys.stdout.isatty()
    
    if not is_tty:
        with open(log_file, 'r') as f:
            print(f.read(), end='')
        return
    
    show_logs_simple(group)

def show_instances_by_directory(group):
    """Show instances organized by their working directories"""
    pos_file = get_pos_file(group)
    if not pos_file.exists():
        print(f"   {DIM}No Claude instances connected{RESET}")
        return
    
    positions = load_positions(pos_file)
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
                last_tool_str = datetime.datetime.fromtimestamp(last_tool).strftime("%H:%M:%S") if last_tool else "unknown"
                
                print(f"   {FG_GREEN}->{RESET} {BOLD}{instance_name}{RESET} {status_block} {DIM}{status_type} {age} - last tool use: {last_tool_name} {last_tool_str}{RESET}")
            print()
    else:
        print(f"   {DIM}Error reading instance data{RESET}")

def get_terminal_height():
    """Get current terminal height"""
    try:
        return shutil.get_terminal_size().lines
    except (AttributeError, OSError):
        return 24  # Fallback for older Python or edge cases

def show_recent_activity_alt_screen(group, limit=None):
    """Show recent messages in alt screen format with dynamic height"""
    if limit is None:
        # Calculate available height: total - header(8) - instances(varies) - footer(4) - input(3)
        available_height = get_terminal_height() - 20
        limit = max(2, available_height // 2)
    show_recent_messages(group, limit, truncate=True)

def alt_screen_detailed_status_and_input(group):
    """Show detailed status in alt screen and get user input"""
    sys.stdout.write("\033[?1049h\033[2J\033[H")
    
    try:
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        print(f"{BOLD} CHC DETAILED STATUS{RESET}")
        print(f"{BOLD}{'=' * 70}{RESET}")
        print(f"{FG_CYAN} GROUP: {group}{RESET}")
        print(f"{DIM} LOG FILE: {get_log_file(group)}{RESET}")
        print(f"{DIM} UPDATED: {timestamp}{RESET}")
        print(f"{BOLD}{'-' * 70}{RESET}")
        print()

        show_instances_by_directory(group)
        
        print()
        print(f"{BOLD} RECENT ACTIVITY:{RESET}")
        
        show_recent_activity_alt_screen(group)
        
        print()
        print(f"{BOLD}{'-' * 70}{RESET}")
        print(f"{FG_GREEN} Type message and press Enter to send (empty to cancel):{RESET}")
        message = input(f"{FG_CYAN} > {RESET}")
        
        print(f"{BOLD}{'=' * 70}{RESET}")
        
    finally:
        sys.stdout.write("\033[?1049l")
    
    return message

def setup_group(group):
    """Setup a group - configure hooks for group communication"""
    
    workspace_path = Path.cwd()
    
    print(f"Setting up group '{group}'")
    
    log_file = get_log_file(group)
    if log_file.exists() and log_file.stat().st_size > 0:
        age_days = (datetime.datetime.now() - datetime.datetime.fromtimestamp(log_file.stat().st_mtime)).days
        with open(log_file, 'r') as f:
            message_count = len([line for line in f if line.strip()])
        print(f"Found existing group: {message_count} messages, last activity {age_days} days ago")
        print(f"To start fresh: chc delete {group}")
    
    claude_dir = workspace_path / '.claude'
    claude_dir.mkdir(exist_ok=True)
    
    settings_path = claude_dir / 'settings.local.json'
    settings = {}
    if settings_path.exists(): #TODO: this is too much. no need to do a whole backup thing for the edge case of invalid json and confusing wtf is happening if you accidently have error you want to check and edit the file yourself not some weird backup and replace thing. it should just show a message and fast fail.
        if not validate_settings_json(settings_path):
            backup_path = settings_path.with_suffix('.json.backup')
            print(f"⚠️  Invalid JSON in {settings_path}")
            print(f"   Backup created: {backup_path}")
            print(f"   Starting fresh - restore other settings if needed")
            settings_path.rename(backup_path)
            settings = {}
        else:
            try:
                with open(settings_path, 'r') as f:
                    settings = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                settings = {}
    
    if 'hooks' not in settings:
        settings['hooks'] = {}
    if 'PostToolUse' not in settings['hooks']:
        settings['hooks']['PostToolUse'] = []
    if 'Stop' not in settings['hooks']:
        settings['hooks']['Stop'] = []
    if 'Notification' not in settings['hooks']:
        settings['hooks']['Notification'] = []
    if 'env' not in settings:
        settings['env'] = {}
    if 'permissions' not in settings:
        settings['permissions'] = {}
    if 'allow' not in settings['permissions']:
        settings['permissions']['allow'] = []
    
    remove_chc_hooks(settings)
    
    settings['env']['CHC_GROUP'] = group
    
    # Add permission to allow CHC_SEND commands without prompting
    chc_send_permission = 'Bash(echo "CHC_SEND:*")'
    if chc_send_permission not in settings['permissions']['allow']:
        settings['permissions']['allow'].append(chc_send_permission)

    try:
        import subprocess
        subprocess.run(['chc', '--help'], capture_output=True, check=True)
        chc_cmd = 'chc'
    except (subprocess.CalledProcessError, FileNotFoundError):
        chc_cmd = f'"{sys.executable}" "{os.path.abspath(__file__)}"'

    
    settings['hooks']['PostToolUse'].append({
        'matcher': '.*',
        'hooks': [{
            'type': 'command',
            'command': f'{chc_cmd} post'
        }]
    })
    
    settings['hooks']['Stop'].append({
        'matcher': '',
        'hooks': [{
            'type': 'command',
            'command': f'{chc_cmd} stop',
            'timeout': WAIT_TIMEOUT
        }]
    })
    
    settings['hooks']['Notification'].append({
        'matcher': '',
        'hooks': [{
            'type': 'command',
            'command': f'{chc_cmd} notify'
        }]
    })
    
    with open(settings_path, 'w') as f:
        json.dump(settings, f, indent=2)
    
    ensure_chc_dir()
    log_file = get_log_file(group)
    pos_file = get_pos_file(group)
    
    if not log_file.exists():
        log_file.touch()
    
    if not pos_file.exists():
        atomic_write(pos_file, json.dumps({}, indent=2))
    
    print()
    print(f"✅ Setup complete! CHC hooks configured in {workspace_path} for group '{group}'")
    print(CLI_HINTS.format(group=group, workspace=workspace_path))
    print()
    print(f"1. Start claude Code in {workspace_path}")
    print()
    print("2. Run: chc")
    print()

def handle_hook_post():
    """Handle PostToolUse hook - check for CHC_SEND messages and deliver any new messages"""
    try:
        hook_data, conversation_uuid, instance, group, transcript_path = parse_hook_input()
        
        if group and instance:
            initialize_instance_in_position_file(group, instance, conversation_uuid)
            
            tool_name = hook_data.get('tool_name', 'unknown')
            update_instance_position(group, instance, {
                "last_tool": int(time.time()),
                "last_tool_name": tool_name,
                "transcript_path": transcript_path,
                "conversation_uuid": conversation_uuid
            })
        
        # Check for CHC_SEND pattern in Bash commands first
        sent_message = False
        if hook_data.get('tool_name') == 'Bash':
            command = hook_data.get('tool_input', {}).get('command', '')
            if 'CHC_SEND:' in command:
                message = command.split('CHC_SEND:', 1)[1].strip()
                # Strip surrounding quotes that might be part of the echo command
                if len(message) >= 2 and message[0] == '"' and message[-1] == '"':
                    message = message[1:-1]
                if group and instance:
                    try:
                        if send_message(group, instance, validate_message(message)):
                            sent_message = True
                    except ValueError:
                        output = {"reason": f"❌ Message too long (max {MAX_MESSAGE_SIZE} chars)"}
                        print(json.dumps(output, ensure_ascii=False), file=sys.stderr)
                        sys.exit(2)
        
        # Then check for new messages and deliver immediately
        if group and instance:
            messages = get_new_messages(group, instance)
            if messages:
                reason = format_hook_messages(messages, instance)
                if sent_message:
                    reason = f"✓ Sent to {group} | {reason}"
                output = {"decision": "block", "reason": reason}
                print(json.dumps(output, ensure_ascii=False), file=sys.stderr)
                sys.exit(2)
            elif sent_message:
                output = {"reason": f"✓ Sent to {group}"}
                print(json.dumps(output, ensure_ascii=False), file=sys.stderr)
                sys.exit(2)
        
        check_and_show_first_use_help(group, instance)
    except Exception:
        pass

def handle_hook_notification():
    """Handle Notification hook - track permission requests"""
    try:
        hook_data, _, instance, group, transcript_path = parse_hook_input()
        
        if group and instance:
            update_instance_position(group, instance, {
                "last_permission_request": int(time.time()),
                "notification_message": hook_data.get('message', ''),
                "transcript_path": transcript_path
            })
            
            # Show welcome message if needed
            check_and_show_first_use_help(group, instance)
        
    except Exception:
        pass  # Silent failure for hooks

def format_hook_messages(messages, instance):
    """Format messages for hooks - keep essential identity info"""
    if len(messages) == 1:
        msg = messages[0]
        reason = f"{msg['from']} → {instance}: {msg['message']}"
    else:
        parts = [f"{msg['from']}: {msg['message']}" for msg in messages]
        reason = f"{len(messages)} messages → {instance}: " + " | ".join(parts)
    
    # Add instance hints
    reason += f" {INSTANCE_HINTS}"
    
    return reason

def handle_hook_stop():
    """Handle Stop hook - check for messages and wait"""
    try:
        _, conversation_uuid, instance, group, transcript_path = parse_hook_input()
        
        if not group or not instance:
            return
        
        # Get parent PID once at startup
        parent_pid = os.getppid()
        
        # Initialize instance in position file if needed
        initialize_instance_in_position_file(group, instance, conversation_uuid)
        
        # Update transcript path for status tracking
        update_instance_position(group, instance, {
            "transcript_path": transcript_path
        })
        
        # Check for unread messages
        messages = get_new_messages(group, instance)
        if messages:
            reason = format_hook_messages(messages, instance)
            output = {"decision": "block", "reason": reason}
            print(json.dumps(output, ensure_ascii=False), file=sys.stderr)
            sys.exit(2)
        
        check_and_show_first_use_help(group, instance)
        
        wait_msg = f"CHC groupchat: '{group}'. To send: echo CHC_SEND:message{INSTANCE_HINTS}".strip()
        print(wait_msg, file=sys.stderr)
        while True:
            time.sleep(1)
            
            # Check if parent process still exists
            if sys.platform == 'win32':
                # Windows: Try to open the process
                try:
                    import ctypes
                    kernel32 = ctypes.windll.kernel32
                    handle = kernel32.OpenProcess(0x0400, False, parent_pid)  # PROCESS_QUERY_INFORMATION
                    if handle == 0:
                        sys.exit(0)  # Parent gone
                    kernel32.CloseHandle(handle)
                except:
                    pass  # If anything fails, keep running
            else:
                # Unix/Mac/Linux: Send signal 0
                try:
                    os.kill(parent_pid, 0)
                except ProcessLookupError:
                    sys.exit(0)  # Parent gone
                except:
                    pass  # If anything fails, keep running
            
            update_instance_position(group, instance, {
                "last_stop": int(time.time())
            })
            
            messages = get_new_messages(group, instance)
            if messages:
                reason = format_hook_messages(messages, instance)
                output = {"decision": "block", "reason": reason}
                print(json.dumps(output, ensure_ascii=False), file=sys.stderr)
                sys.exit(2)
                
    except Exception:
        pass

def show_help():
    print("Claude Hook Comms")
    print("Real-time messaging between Claude Code instances")
    print()
    print("Usage:")
    print("  chc                           - Messaging & status dashboard")
    print("  chc setup <group> <folder(s)> - Setup folder(s) for group")
    print("  chc delete <group>            - Delete group and conversation history")
    print()
    print("Quick Start:")
    print("  1. chc setup mygroup .        - Setup current folder")
    print("  2. claude 'say hi'            - Start Claude in this folder")
    print("  3. chc                        - Open dashboard")
    print()
    print("Non-interactive commands:")
    print("  chc watch <group>             - View group status")
    print("  chc watch <group> --logs      - View message history")
    print(f"  chc send <group> <msg>        - Send message as {SENDER_NAME}")
    print()
    print("Conversation log files stored in ~/.chc/{group}.log")

def main():
    if len(sys.argv) < 2:
        if not is_interactive():
            show_groups_and_select(interactive=False)
        else:
            group = show_groups_and_select(interactive=True)
            if group:
                show_logs(group)
            else:
                show_help()
        return
    
    cmd = sys.argv[1]
    
    if cmd == 'setup':
        require_args(4, "chc setup <group> <folder1> [folder2...]", 
                    "Examples:\n  chc setup myteam .          # Setup current folder\n  chc setup myteam dir1 dir2  # Setup multiple folders")
            
        group = sys.argv[2]
        directories = sys.argv[3:]
        
        try:
            group = validate_group_name(group)
            for directory in directories:
                dir_path = Path(directory).resolve()
                if not dir_path.exists():
                    print(f"Error: Directory '{directory}' does not exist")
                    sys.exit(1)
                
                original_cwd = Path.cwd()
                os.chdir(dir_path)
                try:
                    setup_group(group)
                finally:
                    os.chdir(original_cwd)
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)
            
    elif cmd == 'watch':
        require_args(3, "chc watch <group> [--logs]", "For interactive dashboard, use: chc")
            
        group = sys.argv[2]
        
        show_logs_only = '--logs' in sys.argv[3:]
        
        try:
            group = validate_group_name(group)
            
            if not is_interactive():
                if show_logs_only:
                    log_file = get_log_file(group)
                    if log_file.exists():
                        messages = parse_log_messages(log_file)
                        for msg in messages[-20:]:
                            time_obj = datetime.datetime.fromisoformat(msg['timestamp'])
                            time_str = time_obj.strftime("%H:%M")
                            print(f"{time_str} {msg['from']}: {msg['message']}")
                    else:
                        print(f"No messages in '{group}' yet")
                    print(CLI_HINTS.format(group=group))
                else:
                    print(f"CHC STATUS: {group}")
                    print("INSTANCES:")
                    show_instances_by_directory(group)
                    print("RECENT ACTIVITY:")
                    show_recent_activity_alt_screen(group)
                    print(f"LOG FILE: {get_log_file(group)}")
                    print(CLI_HINTS.format(group=group))
            else:
                show_logs(group)
                
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)
            
    elif cmd == 'delete':
        require_args(3, "chc delete <group>")
        try:
            group = validate_group_name(sys.argv[2])
            log_file = get_log_file(group)
            pos_file = get_pos_file(group)
            
            files_deleted = []
            if log_file.exists():
                log_file.unlink()
                files_deleted.append("conversation log")
            if pos_file.exists():
                pos_file.unlink()
                files_deleted.append("position data")
            
            if files_deleted:
                print(f"Deleted {group}: {', '.join(files_deleted)}")
            else:
                print(f"Group '{group}' not found")
            
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)
            
    elif cmd == 'send':
        require_args(4, "chc send <group> <message>")
        try:
            group = validate_group_name(sys.argv[2])
            message = validate_message(sys.argv[3])
            send_message(group, SENDER_NAME, message)
            print(f"Message sent to {group}")
            print(CLI_HINTS.format(group=group))
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)
            
    elif cmd in ['post']:
        handle_hook_post()
    elif cmd in ['stop']:
        handle_hook_stop()
    elif cmd in ['notify']:
        handle_hook_notification()
    elif cmd in ['--help', 'help']:
        show_help()
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)

if __name__ == '__main__':
    main()