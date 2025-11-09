"""Hook handler functions"""
from __future__ import annotations
from typing import Any
from pathlib import Path
import sys
import os
import time
import json
import re
import socket
import select
import shlex

from ..shared import SENDER, MAX_MESSAGES_PER_DELIVERY, STOP_HOOK_POLL_INTERVAL, HCOM_COMMAND_PATTERN
from ..core.paths import hcom_path, FLAGS_DIR
from ..core.instances import (
    load_instance_position, update_instance_position, set_status,
    in_subagent_context
)
from ..core.messages import (
    get_unread_messages, format_hook_messages,
    should_deliver_message
)
from ..core.config import get_config

from .utils import (
    is_safe_hcom_command,
    build_hcom_bootstrap_text, build_launch_context,
    build_hcom_command, disable_instance, log_hook_error,
    notify_instance
)

# Import identity helpers from core
from ..core.instances import initialize_instance_in_position_file


def format_subagent_hcom_instructions(alias: str) -> str:
    """Format HCOM usage instructions for subagents"""
    hcom_cmd = build_hcom_command()

    # Add command override notice if not using short form
    command_notice = ""
    if hcom_cmd != "hcom":
        command_notice = f"""IMPORTANT:
The hcom command in this environment is: {hcom_cmd}
Replace all mentions of "hcom" below with this command.

"""

    return f"""{command_notice}[HCOM INFORMATION]
Your HCOM alias is: {alias}
HCOM is a communication tool. You are now connected.

- To Send a message, run:
  hcom send 'your message' --_hcom_sender {alias}
  (use '@alias' for direct messages)

- You receive messages automatically via bash feedback or hooks.
  There is no proactive checking for messages needed.

- When finished working or when waiting on a reply, always run:
  hcom done --_hcom_sender {alias}
  (you will be automatically alerted of any new messages immediately)
  hcom done does not accept any other arguments, if you need to send a message use hcom send first then hcom done after.

- {{"decision": "block"}} text is normal operation
- Prioritize @{SENDER} over other participants
- First action: Announce your online presence in hcom chat
------"""


def _setup_subagent_identity(instance_data: dict[str, Any] | None, instance_name: str, tool_input: dict[str, Any], session_id: str) -> dict[str, Any] | None:
    """Parent context: generate/reuse subagent ID, inject instructions

    Handles: resume lookup, ID generation, file init, prompt injection.
    Returns hook output dict or None.

    IMPORTANT: Defensive guards needed - instance_data may be {} if file missing.
    """
    if not instance_data:
        instance_data = {}

    subagent_type = tool_input.get('subagent_type', 'unknown')
    resume_agent_id = tool_input.get('resume')
    parent_enabled = instance_data.get('enabled', False)

    # Resume lookup
    existing_hcom_id = None
    if resume_agent_id:
        mappings = instance_data.get('subagent_mappings', {})
        for hcom_id, agent_id in mappings.items():
            if agent_id == resume_agent_id:
                existing_hcom_id = hcom_id
                break

    if existing_hcom_id:
        # Reuse existing
        subagent_id = existing_hcom_id
        from ..core.db import get_instance

        if not get_instance(subagent_id):
            initialize_instance_in_position_file(subagent_id, parent_session_id=session_id, parent_name=instance_name, enabled=parent_enabled)
        else:
            update_instance_position(subagent_id, {'enabled': parent_enabled})
    else:
        # Generate new (atomic collision detection via DB)
        import sqlite3
        from ..core.db import get_db

        count = 1
        conn = get_db()
        for _ in range(1000):
            subagent_id = f"{instance_name}_{subagent_type}_{count}"
            try:
                # Try to reserve name with placeholder row (use NULL not empty string)
                conn.execute(
                    "INSERT INTO instances (name, session_id, created_at) VALUES (?, ?, ?)",
                    (subagent_id, None, time.time())
                )
                conn.commit()
                break  # Success - name reserved
            except sqlite3.IntegrityError:
                # Name collision - try next number
                count += 1
                continue

        # Initialize with full data (replaces placeholder, sets parent_name)
        if not initialize_instance_in_position_file(subagent_id, parent_session_id=session_id, parent_name=instance_name, enabled=parent_enabled):
            # Initialization failed - delete placeholder and abort
            log_hook_error(f'pretooluse:initialize_subagent({subagent_id})', Exception('Failed to initialize instance'))
            try:
                conn.execute("DELETE FROM instances WHERE name = ?", (subagent_id,))
                conn.commit()
            except Exception:
                pass
            # Continue without subagent setup (fallback to no subagent)
            return

    # Add to parent tracking
    current_subagents = instance_data.get('current_subagents', [])
    if subagent_id not in current_subagents:
        current_subagents.append(subagent_id)
    update_instance_position(instance_name, {'current_subagents': current_subagents})

    # Build prompt injection
    hcom_detected = 'hcom' in tool_input.get('prompt', '').lower()
    original_prompt = tool_input.get('prompt', '')

    if parent_enabled:
        modified_prompt = f"""{format_subagent_hcom_instructions(subagent_id)}

{original_prompt}"""
    elif hcom_detected:
        hcom_cmd = build_hcom_command()
        modified_prompt = f"""[To use hcom run this command: '{hcom_cmd} start --_hcom_sender {subagent_id}']

{original_prompt}"""
    else:
        modified_prompt = original_prompt

    updated_input = tool_input.copy()
    updated_input['prompt'] = modified_prompt

    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "allow",
            "updatedInput": updated_input
        }
    }


def _cleanup_orphaned_subagents(instance_name: str, instance_data: dict[str, Any] | None) -> dict[str, Any]:
    """Parent context: disable orphaned subagent instances.

    Uses parent_name field for unambiguous subagent lookup.

    CRITICAL: Cannot use parent_session_id FK for vanilla instances:
    - Vanilla instances have session_id=NULL
    - Subagents get parent_session_id=NULL
    - Query "WHERE parent_session_id=NULL" returns empty (NULL != NULL in SQL)
    - FK constraint won't work for vanilla instances

    CRITICAL: Cannot use name prefix (WHERE name LIKE 'parent_%'):
    - Ambiguous when instance names overlap (e.g., "alice" vs "alice_foo")
    - "alice_reviewer" could be child of "alice" OR separate instance

    parent_name field provides unambiguous parent→child relationship.

    Returns dict of updates (may be empty).
    """
    if not instance_data:
        return {}

    from ..core.db import get_db
    conn = get_db()

    # Query by parent_name (unambiguous, works for all instances)
    rows = conn.execute(
        "SELECT name, status FROM instances WHERE parent_name = ?",
        (instance_name,)
    ).fetchall()

    for row in rows:
        disable_instance(row['name'])
        if row['status'] != 'exited':
            set_status(row['name'], 'exited', 'orphaned')

    if in_subagent_context(instance_data):
        return {'current_subagents': []}
    return {}


def _setup_tcp_notification(_instance_name: str) -> tuple[socket.socket | None, float, bool]:
    """Parent context: Setup TCP server for instant wake

    Returns (server, timeout, tcp_mode)
    """
    try:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(('127.0.0.1', 0))
        server.listen(128)
        server.setblocking(False)
        return server, 30.0, True
    except Exception as e:
        log_hook_error('stop:notify_setup', e)
        return None, STOP_HOOK_POLL_INTERVAL, False


def _should_exit_polling(instance_data: dict[str, Any] | None, instance_name: str) -> tuple[bool, str, bool]:
    """Parent context: Check all exit conditions

    Returns (should_exit, reason, should_set_exited)
    Only 'disabled' reason should set status to exited.
    """
    # Flag file (highest priority) - preserve current status
    flag_file = get_user_input_flag_file(instance_name)
    if flag_file.exists():
        try:
            flag_file.unlink()
        except (FileNotFoundError, PermissionError):
            pass
        return (True, 'flag_file', False)

    if not instance_data:
        return (False, '', False)

    # Session ended - preserve current status
    if instance_data.get('session_ended'):
        return (True, 'session_ended', False)

    # Recent user input - preserve current status
    if time.time() - instance_data.get('last_user_input', 0) < 0.2:
        return (True, 'user_input', False)

    # Disabled - set exited status
    if not instance_data.get('enabled', False):
        return (True, 'disabled', True)

    return (False, '', False)


def _run_stop_polling_loop(instance_name: str, timeout: float, notify_server: socket.socket | None, poll_timeout: float, first_stop: bool) -> tuple[int, dict[str, Any] | None, bool]:
    """Parent context: Main polling loop with TCP select or fallback sleep

    Returns (exit_code, hook_output_dict | None, timed_out)
    - exit_code: 0 for normal exit, 2 for message delivery
    - hook_output_dict: message output if exit_code is 2
    - timed_out: True only if loop genuinely timed out (not early exits)
    """
    start_time = time.time()
    last_heartbeat = start_time

    while time.time() - start_time < timeout:
        instance_data = load_instance_position(instance_name)

        # Check exit conditions
        should_exit, _reason, should_set_exited = _should_exit_polling(instance_data, instance_name)
        if should_exit:
            if should_set_exited:
                set_status(instance_name, 'exited')
            return (0, None, False)  # Early exit, not timeout

        # Check for messages (preserves current throttling and format)
        messages, _ = get_unread_messages(instance_name, update_position=True)
        if messages:
            messages_to_show = messages[:MAX_MESSAGES_PER_DELIVERY]
            formatted = format_hook_messages(messages_to_show, instance_name)
            set_status(instance_name, 'delivered', messages_to_show[0]['from'])

            output = {"decision": "block", "reason": formatted}
            if first_stop:
                output["systemMessage"] = "hi. im stop hook. dont worry that i returned blocking error its normal hcom message delivery"

            return (2, output, False)  # Message delivery, not timeout

        # Wait for notification or timeout
        if notify_server:
            try:
                readable, _, _ = select.select([notify_server], [], [], poll_timeout)
                # Update heartbeat after select
                try:
                    update_instance_position(instance_name, {'last_stop': time.time()})
                except Exception:
                    pass
                if readable:
                    # Drain notifications
                    while True:
                        try:
                            notify_server.accept()[0].close()
                        except BlockingIOError:
                            break
            except (OSError, ValueError, InterruptedError) as e:
                # Socket became invalid - switch to polling
                log_hook_error(f'stop:select_failed({instance_name})', e)
                try:
                    notify_server.close()
                except:
                    pass
                notify_server = None
                poll_timeout = STOP_HOOK_POLL_INTERVAL
                try:
                    update_instance_position(instance_name, {'tcp_mode': False, 'notify_port': None})
                except Exception:
                    pass
        else:
            # Fallback mode - heartbeat for staleness detection
            now = time.time()
            if now - last_heartbeat >= 0.5:
                try:
                    update_instance_position(instance_name, {'last_stop': now})
                    last_heartbeat = now
                except Exception as e:
                    log_hook_error(f'stop:heartbeat_update({instance_name})', e)
            time.sleep(poll_timeout)

    # Timeout occurred - return flag, let orchestrator set status
    return (0, None, True)


def handle_pretooluse(hook_data: dict[str, Any], instance_name: str) -> None:
    """Handle PreToolUse hook - status tracking, session injection, subagent setup"""
    instance_data = load_instance_position(instance_name)
    tool_name = hook_data.get('tool_name', '')

    # Status tracking (inline - too simple for helper)
    if instance_data.get('enabled', False):
        has_sender_flag = False
        if tool_name == 'Bash':
            command = hook_data.get('tool_input', {}).get('command', '')
            has_sender_flag = '--_hcom_sender' in command

        if not has_sender_flag:
            if in_subagent_context(instance_data):
                for subagent_id in instance_data.get('current_subagents', []):
                    set_status(subagent_id, 'active', tool_name)
            else:
                set_status(instance_name, 'active', tool_name)

    # Auto-approve hcom commands
    if tool_name == 'Bash':
        tool_input = hook_data.get('tool_input', {})
        command = tool_input.get('command', '')
        if command:
            matches = list(re.finditer(HCOM_COMMAND_PATTERN, command))
            if matches and is_safe_hcom_command(command):
                output = {
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "permissionDecision": "allow"
                    }
                }
                print(json.dumps(output, ensure_ascii=False))
                sys.exit(0)

    # Subagent identity injection
    if tool_name == 'Task':
        tool_input = hook_data.get('tool_input', {})
        session_id = hook_data.get('session_id', '')
        if output := _setup_subagent_identity(instance_data, instance_name, tool_input, session_id):
            print(json.dumps(output, ensure_ascii=False))
            sys.exit(0)


def handle_stop(_hook_data: dict[str, Any], instance_name: str, _updates: dict[str, Any], instance_data: dict[str, Any] | None) -> None:
    """Handle Stop hook: poll for messages and deliver"""
    notify_server = None
    try:
        # Check if first stop hook run
        first_stop = not instance_data or instance_data.get('last_stop', 0) == 0

        updates: dict[str, Any] = {}
        updates['last_stop'] = time.time()
        timeout = get_config().timeout
        updates['wait_timeout'] = timeout
        set_status(instance_name, 'waiting')

        # Cleanup orphaned subagents (always returns dict)
        cleanup_updates = _cleanup_orphaned_subagents(instance_name, instance_data)
        updates.update(cleanup_updates)

        # Setup TCP notification
        notify_server, poll_timeout, tcp_mode = _setup_tcp_notification(instance_name)
        updates['notify_port'] = notify_server.getsockname()[1] if notify_server else None
        updates['tcp_mode'] = tcp_mode

        try:
            update_instance_position(instance_name, updates)
        except Exception as e:
            log_hook_error(f'stop:update_instance_position({instance_name})', e)

        # Run polling loop
        exit_code, output, timed_out = _run_stop_polling_loop(instance_name, timeout, notify_server, poll_timeout, first_stop)

        # Emit output if present (preserves decision:block)
        if output:
            print(json.dumps(output, ensure_ascii=False))

        # Set exited status only if genuinely timed out
        if timed_out:
            set_status(instance_name, 'exited')

        sys.exit(exit_code)

    except Exception as e:
        log_hook_error('handle_stop', e)
        sys.exit(0)
    finally:
        if notify_server:
            try:
                notify_server.close()
                update_instance_position(instance_name, {
                    'notify_port': None,
                    'tcp_mode': False
                })
            except Exception:
                pass


def handle_subagent_stop(_hook_data: dict[str, Any], parent_name: str, _updates: dict[str, Any], _instance_data: dict[str, Any] | None) -> None:
    """SubagentStop: Guard hook - directs subagents to use 'done' command.
    Group timeout: if any subagent disabled or idle too long, disable ALL.
    parent_name is because subagents share the same session_id as parents
    (so instance_data in this case is the same for parents and children).
    Parents will never run this hook. Normal instances will never hit this hook.
    This hook is only for subagents. Only subagent will ever hit this hook.
    The name will resolve to parents name. This is normal and does not mean that
    the parent is running this hook. Only subagents run this hook.
    """
    # Check subagent states using SQL query
    from ..core.db import get_db
    conn = get_db()

    # Get all subagents for this parent
    subagents = conn.execute(
        "SELECT name, enabled, last_stop FROM instances WHERE parent_name = ?",
        (parent_name,)
    ).fetchall()

    # Exit silently if no subagents
    if not subagents:
        sys.exit(0)

    has_enabled = any(row['enabled'] for row in subagents)
    has_disabled = any(not row['enabled'] for row in subagents)

    # If any subagent disabled (timed out in PostToolUse), disable all and exit
    if has_disabled:
        for row in subagents:
            update_instance_position(row['name'], {'enabled': False})
            set_status(row['name'], 'exited', 'timeout')
        sys.exit(0)

    # Exit silently if all disabled
    if not has_enabled:
        sys.exit(0)

    # Check timeout - if any subagent idle too long, disable all and exit
    timeout = get_config().subagent_timeout
    now = time.time()

    for row in subagents:
        if row['enabled']:
            last_stop = row['last_stop'] or 0
            if last_stop > 0 and (now - last_stop) > timeout:
                # Timeout exceeded - disable all subagents
                for row2 in subagents:
                    update_instance_position(row2['name'], {'enabled': False})
                    set_status(row2['name'], 'exited', 'timeout')
                sys.exit(0)

    # reminder to run 'done' command
    reminder = (
        "[HCOM]: You MUST run 'hcom done --_hcom_sender <your_alias>' "
        "This allows you to receive messages and prevents timeout. "
        "Run this command NOW."
    )

    output = {"decision": "block", "reason": reminder}
    print(json.dumps(output, ensure_ascii=False), file=sys.stderr)
    sys.exit(2)


def handle_notify(hook_data: dict[str, Any], instance_name: str, updates: dict[str, Any], instance_data: dict[str, Any] | None) -> None:
    """Handle Notification hook - track permission requests"""
    message = hook_data.get('message', '')

    # Filter generic message only when instance is already idle (Stop hook running)
    if message == "Claude is waiting for your input":
        current_status = instance_data.get('status', '') if instance_data else ''
        if current_status == 'waiting':
            return  # Instance is idle, Stop hook will maintain waiting status

    # Update status based on subagent context
    if instance_data and in_subagent_context(instance_data):
        # In subagent context - update only subagents in current_subagents list (don't update parent)
        current_list = instance_data.get('current_subagents', [])
        for subagent_id in current_list:
            set_status(subagent_id, 'blocked', message)
    else:
        # Not in Task (parent context) - update parent only
        updates['notification_message'] = message
        update_instance_position(instance_name, updates)
        set_status(instance_name, 'blocked', message)


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
    notify_port = instance_data.get('notify_port') if instance_data else None

    # Session_ended prevents user receiving messages(?) so reset it.
    if is_matched_resume and instance_data and instance_data.get('session_ended'):
        update_instance_position(instance_name, {'session_ended': False})
        instance_data['session_ended'] = False  # Resume path reactivates Stop hook polling

    # Disable orphaned subagents (user cancelled/interrupted Task or resumed)
    if instance_data:
        from ..core.db import get_db
        conn = get_db()
        subagents = conn.execute(
            "SELECT name, status FROM instances WHERE parent_name = ?",
            (instance_name,)
        ).fetchall()
        for row in subagents:
            disable_instance(row['name'])
            # Only set exited if not already exited
            if row['status'] != 'exited':
                set_status(row['name'], 'exited', 'orphaned')
        # Clear current subagents list if set
        if (instance_data.get('current_subagents')):
            update_instance_position(instance_name, {'current_subagents': []})

    # Coordinate with Stop hook only if enabled AND Stop hook is active
    # Determine if stop hook is active - check tcp_mode or timestamp
    tcp_mode = instance_data.get('tcp_mode', False) if instance_data else False
    if tcp_mode:
        # TCP mode - assume active (stop hook self-reports if it exits/fails)
        stop_is_active = True
    else:
        # Fallback mode - check timestamp
        stop_is_active = (time.time() - last_stop) < 1.0

    if is_enabled and stop_is_active:
        # Create flag file FIRST (must exist before Stop hook wakes)
        flag_file = get_user_input_flag_file(instance_name)
        try:
            flag_file.touch()
        except (OSError, PermissionError):
            # Failed to create flag, fall back to timestamp-only coordination
            pass

        # Set timestamp (backup mechanism)
        updates['last_user_input'] = time.time()
        update_instance_position(instance_name, updates)

        # Send TCP notification LAST (Stop hook wakes, sees flag, exits immediately)
        if notify_port:
            notify_instance(instance_name)

        # Wait for Stop hook to delete flag file (PROOF of exit)
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
    """Handle SessionStart hook - write session ID to env file & show initial msg"""
    # Write session ID to CLAUDE_ENV_FILE for automatic identity resolution
    session_id = hook_data.get('session_id')
    env_file = os.environ.get('CLAUDE_ENV_FILE')

    if session_id and env_file:
        try:
            with open(env_file, 'a') as f:
                f.write(f'\nexport HCOM_SESSION_ID={session_id}\n')
        except Exception as e:
            # Fail silently - hook safety
            log_hook_error('sessionstart:env_file', e)

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


def _deliver_task_freeze_messages(instance_name: str, parent_event_id: int) -> dict[str, Any]:
    """Parent context: deliver messages from Task execution period"""
    from ..core.db import get_events_since

    # Query all message events since parent froze (SINGLE query to avoid race condition)
    events = get_events_since(parent_event_id, event_type='message')

    if not events:
        return {'last_event_id': parent_event_id}

    # Determine last_event_id from the events we actually retrieved
    last_id = max(e['id'] for e in events)

    # Build message dicts from events (reuse for both subagent and parent filtering)
    all_messages = [
        {
            'timestamp': e['timestamp'],
            'from': e['data']['from'],
            'message': e['data']['text']
        }
        for e in events
    ]

    # Get database connection for instance queries
    from ..core.db import get_db
    conn = get_db()

    # Get all instance names for filtering
    all_instance_names = [row['name'] for row in conn.execute("SELECT name FROM instances").fetchall()]

    # Get subagent names using parent_name field (not prefix matching - avoids collision)
    subagent_names = [row['name'] for row in conn.execute(
        "SELECT name FROM instances WHERE parent_name = ?", (instance_name,)
    ).fetchall()]

    # Filter subagent messages from the SAME event set (no second query)
    subagent_msgs = []
    for msg in all_messages:
        sender = msg['from']
        # Messages FROM subagents
        if sender in subagent_names:
            subagent_msgs.append(msg)
        # Messages TO subagents via @mentions or broadcasts
        elif subagent_names and any(should_deliver_message(msg, name, subagent_names) for name in subagent_names):
            if msg not in subagent_msgs:  # Avoid duplicates
                subagent_msgs.append(msg)

    # Get parent-relevant messages
    parent_msgs = [
        msg for msg in all_messages
        if msg not in subagent_msgs and should_deliver_message(msg, instance_name, all_instance_names)
    ]

    # Combine and format
    all_relevant = subagent_msgs + parent_msgs
    all_relevant.sort(key=lambda m: m['timestamp'])

    if all_relevant:
        formatted = '\n'.join(f"{msg['from']}: {msg['message']}" for msg in all_relevant)
        summary = (
            f"[Task tool completed - Message history during Task tool]\n"
            f"The following {len(all_relevant)} message(s) occurred:\n\n"
            f"{formatted}\n\n"
            f"[End of message history. Subagents have finished and are no longer active.]"
        )
        return {'summary': summary, 'last_event_id': last_id}

    return {'last_event_id': last_id}


def _save_resume_mapping(instance_name: str, instance_data: dict[str, Any] | None, tool_input: dict[str, Any], tool_response: dict[str, Any]) -> None:
    """Parent context: store agentId → HCOM ID mapping for Task resume"""
    agent_id = tool_response.get('agentId')
    if not agent_id:
        return

    prompt = tool_input.get('prompt', '')
    match = re.search(r'--_hcom_sender\s+(\S+)', prompt)
    if not match:
        return

    hcom_subagent_id = match.group(1)
    mappings = instance_data.get('subagent_mappings', {}) if instance_data else {}
    mappings[hcom_subagent_id] = agent_id
    update_instance_position(instance_name, {'subagent_mappings': mappings})


def _mark_task_subagents_exited(instance_name: str) -> None:
    """Parent context: mark all subagents exited after Task completion"""
    from ..core.db import get_db
    conn = get_db()
    subagents = conn.execute("SELECT name FROM instances WHERE parent_name = ?", (instance_name,)).fetchall()
    for row in subagents:
        set_status(row['name'], 'exited', 'task_completed')


def _handle_task_completion(instance_name: str, instance_data: dict[str, Any] | None, tool_input: dict[str, Any], tool_response: dict[str, Any]) -> dict[str, Any] | None:
    """Parent context: Task tool completion flow

    CRITICAL: Maintains atomic sequencing of state updates:
    1. Deliver freeze messages
    2. Update position + clear current_subagents (atomic)
    3. Save resume mapping
    4. Mark subagents exited

    Breaking this sequence can cause resume/timeout race bugs.
    Returns hook output dict if messages to deliver, else None.
    """
    parent_event_id = instance_data.get('last_event_id', 0) if instance_data else 0

    # Deliver freeze-period messages
    result = _deliver_task_freeze_messages(instance_name, parent_event_id)

    # Update position and clear tracking (atomic)
    update_instance_position(instance_name, {
        'last_event_id': result['last_event_id'],
        'current_subagents': []
    })

    # Save resume mapping
    _save_resume_mapping(instance_name, instance_data, tool_input, tool_response)

    # Mark subagents exited
    _mark_task_subagents_exited(instance_name)

    # Return output for delivery if present
    if summary := result.get('summary'):
        return {
            "systemMessage": "[Task subagent messages shown to instance]",
            "hookSpecificOutput": {
                "hookEventName": "PostToolUse",
                "additionalContext": summary
            }
        }
    return None


def _inject_subagent_hcom_start_instructions(command: str, instance_data: dict[str, Any] | None) -> dict[str, Any] | None:
    """Subagent context: inject instructions when subagent runs 'hcom start'

    Returns hook output dict or None.
    """
    if not (instance_data and in_subagent_context(instance_data)):
        return None

    # Match all hcom invocation variants (hcom, uvx hcom, python -m hcom, .pyz)
    start_pattern = re.compile(r'((?:uvx\s+)?hcom|python3?\s+-m\s+hcom|(?:python3?\s+)?\S*hcom\.pyz?)\s+start\b')
    if '--_hcom_sender' not in command or not start_pattern.search(command):
        return None

    match = re.search(r'--_hcom_sender\s+(\S+)', command)
    if not match:
        return None

    subagent_alias = match.group(1)
    msg = format_subagent_hcom_instructions(subagent_alias)

    return {
        "hookSpecificOutput": {
            "hookEventName": "PostToolUse",
            "additionalContext": msg
        }
    }


def _run_subagent_done_polling(command: str) -> tuple[bool, int, dict[str, Any] | None]:
    """Subagent context: polling loop when subagent runs 'hcom done'

    Returns (matched, exit_code, hook_output_dict | None)
    - matched: whether this is a done command (False = not done polling)
    - exit_code: 0 for timeout/disabled, 0 for message delivery (PostToolUse uses exit 0)
    - output: hook output if messages delivered
    """
    # Extract subagent ID
    done_pattern = re.compile(r'((?:uvx\s+)?hcom|python3?\s+-m\s+hcom|(?:python3?\s+)?\S*hcom\.pyz?)\s+done\b')
    if not done_pattern.search(command) or '--_hcom_sender' not in command:
        return (False, 0, None)

    try:
        tokens = shlex.split(command)
        idx = tokens.index('--_hcom_sender')
        if idx + 1 >= len(tokens):
            return (False, 0, None)
        subagent_id = tokens[idx + 1]
    except (ValueError, IndexError):
        return (False, 0, None)

    # Check if disabled
    instance_data_sub = load_instance_position(subagent_id)
    if not instance_data_sub or not instance_data_sub.get('enabled', False):
        return (True, 0, None)

    # Polling loop
    update_instance_position(subagent_id, {'last_stop': time.time()})
    set_status(subagent_id, 'waiting')

    timeout = get_config().subagent_timeout
    start = time.time()

    while time.time() - start < timeout:
        instance_data_sub = load_instance_position(subagent_id)
        if not instance_data_sub or not instance_data_sub.get('enabled', False):
            return (True, 0, None)

        messages, max_event_id = get_unread_messages(subagent_id, update_position=False)
        # Always update position to mark events as seen (even if filtered out)
        update_instance_position(subagent_id, {'last_event_id': max_event_id})
        if messages:
            formatted = format_hook_messages(messages, subagent_id)
            set_status(subagent_id, 'delivered', messages[-1]['from'])

            output = {
                "hookSpecificOutput": {
                    "hookEventName": "PostToolUse",
                    "additionalContext": formatted
                }
            }
            return (True, 0, output)

        update_instance_position(subagent_id, {'last_stop': time.time()})
        time.sleep(1)

    # Timeout
    update_instance_position(subagent_id, {'enabled': False})
    set_status(subagent_id, 'waiting', 'timeout')
    return (True, 0, None)


def _inject_launch_context_if_needed(instance_name: str, command: str, instance_data: dict[str, Any] | None) -> dict[str, Any] | None:
    """Parent context: inject launch context for help/launch commands

    Returns hook output dict or None.
    """
    # Match all hcom invocation variants (hcom, uvx hcom, python -m hcom, .pyz)
    launch_pattern = re.compile(
        r'((?:uvx\s+)?hcom|python3?\s+-m\s+hcom|(?:python3?\s+)?\S*hcom\.pyz?)\s+'
        r'(?:(?:help|--help|-h)\b|\d+)'
    )
    if not launch_pattern.search(command):
        return None

    if instance_data and instance_data.get('launch_context_announced', False):
        return None

    msg = build_launch_context(instance_name)
    update_instance_position(instance_name, {'launch_context_announced': True})

    return {
        "systemMessage": "[HCOM launch info shown to instance]",
        "hookSpecificOutput": {
            "hookEventName": "PostToolUse",
            "additionalContext": msg
        }
    }


def _check_external_stop_notification(instance_name: str, instance_data: dict[str, Any] | None, command: str) -> dict[str, Any] | None:
    """Parent or subagent context: show notification if externally stopped

    Returns hook output dict or None.
    """
    check_name = instance_name
    check_data = instance_data

    # Subagent override
    if '--_hcom_sender' in command:
        match = re.search(r'--_hcom_sender\s+(\S+)', command)
        if match:
            check_name = match.group(1)
            check_data = load_instance_position(check_name)

    if not check_data or not check_data.get('external_stop_pending'):
        return None

    update_instance_position(check_name, {'external_stop_pending': False})

    if not check_data.get('enabled', False) and check_data.get('previously_enabled', False):
        message = (
            "[HCOM NOTIFICATION]\n"
            "Your HCOM connection has been stopped by an external command.\n"
            "You will no longer receive messages. Stop your current work immediately."
        )
        return {
            "systemMessage": "[hcom stop notification]",
            "hookSpecificOutput": {
                "hookEventName": "PostToolUse",
                "additionalContext": message
            }
        }

    return None


def _inject_bootstrap_if_needed(instance_name: str, instance_data: dict[str, Any] | None) -> dict[str, Any] | None:
    """Parent context: inject bootstrap text if not announced

    Returns hook output dict or None.
    """
    if instance_data and instance_data.get('alias_announced', False):
        return None

    msg = build_hcom_bootstrap_text(instance_name)
    update_instance_position(instance_name, {'alias_announced': True})

    return {
        "systemMessage": "[HCOM info shown to instance]",
        "hookSpecificOutput": {
            "hookEventName": "PostToolUse",
            "additionalContext": msg
        }
    }


def _get_posttooluse_messages(instance_name: str, instance_data: dict[str, Any] | None) -> dict[str, Any] | None:
    """Parent context: check for unread messages
    Returns hook output dict or None.
    """
    if instance_data and in_subagent_context(instance_data):
        return None

    # Skip message delivery if instance is disabled
    if not instance_data or not instance_data.get('enabled', False):
        return None

    messages, _ = get_unread_messages(instance_name, update_position=True)
    if not messages:
        return None

    formatted = format_hook_messages(messages, instance_name)
    set_status(instance_name, 'delivered', messages[0]['from'])

    return {
        "systemMessage": formatted,
        "hookSpecificOutput": {
            "hookEventName": "PostToolUse",
            "additionalContext": formatted
        }
    }


def _combine_posttooluse_outputs(outputs: list[dict[str, Any]]) -> dict[str, Any]:
    """Combine multiple PostToolUse outputs
    Returns combined hook output dict.
    """
    if len(outputs) == 1:
        return outputs[0]

    # Combine systemMessages
    system_msgs = [o.get('systemMessage') for o in outputs if o.get('systemMessage')]
    combined_system = ' + '.join(system_msgs) if system_msgs else None

    # Combine additionalContext with separator
    contexts = [
        o['hookSpecificOutput']['additionalContext']
        for o in outputs
        if 'hookSpecificOutput' in o
    ]
    combined_context = '\n\n---\n\n'.join(contexts)

    result = {
        "hookSpecificOutput": {
            "hookEventName": "PostToolUse",
            "additionalContext": combined_context
        }
    }
    if combined_system:
        result["systemMessage"] = combined_system

    return result


def handle_posttooluse(hook_data: dict[str, Any], instance_name: str) -> None:
    """PostToolUse: Task completion, done polling, context injection, message delivery"""
    tool_name = hook_data.get('tool_name', '')
    tool_input = hook_data.get('tool_input', {})
    tool_response = hook_data.get('tool_response', {})
    instance_data = load_instance_position(instance_name)

    # Task completion (parent context) - exits early
    if tool_name == 'Task':
        if output := _handle_task_completion(instance_name, instance_data, tool_input, tool_response):
            print(json.dumps(output, ensure_ascii=False))
        sys.exit(0)

    # Bash-specific flows
    if tool_name == 'Bash':
        command = tool_input.get('command', '')

        # Subagent 'hcom start' instructions - exits early
        if output := _inject_subagent_hcom_start_instructions(command, instance_data):
            print(json.dumps(output, ensure_ascii=False))
            sys.exit(0)

        # Subagent 'done' polling - exits early
        matched, exit_code, output = _run_subagent_done_polling(command)
        if matched:
            if output:
                print(json.dumps(output, ensure_ascii=False))
            sys.exit(exit_code)

        # Parent flows - collect outputs, don't exit
        outputs_to_combine: list[dict[str, Any]] = []

        # Launch context
        if output := _inject_launch_context_if_needed(instance_name, command, instance_data):
            outputs_to_combine.append(output)

        # Check hcom command pattern
        matches = list(re.finditer(HCOM_COMMAND_PATTERN, command))
        if matches:
            # External stop notification
            if output := _check_external_stop_notification(instance_name, instance_data, command):
                outputs_to_combine.append(output)

            # Bootstrap
            if output := _inject_bootstrap_if_needed(instance_name, instance_data):
                outputs_to_combine.append(output)

    else:
        outputs_to_combine = []

    # Message delivery for ALL tools (parent only)
    if output := _get_posttooluse_messages(instance_name, instance_data):
        outputs_to_combine.append(output)

    # Combine and deliver if any outputs
    if outputs_to_combine:
        combined = _combine_posttooluse_outputs(outputs_to_combine)
        print(json.dumps(combined, ensure_ascii=False))
        sys.exit(0)

    sys.exit(0)


def handle_sessionend(hook_data: dict[str, Any], instance_name: str, updates: dict[str, Any]) -> None:
    """Handle SessionEnd hook - mark session as ended and set final status"""
    reason = hook_data.get('reason', 'unknown')

    # Set session_ended flag to tell Stop hook to exit
    updates['session_ended'] = True

    # Set status to exited with reason as context (reason: clear, logout, prompt_input_exit, other)
    set_status(instance_name, 'exited', reason)

    try:
        update_instance_position(instance_name, updates)
    except Exception as e:
        log_hook_error(f'sessionend:update_instance_position({instance_name})', e)

    # Notify instance to wake and exit cleanly
    notify_instance(instance_name)
