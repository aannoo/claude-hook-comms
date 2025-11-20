"""Shared hook helper functions"""
from __future__ import annotations
from typing import Any
import sys
import time
import re
import shlex
import os
import socket

from ..shared import MAX_MESSAGES_PER_DELIVERY
from ..core.instances import (
    load_instance_position, update_instance_position, set_status,
    initialize_instance_in_position_file
)
from ..core.messages import get_unread_messages, format_hook_messages, should_deliver_message
from ..core.config import get_config
from .utils import is_safe_hcom_command, build_hcom_command, disable_instance, log_hook_error


def cleanup_orphaned_subagents(instance_name: str, instance_data: dict[str, Any] | None) -> None:
    """Parent context: disable orphaned subagent instances.

    CRITICAL: Cannot use parent_session_id FK for vanilla instances:
    - Vanilla instances have session_id=NULL
    - Subagents get parent_session_id=NULL
    - Query "WHERE parent_session_id=NULL" returns empty (NULL != NULL in SQL)
    - FK constraint won't work for vanilla instances

    CRITICAL: Cannot use name prefix (WHERE name LIKE 'parent_%'):
    - Ambiguous when instance names overlap (e.g., "alice" vs "alice_foo")
    - "alice_reviewer" could be child of "alice" OR separate instance

    Uses parent_name field for unambiguous parent→child relationship.
    """
    if not instance_data:
        return

    from ..core.db import get_db
    conn = get_db()

    # Query by parent_name (unambiguous, works for all instances)
    # Only cleanup subagents that aren't already exited (skip historical ones)
    rows = conn.execute(
        "SELECT name, status FROM instances WHERE parent_name = ? AND NOT (status = 'inactive' AND status_context LIKE 'exit:%')",
        (instance_name,)
    ).fetchall()

    for row in rows:
        disable_instance(row['name'], initiated_by=instance_name, reason='orphaned')
        set_status(row['name'], 'inactive', 'exit:orphaned')


def setup_subagent_identity(instance_data: dict[str, Any] | None, instance_name: str, tool_input: dict[str, Any], session_id: str) -> dict[str, Any] | None:
    """Parent context: generate/reuse subagent ID, inject instructions

    Handles: resume lookup, ID generation, file init, prompt injection.
    Returns hook output dict or None.

    IMPORTANT: Defensive guards needed - instance_data may be {} if file missing.
    """
    if not instance_data:
        instance_data = {}

    subagent_type = tool_input.get('subagent_type', 'unknown')
    resume_agent_id = tool_input.get('resume')

    # Resume lookup: query by agent_id
    existing_hcom_id = None
    resume_enabled = None
    if resume_agent_id:
        from ..core.db import get_db
        conn = get_db()
        row = conn.execute(
            "SELECT name, enabled FROM instances WHERE parent_name = ? AND agent_id = ?",
            (instance_name, resume_agent_id)
        ).fetchone()
        if row:
            existing_hcom_id = row['name']
            resume_enabled = bool(row['enabled'])

    if existing_hcom_id:
        # Reuse existing - preserve enabled state
        subagent_id = existing_hcom_id
        from ..core.db import get_instance

        if not get_instance(subagent_id):
            # Preserve enabled state from DB (if they had agent_id, they were already running)
            initialize_instance_in_position_file(subagent_id, parent_session_id=session_id, parent_name=instance_name, enabled=resume_enabled)
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

        # Initialize with full data (disabled - agent must run hcom start)
        if not initialize_instance_in_position_file(subagent_id, parent_session_id=session_id, parent_name=instance_name, enabled=False):
            # Initialization failed - delete placeholder and abort
            log_hook_error(f'pretooluse:initialize_subagent({subagent_id})', Exception('Failed to initialize instance'))
            try:
                conn.execute("DELETE FROM instances WHERE name = ?", (subagent_id,))
                conn.commit()
            except Exception:
                pass
            return None

    # Always inject opt-in hint
    hcom_cmd = build_hcom_command()
    original_prompt = tool_input.get('prompt', '')

    # Different message for resume vs new
    if existing_hcom_id:
        hint = f"[To reconnect to hcom, run this command: '{hcom_cmd} start --_hcom_sender {subagent_id}']"
    else:
        hint = f"[To send and receive messages with hcom, you must first run this command: '{hcom_cmd} start --_hcom_sender {subagent_id}']"

    modified_prompt = f"""{hint}

{original_prompt}"""

    updated_input = tool_input.copy()
    updated_input['prompt'] = modified_prompt

    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "allow",
            "updatedInput": updated_input
        }
    }


def deliver_freeze_messages(instance_name: str, parent_event_id: int) -> dict[str, Any]:
    """Parent context: deliver messages from Task execution period with scope validation"""
    from ..core.db import get_events_since, get_db

    # Query all message events since parent froze (SINGLE query to avoid race condition)
    events = get_events_since(parent_event_id, event_type='message')

    if not events:
        return {'last_event_id': parent_event_id}

    # Determine last_event_id from the events we actually retrieved
    last_id = max(e['id'] for e in events)

    # Get database connection for instance queries
    conn = get_db()

    # Get subagent names and agent IDs using parent_name field (not prefix matching - avoids collision)
    subagent_rows = conn.execute(
        "SELECT name, agent_id FROM instances WHERE parent_name = ?", (instance_name,)
    ).fetchall()
    subagent_names = [row['name'] for row in subagent_rows]
    subagent_agent_ids = [row['agent_id'] for row in subagent_rows if row['agent_id']]

    # Filter events with scope validation
    subagent_msgs = []
    parent_msgs = []

    for event in events:
        event_data = event['data']

        # Validate scope field present
        if 'scope' not in event_data:
            print(
                f"Error: Message event {event['id']} missing 'scope' field (old format). "
                f"Run 'hcom reset logs' to clear old messages.",
                file=sys.stderr
            )
            continue

        sender_name = event_data['from']

        # Build display message dict
        msg = {
            'timestamp': event['timestamp'],
            'from': sender_name,
            'message': event_data['text']
        }

        try:
            # Messages FROM subagents
            if sender_name in subagent_names:
                subagent_msgs.append(msg)
            # Messages TO subagents via scope routing
            elif subagent_names and any(
                should_deliver_message(event_data, name, sender_name) for name in subagent_names
            ):
                if msg not in subagent_msgs:  # Avoid duplicates
                    subagent_msgs.append(msg)
            # Messages TO parent via scope routing
            elif should_deliver_message(event_data, instance_name, sender_name):
                parent_msgs.append(msg)
        except ValueError as e:
            print(
                f"Error: Corrupt message data in event {event['id']}: {e}. "
                f"Run 'hcom reset logs' to clear corrupt messages.",
                file=sys.stderr
            )
            continue

    # Combine and format
    all_relevant = subagent_msgs + parent_msgs
    all_relevant.sort(key=lambda m: m['timestamp'])

    if all_relevant:
        formatted = '\n'.join(f"{msg['from']}: {msg['message']}" for msg in all_relevant)
        # Format as "name (agent_id: xyz)" for correlation
        subagent_list = ', '.join(
            f"{row['name']} (agent_id: {row['agent_id']})" if row['agent_id'] else row['name']
            for row in subagent_rows
        ) if subagent_rows else 'none'
        summary = (
            f"[Task tool completed - Message history during Task tool]\n"
            f"Subagents: {subagent_list}\n"
            f"The following {len(all_relevant)} message(s) occurred:\n\n"
            f"{formatted}\n\n"
            f"[End of message history. Subagents have finished and are no longer active.]"
        )
        return {'summary': summary, 'last_event_id': last_id}

    return {'last_event_id': last_id}


def mark_task_subagents_exited(instance_name: str) -> None:
    """Parent context: mark all subagents exited after Task completion"""
    from ..core.db import get_db
    conn = get_db()
    # Only mark non-exited subagents (skip historical ones)
    subagents = conn.execute(
        "SELECT name FROM instances WHERE parent_name = ? AND NOT (status = 'inactive' AND status_context LIKE 'exit:%')",
        (instance_name,)
    ).fetchall()
    for row in subagents:
        set_status(row['name'], 'inactive', 'exit:task_completed')


def handle_task_completion(instance_name: str, instance_data: dict[str, Any] | None) -> dict[str, Any] | None:
    """Parent context: Task tool completion flow

    Maintains atomic sequencing of state updates:
    1. Deliver freeze messages
    2. Update position
    3. Mark subagents exited

    Returns hook output dict if messages to deliver, else None.
    """
    parent_event_id = instance_data.get('last_event_id', 0) if instance_data else 0

    # Deliver freeze-period messages
    result = deliver_freeze_messages(instance_name, parent_event_id)

    # Update position
    update_instance_position(instance_name, {
        'last_event_id': result['last_event_id']
    })

    # Mark subagents exited
    mark_task_subagents_exited(instance_name)

    # Clear Task context so parent can update status normally again
    set_status(instance_name, 'active', 'tool:task_completed')

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


def _check_claude_alive() -> bool:
    """Check if Claude process still alive (orphan detection)"""
    # Background instances are intentionally detached
    if os.environ.get('HCOM_BACKGROUND') == '1':
        return True
    # stdin closed = Claude Code died
    return not sys.stdin.closed


def _setup_tcp_notification(instance_name: str) -> tuple[socket.socket | None, float, bool]:
    """Setup TCP server for instant wake (shared by parent and subagent)

    Returns (server, poll_timeout, tcp_mode)
    """
    try:
        notify_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        notify_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        notify_server.bind(('127.0.0.1', 0))
        notify_server.listen(128)  # Larger backlog for notification bursts
        notify_server.setblocking(False)
        return (notify_server, 30.0, True)  # 30s select timeout, 0% CPU
    except Exception as e:
        log_hook_error(f'tcp_notification:{instance_name}', e)
        return (None, 0.1, False)  # Fallback: 100ms polling


def poll_messages(
    instance_id: str,
    timeout: int,
    disable_on_timeout: bool = False
) -> tuple[int, dict[str, Any] | None, bool]:
    """Shared message polling for parent Stop and SubagentStop

    Args:
        instance_id: Instance name to poll for
        timeout: Timeout in seconds (wait_timeout for parent, subagent_timeout for subagent)
        disable_on_timeout: Whether to disable instance on timeout (True for subagents)

    Returns:
        (exit_code, hook_output, timed_out)
        - exit_code: 0 for timeout/disabled, 2 for message delivery
        - output: hook output dict if messages delivered
        - timed_out: True if polling timed out
    """
    try:
        instance_data = load_instance_position(instance_id)
        if not instance_data or not instance_data.get('enabled', False):
            if instance_data and not instance_data.get('enabled'):
                set_status(instance_id, 'inactive', 'exit:disabled')
            return (0, None, False)

        # Setup TCP notification (both parent and subagent use it)
        notify_server, poll_timeout, tcp_mode = _setup_tcp_notification(instance_id)

        # Extract notify_port with error handling
        notify_port = None
        if notify_server:
            try:
                notify_port = notify_server.getsockname()[1]
            except Exception:
                # getsockname failed - close socket and fall back to polling
                try:
                    notify_server.close()
                except Exception:
                    pass
                notify_server = None
                tcp_mode = False
                poll_timeout = 0.1

        update_instance_position(instance_id, {
            'notify_port': notify_port,
            'tcp_mode': tcp_mode
        })

        # Set status BEFORE loop (visible immediately)
        update_instance_position(instance_id, {'last_stop': time.time()})
        set_status(instance_id, 'idle')

        start = time.time()

        try:
            while time.time() - start < timeout:
                # Check for disabled/session_ended
                instance_data = load_instance_position(instance_id)
                if not instance_data or not instance_data.get('enabled', False):
                    return (0, None, False)
                if instance_data.get('session_ended'):
                    return (0, None, False)

                # Poll BEFORE select() to catch messages from PostToolUse→Stop transition gap
                # Messages arriving while Stop hook not running already sent (failed) notifications
                # to old/None port. Check immediately on restart to avoid 30s select() timeout.
                messages, max_event_id = get_unread_messages(instance_id, update_position=False)

                if messages:
                    # Orphan detection - don't mark as read if Claude died
                    if not _check_claude_alive():
                        return (0, None, False)

                    # Mark as read and deliver
                    update_instance_position(instance_id, {'last_event_id': max_event_id})

                    # Limit messages (both parent and subagent)
                    messages = messages[:MAX_MESSAGES_PER_DELIVERY]
                    formatted = format_hook_messages(messages, instance_id)
                    set_status(instance_id, 'active', f"deliver:{messages[0]['from']}")

                    output = {
                        "decision": "block",
                        "reason": formatted
                    }
                    return (2, output, False)

                # Calculate remaining time to prevent timeout overshoot
                remaining = timeout - (time.time() - start)
                if remaining <= 0:
                    break

                # Use minimum of remaining time and poll_timeout
                wait_time = min(remaining, poll_timeout)

                # TCP notification or fallback polling
                if notify_server:
                    import select
                    readable, _, _ = select.select([notify_server], [], [], wait_time)
                    if readable:
                        # Drain all pending notifications
                        while True:
                            try:
                                notify_server.accept()[0].close()
                            except BlockingIOError:
                                break
                else:
                    time.sleep(wait_time)

                # Update heartbeat
                update_instance_position(instance_id, {'last_stop': time.time()})

            # Timeout reached
            if disable_on_timeout:
                update_instance_position(instance_id, {'enabled': False})
                set_status(instance_id, 'inactive', 'exit:timeout')
            return (0, None, True)

        finally:
            # Close socket but keep notify_port in DB (stale reference acceptable)
            # Notifications to stale port fail silently (best-effort). Better than None which skips notification.
            # Next Stop cycle updates to new port. Only clear on true exit (disabled/session ended).
            if notify_server:
                try:
                    notify_server.close()
                except Exception:
                    pass

    except Exception:
        # Hook safety - never crash on DB/IO errors
        return (0, None, False)
