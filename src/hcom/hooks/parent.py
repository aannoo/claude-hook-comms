"""Parent instance hook implementations"""
from __future__ import annotations
from typing import Any
from pathlib import Path
import sys
import os
import time
import json

from ..shared import HCOM_INVOCATION_PATTERN
from ..core.paths import hcom_path
from ..core.instances import (
    load_instance_position, update_instance_position, set_status,
    in_subagent_context
)
from ..core.config import get_config

from .relay import (
    setup_subagent_identity,
    cleanup_orphaned_subagents,
    handle_task_completion
)

from .utils import (
    build_hcom_bootstrap_text, build_launch_context,
    disable_instance, log_hook_error, notify_instance
)


def update_status(instance_name: str, tool_name: str) -> None:
    """Update parent status"""
    set_status(instance_name, 'active', tool_name)


def setup_task_subagent(hook_data: dict[str, Any], instance_name: str, instance_data: dict[str, Any] | None) -> None:
    """Task tool subagent setup"""
    tool_input = hook_data.get('tool_input', {})
    session_id = hook_data.get('session_id', '')
    if output := setup_subagent_identity(instance_data, instance_name, tool_input, session_id):
        print(json.dumps(output, ensure_ascii=False))
        sys.exit(0)


def stop(hook_data: dict[str, Any], instance_name: str, updates: dict[str, Any], instance_data: dict[str, Any] | None) -> None:
    """Parent Stop: TCP polling loop using shared helper"""
    from .relay import poll_messages

    # Cleanup orphaned subagents
    cleanup_orphaned_subagents(instance_name, instance_data)

    # Use shared polling helper
    wait_timeout = instance_data.get('wait_timeout') if instance_data else None
    timeout = wait_timeout or get_config().timeout

    # Persist effective timeout for observability (hcom list --json, TUI)
    update_instance_position(instance_name, {'wait_timeout': timeout})

    exit_code, output, timed_out = poll_messages(
        instance_name,
        timeout,
        disable_on_timeout=False  # Parents don't auto-disable on timeout
    )

    if output:
        print(json.dumps(output, ensure_ascii=False))

    if timed_out:
        set_status(instance_name, 'exited')

    sys.exit(exit_code)


def task_completion(hook_data: dict[str, Any], instance_name: str) -> None:
    """Task PostToolUse: freeze-period message delivery"""
    instance_data = load_instance_position(instance_name)

    if output := handle_task_completion(instance_name, instance_data):
        print(json.dumps(output, ensure_ascii=False))
    sys.exit(0)


def posttooluse(hook_data: dict[str, Any], instance_name: str, instance_data: dict[str, Any] | None) -> None:
    """Parent PostToolUse: launch context, bootstrap, messages

    This contains all the parent-specific PostToolUse logic from handlers.py
    """
    from ..core.messages import get_unread_messages, format_hook_messages
    from ..shared import HCOM_COMMAND_PATTERN
    import re

    tool_name = hook_data.get('tool_name', '')
    tool_input = hook_data.get('tool_input', {})
    outputs_to_combine: list[dict[str, Any]] = []

    # Bash-specific flows
    if tool_name == 'Bash':
        command = tool_input.get('command', '')

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

    # Message delivery for ALL tools (parent only)
    if output := _get_posttooluse_messages(instance_name, instance_data):
        outputs_to_combine.append(output)

    # Combine and deliver if any outputs
    if outputs_to_combine:
        combined = _combine_posttooluse_outputs(outputs_to_combine)
        print(json.dumps(combined, ensure_ascii=False))
        sys.exit(0)

    sys.exit(0)


def _inject_launch_context_if_needed(instance_name: str, command: str, instance_data: dict[str, Any] | None) -> dict[str, Any] | None:
    """Parent context: inject launch context for help/launch commands

    Returns hook output dict or None.
    """
    # Match all hcom invocation variants (hcom, uvx hcom, python -m hcom, .pyz)
    import re
    launch_pattern = re.compile(
        rf'({HCOM_INVOCATION_PATTERN})\s+'
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
    import re
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
    from ..core.messages import get_unread_messages, format_hook_messages

    if in_subagent_context(instance_name):
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


def userpromptsubmit(hook_data: dict[str, Any], instance_name: str, updates: dict[str, Any], is_matched_resume: bool, instance_data: dict[str, Any] | None) -> None:
    """Parent UserPromptSubmit: timestamp, orphan cleanup, bootstrap"""
    import re

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
        # Only process non-exited subagents (skip historical ones)
        subagents = conn.execute(
            "SELECT name FROM instances WHERE parent_name = ? AND status != 'exited'",
            (instance_name,)
        ).fetchall()
        for row in subagents:
            disable_instance(row['name'], initiated_by=instance_name, reason='orphaned')
            set_status(row['name'], 'exited', 'orphaned')

    # Persist updates (transcript_path, directory, tag, etc.) unconditionally
    update_instance_position(instance_name, updates)

    # Set status to active (user submitted prompt)
    set_status(instance_name, 'active', 'prompt')

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


def notify(hook_data: dict[str, Any], instance_name: str, updates: dict[str, Any], instance_data: dict[str, Any] | None) -> None:
    """Parent Notification: update status to blocked"""
    # Skip status updates when parent is frozen during Task execution
    if in_subagent_context(instance_name):
        return

    message = hook_data.get('message', '')
    if updates:
        update_instance_position(instance_name, updates)
    set_status(instance_name, 'blocked', message)


def sessionend(hook_data: dict[str, Any], instance_name: str, updates: dict[str, Any]) -> None:
    """Parent SessionEnd: mark ended, set final status"""
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
