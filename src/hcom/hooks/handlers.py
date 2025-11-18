"""Hook handler functions - Pure routing layer"""
from __future__ import annotations
from typing import Any
import sys
import os
import json
import re

from ..shared import HCOM_COMMAND_PATTERN
from ..core.instances import (
    load_instance_position, update_instance_position,
    in_subagent_context
)
from ..core.config import get_config

from .utils import (
    is_safe_hcom_command,
    build_hcom_bootstrap_text,
    build_hcom_command,
    notify_instance
)

# Import parent and subagent modules
from . import parent, subagent

# Import identity helpers from core
from ..core.instances import initialize_instance_in_position_file


def handle_pretooluse(hook_data: dict[str, Any], instance_name: str) -> None:
    """Handle PreToolUse hook - status tracking, auto-approve, subagent setup

    Status tracking is context-specific, but auto-approve and Task setup always run.
    """
    instance_data = load_instance_position(instance_name)
    tool_name = hook_data.get('tool_name', '')

    # 1. Status tracking (context-specific)
    if instance_data and instance_data.get('enabled', False):
        # Skip status updates when parent is frozen during Task execution
        if not in_subagent_context(instance_name):
            has_sender_flag = False
            if tool_name == 'Bash':
                command = hook_data.get('tool_input', {}).get('command', '')
                has_sender_flag = '--_hcom_sender' in command

            if not has_sender_flag:
                # Only update parent status (individual semantics - no group updates)
                parent.update_status(instance_name, tool_name)

    # 2. Auto-approve hcom commands (always runs - needed for subagent hcom commands)
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

    # 3. Task setup (always runs - parents create subagents)
    if tool_name == 'Task':
        parent.setup_task_subagent(hook_data, instance_name, instance_data)


def handle_posttooluse(hook_data: dict[str, Any], instance_name: str) -> None:
    """PostToolUse: Route based on tool and command"""
    tool_name = hook_data.get('tool_name', '')

    # Task completion - always parent
    if tool_name == 'Task':
        parent.task_completion(hook_data, instance_name)
        return

    # Load instance_data once for routing decisions
    instance_data = load_instance_position(instance_name)

    # Subagent-specific Bash commands (only if in subagent context)
    # Default parent (instance_data already loaded)
    parent.posttooluse(hook_data, instance_name, instance_data)


def handle_stop(_hook_data: dict[str, Any], instance_name: str, _updates: dict[str, Any], instance_data: dict[str, Any] | None) -> None:
    """Handle Stop hook: parent only (subagents use SubagentStop)"""
    parent.stop(_hook_data, instance_name, _updates, instance_data)


def handle_subagent_stop(_hook_data: dict[str, Any], parent_name: str, _updates: dict[str, Any], _instance_data: dict[str, Any] | None) -> None:
    """SubagentStop: Subagent message polling"""
    subagent.subagent_stop(_hook_data, parent_name, _updates, _instance_data)


def handle_notify(hook_data: dict[str, Any], instance_name: str, updates: dict[str, Any], instance_data: dict[str, Any] | None) -> None:
    """Handle Notification hook - filter generic messages, then route"""
    message = hook_data.get('message', '')

    # Filter generic "waiting for input" when already waiting
    if message == "Claude is waiting for your input":
        current_status = instance_data.get('status', '') if instance_data else ''
        if current_status == 'waiting':
            return  # Instance is idle, Stop hook will maintain waiting status

    # Individual semantics - all instances use same notify handler
    parent.notify(hook_data, instance_name, updates, instance_data)


def handle_userpromptsubmit(hook_data: dict[str, Any], instance_name: str, updates: dict[str, Any], is_matched_resume: bool, instance_data: dict[str, Any] | None) -> None:
    """Handle UserPromptSubmit hook - parent only (orphan cleanup)"""
    parent.userpromptsubmit(hook_data, instance_name, updates, is_matched_resume, instance_data)


def handle_sessionstart(hook_data: dict[str, Any]) -> None:
    """Handle SessionStart hook - write session ID to env file & show initial msg"""
    # Write session ID to CLAUDE_ENV_FILE for automatic identity resolution
    session_id = hook_data.get('session_id')
    env_file = os.environ.get('CLAUDE_ENV_FILE')

    if session_id and env_file:
        try:
            with open(env_file, 'a', newline='\n') as f:
                f.write(f'\nexport HCOM_SESSION_ID={session_id}\n')
        except Exception:
            # Fail silently - hook safety
            pass

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


def handle_sessionend(hook_data: dict[str, Any], instance_name: str, updates: dict[str, Any]) -> None:
    """Handle SessionEnd hook - parent only"""
    parent.sessionend(hook_data, instance_name, updates)
