"""Hook dispatcher - single entry point with sys.exit"""
from __future__ import annotations
from typing import Any
import sys
import json
import os
import re

from ..core.paths import ensure_hcom_directories
from ..core.instances import load_instance_position, in_subagent_context, get_display_name
from ..core.config import get_config
from ..core.db import get_instance, find_instance_by_session, get_db
from .handlers import (
    handle_pretooluse, handle_posttooluse, handle_stop,
    handle_subagent_stop, handle_userpromptsubmit,
    handle_sessionstart, handle_sessionend, handle_notify,
)
from .utils import init_hook_context, log_hook_error


def should_skip_vanilla_instance(hook_type: str, hook_data: dict) -> bool:
    """
    Returns True if hook should exit early.
    Vanilla instances (never opted in via previously_enabled) exit early unless:
    - PreToolUse (handles auto-approval)
    - HCOM-launched (opted in at launch)
    - UserPromptSubmit with hcom command in prompt (shows preemptive bootstrap)
    """
    # PreToolUse always runs (auto-approval for hcom commands)
    if hook_type == 'pre':
        return False

    # HCOM-launched instances always run (opted in at launch)
    if os.environ.get('HCOM_LAUNCHED') == '1':
        return False

    # Get session_id
    session_id = hook_data.get('session_id', '')
    if not session_id:
        return True  # No identity = skip

    # Check if instance exists
    stored_name = find_instance_by_session(session_id)
    if not stored_name:
        # No instance = never opted in, skip all hooks
        return True

    # Instance exists - check if ever participated
    instance_data = get_instance(stored_name)
    if not instance_data:
        return True  # Shouldn't happen, but defensive

    # Check previously_enabled flag
    if not instance_data.get('previously_enabled', False):
        return True  # Never participated = skip

    return False  # Participated before = run hook


def handle_hook(hook_type: str) -> None:
    """Unified hook handler for all HCOM hooks"""
    hook_data = json.load(sys.stdin)

    if not ensure_hcom_directories():
        log_hook_error('handle_hook', Exception('Failed to create directories'))
        sys.exit(0)

    # Ensure database connection (runs schema/migrations on first use)
    get_db()

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
        # Skip enabled check for PostToolUse when launch context needs to be shown
        # Skip enabled check for PostToolUse in subagent context (need to deliver subagent messages)
        # Skip enabled check for SubagentStop (resolves to parent name, but runs for subagents)
        skip_enabled_check = (
            (hook_type == 'userpromptsubmit' and not instance_data.get('alias_announced', False)) or
            (hook_type == 'post' and not instance_data.get('launch_context_announced', False)) or
            (hook_type == 'post' and in_subagent_context(instance_name)) or
            (hook_type == 'subagent-stop')
        )

        if not skip_enabled_check:
            # Skip vanilla instances (never participated)
            if not instance_data.get('previously_enabled', False):
                sys.exit(0)

            # Skip exited instances - frozen until restart
            status = instance_data.get('status')
            status_context = instance_data.get('status_context', '')
            if status == 'inactive' and status_context.startswith('exit:'):
                # Exception: Allow Stop hook to run when re-enabled (transitions back to 'idle')
                if not (hook_type == 'poll' and instance_data.get('enabled', False)):
                    sys.exit(0)

    match hook_type:
        case 'pre':
            handle_pretooluse(hook_data, instance_name)
        case 'post':
            handle_posttooluse(hook_data, instance_name)
        case 'poll':
            handle_stop(hook_data, instance_name, updates, instance_data)
        case 'subagent-stop':
            handle_subagent_stop(hook_data, instance_name, updates, instance_data)
        case 'notify':
            handle_notify(hook_data, instance_name, updates, instance_data)
        case 'userpromptsubmit':
            handle_userpromptsubmit(hook_data, instance_name, updates, is_matched_resume, instance_data)
        case 'sessionend':
            handle_sessionend(hook_data, instance_name, updates)

    sys.exit(0)
