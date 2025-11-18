"""Subagent context hook implementations"""
from __future__ import annotations
from typing import Any
import sys
import json
import re

from ..core.instances import (
    load_instance_position, update_instance_position, set_status
)
from ..core.config import get_config


def extract_subagent_id_from_transcript(transcript_path: str) -> str | None:
    """Parse agent transcript for --_hcom_sender flag from hcom start command

    Subagents must run: hcom start --_hcom_sender <alias>
    This pattern appears in their transcript when they execute the command.
    """
    import os

    expanded = os.path.expanduser(transcript_path)
    # Match --_hcom_sender followed by the alias
    pattern = re.compile(r'--_hcom_sender\s+(\S+)')

    try:
        with open(expanded, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line)

                    # Check message.content for tool_use (actual Claude Code transcript format)
                    if 'message' in entry:
                        content = entry['message'].get('content', [])
                        if isinstance(content, list):
                            for block in content:
                                if isinstance(block, dict) and block.get('type') == 'tool_use':
                                    if block.get('name') == 'Bash':
                                        command = block.get('input', {}).get('command', '')
                                        if '--_hcom_sender' in command:
                                            match = pattern.search(command)
                                            if match:
                                                return match.group(1)
                except (json.JSONDecodeError, KeyError, TypeError):
                    continue
        return None
    except Exception:
        return None


def subagent_stop(hook_data: dict[str, Any], parent_name: str, _updates: dict[str, Any], _instance_data: dict[str, Any] | None) -> None:
    """SubagentStop: Message polling using agent_id"""
    from ..core.db import get_db

    # Extract agent_id
    agent_id = hook_data.get('agent_id')
    if not agent_id:
        sys.exit(0)

    # Optimization: Check if parent has any enabled subagents before parsing
    conn = get_db()
    has_subagents = conn.execute(
        "SELECT 1 FROM instances WHERE parent_name = ? AND enabled = 1 LIMIT 1",
        (parent_name,)
    ).fetchone()

    if not has_subagents:
        sys.exit(0)  # Parent has no enabled subagents - skip

    # Query for existing subagent with this agent_id (O(1) indexed lookup)
    row = conn.execute(
        "SELECT name FROM instances WHERE parent_name = ? AND agent_id = ?",
        (parent_name, agent_id)
    ).fetchone()

    if row:
        # Found cached - no transcript parsing needed
        subagent_id = row['name']
    else:
        # First SubagentStop fire - parse transcript for hcom start command
        transcript_path = hook_data.get('agent_transcript_path')
        if not transcript_path:
            sys.exit(0)

        subagent_id = extract_subagent_id_from_transcript(transcript_path)
        if not subagent_id:
            sys.exit(0)  # Agent didn't run hcom start - not opted in

        # Store agent_id and transcript_path in subagent's own record (cache for next time)
        update_instance_position(subagent_id, {
            'agent_id': agent_id,
            'transcript_path': transcript_path
        })

    # Poll messages using shared helper
    timeout = get_config().subagent_timeout
    from .relay import poll_messages
    exit_code, output, _ = poll_messages(
        subagent_id,
        timeout,
        disable_on_timeout=True  # Subagents die on timeout
    )

    if output:
        print(json.dumps(output, ensure_ascii=False), file=sys.stderr)

    sys.exit(exit_code)


