"""Helper functions for scope-based message routing"""
from __future__ import annotations

import re
from typing import Any

# Valid scope values for message routing
VALID_SCOPES = {'broadcast', 'mentions', 'parent_broadcast', 'subagent_group'}

def get_group_session_id(instance_data: dict[str, Any] | None) -> str | None:
    """Get the session_id that defines this instance's group.
    For parents: their own session_id, for subagents: parent_session_id
    """
    if not instance_data:
        return None
    # Subagent - use parent_session_id
    if parent_sid := instance_data.get('parent_session_id'):
        return parent_sid
    # Parent - use own session_id
    return instance_data.get('session_id')

def in_same_group_by_id(group_id: str | None, receiver_data: dict[str, Any] | None) -> bool:
    """Check if receiver is in the same group as the given group_id.

    Args:
        group_id: The sender's group ID (session_id)
        receiver_data: Receiver instance data

    Returns:
        True if receiver is in same group, False otherwise
    """
    if not group_id or not receiver_data:
        return False
    receiver_group = get_group_session_id(receiver_data)
    if not receiver_group:
        return False
    return group_id == receiver_group

def validate_scope(scope: str) -> None:
    """Validate that scope is a valid value.

    Args:
        scope: Scope value to validate

    Raises:
        ValueError: If scope is not in VALID_SCOPES
    """
    if scope not in VALID_SCOPES:
        raise ValueError(
            f"Invalid scope '{scope}'. Must be one of: {', '.join(sorted(VALID_SCOPES))}"
        )

def is_mentioned(text: str, name: str) -> bool:
    """Check if instance name is @-mentioned in text using prefix matching.

    Uses same prefix matching logic as compute_scope() for consistency.
    This allows @api to match instances like "api-worker-1" and "api-worker-2".

    Args:
        text: Text to search in
        name: Instance name to look for (without @ prefix)

    Returns:
        True if @mention prefix-matches name, False otherwise

    Examples:
        >>> is_mentioned("Hey @john, can you help?", "john")
        True
        >>> is_mentioned("Hey @john, can you help?", "johnsmith")  # prefix match
        True
        >>> is_mentioned("Hey @johnsmith, test", "john")  # doesn't match
        False
        >>> is_mentioned("email@john.com", "john")  # not a mention
        False
    """
    from ..shared import MENTION_PATTERN

    # Extract all @mentions from text
    mentions = MENTION_PATTERN.findall(text)

    # Check if any mention prefix-matches the instance name (case-insensitive)
    for mention in mentions:
        if name.lower().startswith(mention.lower()):
            return True

    return False

__all__ = [
    'VALID_SCOPES',
    'get_group_session_id',
    'in_same_group_by_id',
    'validate_scope',
    'is_mentioned',
]
