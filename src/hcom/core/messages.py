"""Message operations - filtering, routing, and delivery"""
from __future__ import annotations

from .instances import (
    load_instance_position, load_all_positions,
    update_instance_position, is_parent_instance, in_same_group
)
from .config import get_config
from ..shared import MENTION_PATTERN, SENDER

# ==================== Core Message Operations ====================

def unescape_bash(text: str) -> str:
    """Remove bash escape sequences from message content.

    Bash escapes special characters when constructing commands. Since hcom
    receives messages as command arguments, we unescape common sequences
    that don't affect the actual message intent.
    """
    # Common bash escapes that appear in double-quoted strings
    replacements = [
        ('\\!', '!'),   # History expansion
        ('\\$', '$'),   # Variable expansion
        ('\\`', '`'),   # Command substitution
        ('\\"', '"'),   # Double quote
        ("\\'", "'"),   # Single quote (less common in double quotes but possible)
    ]
    for escaped, unescaped in replacements:
        text = text.replace(escaped, unescaped)
    return text

def send_message(from_instance: str, message: str) -> bool:
    """Send a message to the database and notify all instances.

    This function handles both writing to SQLite and sending TCP
    notifications to wake instances for immediate delivery.
    """
    try:
        from .db import log_event

        # Extract recipients from @mentions or default to "all"
        recipients = "all"
        has_mention = False
        if '@' in message:
            mentions = MENTION_PATTERN.findall(message)
            if mentions:
                recipients = mentions
                has_mention = True

        # Log to SQLite
        log_event(
            event_type='message',
            instance=from_instance,
            data={
                'from': from_instance,
                'to': recipients,
                'text': message,
                'mention': has_mention
            }
        )

        # Notify all instances after successful write
        from .runtime import notify_all_instances
        notify_all_instances()

        return True
    except Exception:
        return False


def get_unread_messages(instance_name: str, update_position: bool = False) -> tuple[list[dict[str, str]], int]:
    """Get unread messages for instance with @-mention filtering
    Args:
        instance_name: Name of instance to get messages for
        update_position: If True, mark messages as read by updating position
    Returns:
        Tuple of (messages, max_event_id)
    """
    from .db import get_events_since

    # Get last processed event ID from instance file
    instance_data = load_instance_position(instance_name)
    last_event_id = instance_data.get('last_event_id', 0)

    # Query new message events
    events = get_events_since(last_event_id, event_type='message')

    if not events:
        return [], last_event_id

    # Filter messages:
    # 1. Exclude own messages
    # 2. Apply @-mention filtering
    from .db import get_db
    conn = get_db()
    all_instance_names = [row['name'] for row in conn.execute("SELECT name FROM instances").fetchall()]
    messages = []

    for event in events:
        event_data = event['data']

        # Skip own messages
        if event_data['from'] == instance_name:
            continue

        # Build message dict for filtering
        msg = {
            'timestamp': event['timestamp'],
            'from': event_data['from'],
            'message': event_data['text']
        }

        # Apply existing filtering logic
        if should_deliver_message(msg, instance_name, all_instance_names):
            messages.append(msg)

    # Max event ID from events we processed
    max_event_id = events[-1]['id'] if events else last_event_id

    # Only update position (ie mark as read) if explicitly requested (after successful delivery)
    if update_position:
        update_instance_position(instance_name, {'last_event_id': max_event_id})

    return messages, max_event_id

# ==================== Message Filtering & Routing ====================

def should_deliver_message(msg: dict[str, str], instance_name: str, all_instance_names: list[str] | None = None) -> bool:
    """Check if message should be delivered based on @-mentions and group isolation.
    Group isolation rules:
    - CLI (bigboss) broadcasts → everyone (all parents and subagents)
    - Parent broadcasts → other parents only (subagents shut down during their own parent activity)
    - Subagent broadcasts → same group subagents only (parent frozen during their subagents activity)
    - @-mentions → cross all boundaries like a nice piece of chocolate cake or fried chicken
    """
    text = msg['message']
    sender = msg['from']

    # Load instance data for group membership
    sender_data = load_instance_position(sender)
    receiver_data = load_instance_position(instance_name)

    # Determine if sender/receiver are parents or subagents
    sender_is_parent = is_parent_instance(sender_data)
    receiver_is_parent = is_parent_instance(receiver_data)

    # Check for @-mentions first (crosses all boundaries! yay!)
    if '@' in text:
        mentions = MENTION_PATTERN.findall(text)

        if mentions:
            # Check if this instance matches any mention
            this_instance_matches = any(instance_name.lower().startswith(mention.lower()) for mention in mentions)
            if this_instance_matches:
                return True

            # Check if CLI sender (bigboss) is mentioned
            sender_mentioned = any(SENDER.lower().startswith(mention.lower()) for mention in mentions)

            # Broadcast fallback: no matches anywhere = broadcast with group rules
            if all_instance_names:
                any_mention_matches = any(
                    any(name.lower().startswith(mention.lower()) for name in all_instance_names)
                    for mention in mentions
                ) or sender_mentioned

                if not any_mention_matches:
                    # Fall through to group isolation rules
                    pass
                else:
                    # Mention matches someone else, not us
                    return False
            else:
                # No instance list provided, assume mentions are valid and we're not the target
                return False
        # else: Has @ but no valid mentions, fall through to broadcast rules

    # Special case: CLI sender (bigboss) broadcasts to everyone
    if sender == SENDER:
        return True

    # GROUP ISOLATION for broadcasts
    # Rule 1: Parent → Parent (main communication)
    if sender_is_parent and receiver_is_parent:
        # Different groups = allow (parent-to-parent is the main channel)
        return True

    # Rule 2: Subagent → Subagent (same group only)
    if not sender_is_parent and not receiver_is_parent:
        return in_same_group(sender_data, receiver_data)

    # Rule 3: Parent → Subagent or Subagent → Parent (temporally impossible, filter)
    # This shouldn't happen due to temporal isolation, but filter defensively TODO: consider if better to not filter these as parent could get it after children die - messages can be recieved any time you dont both have to be alive at the same time. like fried chicken.
    return False

def get_subagent_messages(parent_name: str, since_id: int = 0, limit: int = 0) -> tuple[list[dict[str, str]], int, dict[str, int]]:
    """Get messages from/to subagents of parent instance
    Args:
        parent_name: Parent instance name (e.g., 'alice')
        since_id: Event ID to read from (default 0 = all messages)
        limit: Max messages to return (0 = all)
    Returns:
        Tuple of (messages from/to subagents, last_event_id, per_subagent_counts)
        per_subagent_counts: {'alice_reviewer': 2, 'alice_debugger': 0, ...}
    """
    from .db import get_events_since

    # Query all message events since last check
    events = get_events_since(since_id, event_type='message')

    if not events:
        return [], since_id, {}

    # Get all subagent names for this parent using SQL query
    from .db import get_db
    conn = get_db()
    subagent_names = [row['name'] for row in
                      conn.execute("SELECT name FROM instances WHERE parent_name = ?", (parent_name,)).fetchall()]

    # Initialize per-subagent counts
    per_subagent_counts = {name: 0 for name in subagent_names}
    subagent_names_set = set(subagent_names)  # For fast lookup

    # Filter for messages from/to subagents and track per-subagent counts
    subagent_messages = []
    for event in events:
        event_data = event['data']
        sender = event_data['from']

        # Build message dict
        msg = {
            'timestamp': event['timestamp'],
            'from': sender,
            'message': event_data['text']
        }

        # Messages FROM subagents
        if sender in subagent_names_set:
            subagent_messages.append(msg)
            # Track which subagents would receive this message
            for subagent_name in subagent_names:
                if subagent_name != sender and should_deliver_message(msg, subagent_name, subagent_names):
                    per_subagent_counts[subagent_name] += 1
        # Messages TO subagents via @mentions or broadcasts
        elif subagent_names:
            # Check which subagents should receive this message
            matched = False
            for subagent_name in subagent_names:
                if should_deliver_message(msg, subagent_name, subagent_names):
                    if not matched:
                        subagent_messages.append(msg)
                        matched = True
                    per_subagent_counts[subagent_name] += 1

    if limit > 0:
        subagent_messages = subagent_messages[-limit:]

    last_event_id = events[-1]['id'] if events else since_id
    return subagent_messages, last_event_id, per_subagent_counts

# ==================== Message Formatting ====================

def format_hook_messages(messages: list[dict[str, str]], instance_name: str) -> str:
    """Format messages for hook feedback"""
    if len(messages) == 1:
        msg = messages[0]
        reason = f"[new message] {msg['from']} → {instance_name}: {msg['message']}"
    else:
        parts = [f"{msg['from']} → {instance_name}: {msg['message']}" for msg in messages]
        reason = f"[{len(messages)} new messages] | {' | '.join(parts)}"

    # Only append hints to messages
    hints = get_config().hints
    if hints:
        reason = f"{reason} | [{hints}]"

    return reason

__all__ = [
    'unescape_bash',
    'send_message',
    'get_unread_messages',
    'should_deliver_message',
    'get_subagent_messages',
    'format_hook_messages',
]
