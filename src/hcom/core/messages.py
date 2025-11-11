"""Message operations - filtering, routing, and delivery"""
from __future__ import annotations

from .instances import (
    load_instance_position,
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

    NOTE: We do NOT unescape '\\\\' to '\\'. If double backslashes survived
    bash processing, the user intended them (e.g., Windows paths, regex, JSON).
    Unescaping would corrupt legitimate data.
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

        # Snapshot actual recipients at send time for accurate read receipts
        actual_recipients = determine_message_recipients(message, from_instance)

        # Log to SQLite
        log_event(
            event_type='message',
            instance=from_instance,
            data={
                'from': from_instance,
                'to': recipients,
                'text': message,
                'mention': has_mention,
                'recipients': actual_recipients
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

    # Rule 3: Parent ↔ Subagent (allow - delivered via PostToolUse)
    # Parent receives subagent messages after Task completes (PostToolUse delivers freeze-period history)
    # Subagents can receive parent messages if sent while subagent still running
    # Temporal isolation is handled by delivery mechanism (PostToolUse), not filtering
    if sender_is_parent or receiver_is_parent:
        return in_same_group(sender_data, receiver_data)

    # Fallback: should not reach here
    return False


def determine_message_recipients(message_text: str, sender_name: str) -> list[str]:
    """Determine which instances will receive a message using actual delivery logic.
    Args:
        message_text: The message content (for @mention parsing)
        sender_name: Name of sender (excluded from recipients)
    Returns:
        List of instance names that will receive the message
    """
    from .db import get_db

    conn = get_db()

    # Get all instances that could potentially receive (previously_enabled = 1)
    # Exclude sender from recipient list
    all_instances = conn.execute(
        "SELECT name FROM instances WHERE previously_enabled = 1 AND name != ?",
        (sender_name,)
    ).fetchall()

    if not all_instances:
        return []

    # Get all instance names for @mention validation
    all_instance_names = [row['name'] for row in all_instances]

    # Build message dict for should_deliver_message()
    msg = {
        'from': sender_name,
        'message': message_text
    }

    # Filter instances using actual delivery logic
    recipients = []
    for row in all_instances:
        instance_name = row['name']
        if should_deliver_message(msg, instance_name, all_instance_names):
            recipients.append(instance_name)

    return recipients


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

def get_read_receipts(instance_name: str, max_text_length: int = 50, limit: int = None) -> list[dict]:
    """Get read receipts for messages sent by instance.
    Args:
        instance_name: Name of instance to check sent messages for
        max_text_length: Maximum text length before truncation (default 50)
        limit: Maximum number of recent messages to return (default None = all)
    Returns:
        List of dicts with keys: id, age, text, read_by, total_recipients
    """
    from .db import get_db
    from ..shared import format_age
    from datetime import datetime, timezone
    import json

    conn = get_db()

    # Get messages sent by this instance (most recent first)
    query = """
        SELECT e.id, e.timestamp, e.data
        FROM events e
        WHERE e.type = 'message'
          AND e.instance = ?
        ORDER BY e.id DESC
    """
    if limit is not None:
        query += f" LIMIT {int(limit)}"

    sent_messages = conn.execute(query, (instance_name,)).fetchall()

    if not sent_messages:
        return []

    # Get all active instances (previously_enabled = True)
    active_instances_query = """
        SELECT name, last_event_id
        FROM instances
        WHERE previously_enabled = 1 AND name != ?
    """
    active_instances = conn.execute(active_instances_query, (instance_name,)).fetchall()

    if not active_instances:
        return []

    instance_reads = {row['name']: row['last_event_id'] for row in active_instances}
    receipts = []
    now = datetime.now(timezone.utc)

    for msg_row in sent_messages:
        msg_id = msg_row['id']
        msg_timestamp = msg_row['timestamp']
        msg_data = json.loads(msg_row['data'])
        msg_text = msg_data['text']

        # Use snapshotted recipients from send time
        if 'recipients' not in msg_data:
            continue

        recipients = msg_data['recipients']

        # Find recipients that HAVE read this message
        read_by = []
        for inst_name in recipients:
            if instance_reads.get(inst_name, 0) >= msg_id:
                read_by.append(inst_name)

        total_recipients = len(recipients)

        # Only include if there are recipients
        if total_recipients > 0:
            # Calculate age
            try:
                msg_time = datetime.fromisoformat(msg_timestamp.replace('Z', '+00:00'))
                age_seconds = (now - msg_time).total_seconds()
                age_str = format_age(age_seconds)
            except (ValueError, AttributeError):
                age_str = "?"

            # Truncate text
            if len(msg_text) > max_text_length:
                truncated_text = msg_text[:max_text_length - 3] + "..."
            else:
                truncated_text = msg_text

            receipts.append({
                'id': msg_id,
                'age': age_str,
                'text': truncated_text,
                'read_by': read_by,
                'total_recipients': total_recipients
            })

    return receipts

__all__ = [
    'unescape_bash',
    'send_message',
    'get_unread_messages',
    'should_deliver_message',
    'determine_message_recipients',
    'get_subagent_messages',
    'format_hook_messages',
    'get_read_receipts',
]
