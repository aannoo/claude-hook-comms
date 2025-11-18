"""Message operations - filtering, routing, and delivery"""
from __future__ import annotations

from .instances import (
    load_instance_position,
    update_instance_position, is_parent_instance
)
from .config import get_config
from ..shared import MENTION_PATTERN, SENDER, CLAUDE_SENDER, SenderIdentity
from .helpers import in_same_group_by_id, validate_scope, get_group_session_id, is_mentioned
import sys

# ==================== Scope Computation ====================

def compute_scope(identity: SenderIdentity, message: str, all_instances: list[str]) -> tuple[str, dict] | None:
    """Compute message scope and routing data.

    Args:
        identity: Sender identity (kind + name + instance_data)
        message: Message text
        all_instances: List of all active instance names (for @mention validation)

    Returns:
        (scope, extra_data) tuple, or None on validation failure

    Scope types:
        - 'broadcast': External senders to everyone
        - 'mentions': @-mention routing (cross all boundaries)
        - 'parent_broadcast': Parent broadcasting to other parents
        - 'subagent_group': Subagent broadcasting to same group

    STRICT FAILURE: Unmatched @mentions return None and print error
    """
    # Check for @mentions FIRST (crosses all boundaries)
    if '@' in message:
        mentions = MENTION_PATTERN.findall(message)
        if mentions:
            # Validate all mentions match real instances
            matched_instances = []
            unmatched = []

            for mention in mentions:
                # Check if mention matches any instance (prefix match, case insensitive)
                matches = [name for name in all_instances
                          if name.lower().startswith(mention.lower())]
                if matches:
                    matched_instances.extend(matches)
                else:
                    unmatched.append(mention)

            # STRICT: fail loud on unmatched mentions
            if unmatched:
                print(
                    f"Error: Unmatched @mentions: {', '.join(f'@{m}' for m in unmatched)}",
                    file=sys.stderr
                )
                # Show available instances (up to 10)
                available = all_instances[:10]
                if all_instances:
                    more = f" (and {len(all_instances) - 10} more)" if len(all_instances) > 10 else ""
                    print(f"Available instances: {', '.join(available)}{more}", file=sys.stderr)
                return None

            # Deduplicate matched instances
            unique_instances = list(dict.fromkeys(matched_instances))
            return ('mentions', {'mentioned_instances': unique_instances})

    # External senders broadcast (no mentions)
    if identity.broadcasts:
        return ('broadcast', {})

    # No mentions - determine broadcast scope
    if not identity.instance_data:
        # Shouldn't happen for kind='instance', but handle gracefully
        return ('broadcast', {})

    # Check if sender is parent or subagent
    if is_parent_instance(identity.instance_data):
        return ('parent_broadcast', {})
    else:
        # Subagent - broadcast within group only
        group_id = identity.group_id
        if not group_id:
            # Subagent without group? Shouldn't happen, fail loud
            print(f"Error: Subagent '{identity.name}' has no group_id", file=sys.stderr)
            return None
        return ('subagent_group', {'group_id': group_id})


def _should_deliver(scope: str, extra: dict, receiver_name: str, sender_name: str) -> bool:
    """Check if message should be delivered to receiver based on scope.

    Internal helper for computing recipient snapshots at send time.

    Args:
        scope: Message scope ('broadcast', 'mentions', 'parent_broadcast', 'subagent_group')
        extra: Extra scope data (mentioned_instances, group_id, etc)
        receiver_name: Instance to check delivery for
        sender_name: Sender name (excluded from delivery)

    Returns:
        True if receiver should get the message
    """
    # Never deliver to self
    if receiver_name == sender_name:
        return False

    # Validate scope
    validate_scope(scope)

    # Match on scope
    if scope == 'broadcast':
        return True

    if scope == 'mentions':
        return receiver_name in extra.get('mentioned_instances', [])

    if scope == 'parent_broadcast':
        receiver_data = load_instance_position(receiver_name)
        return is_parent_instance(receiver_data)

    if scope == 'subagent_group':
        group_id = extra.get('group_id')
        if not group_id:
            return False
        receiver_data = load_instance_position(receiver_name)
        return in_same_group_by_id(group_id, receiver_data)

    # Should never reach here if validate_scope works
    return False

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

def send_message(identity: SenderIdentity, message: str) -> list[str] | None:
    """Send a message to the database and notify all instances.

    Args:
        identity: Sender identity (kind + name + instance_data)
        message: Message text

    Returns:
        List of recipient names on success, None on failure (unmatched @mentions or DB error)
    """
    from .db import log_event, get_db

    # Get real instances for recipient computation
    conn = get_db()
    real_instances = [
        row['name'] for row in
        conn.execute("SELECT name FROM instances WHERE previously_enabled = 1").fetchall()
    ]

    # For @mention validation: include CLI/TUI identities as valid targets
    mentionable = real_instances + [SENDER, CLAUDE_SENDER]

    # Compute scope and routing data
    scope_result = compute_scope(identity, message, mentionable)
    if scope_result is None:
        # Failed validation (unmatched @mentions) - error already printed
        return None

    scope, extra = scope_result

    # Compute recipient snapshot using only real instances
    recipients = [
        inst_name for inst_name in real_instances
        if _should_deliver(scope, extra, inst_name, identity.name)
    ]

    # Build event data
    data = {
        'from': identity.name,          # Display name for output
        'sender_kind': identity.kind,   # 'external' or 'instance' for filtering
        'scope': scope,                 # Routing scope
        'text': message,
        'recipients': recipients        # Snapshot for read receipts
    }

    # Add scope extra data
    if extra:
        data.update(extra)

    # Log to SQLite with namespace separation
    # External senders use 'ext_{name}' prefix for clear namespace isolation
    # System senders use 'sys_{name}' prefix (e.g., sys_[hcom-launcher])
    # Instance senders use real instance name
    # Actual sender name preserved in data['from'] for display
    if identity.kind == 'external':
        routing_instance = f'ext_{identity.name}'
    elif identity.kind == 'system':
        routing_instance = f'sys_{identity.name}'
    else:
        routing_instance = identity.name

    try:
        log_event(
            event_type='message',
            instance=routing_instance,
            data=data
        )
    except Exception as e:
        # DB write failed - print clear error for debugging
        print(f"Error: Failed to write message to database: {e}", file=sys.stderr)
        return None

    # Notify all instances after successful write
    from .runtime import notify_all_instances
    notify_all_instances()

    return recipients


def send_system_message(sender_name: str, message: str) -> list[str] | None:
    """Send a system notification message.

    Args:
        sender_name: System sender identifier (e.g., 'hcom-launcher', 'hcom-watchdog')
        message: Message text (can include @mentions for targeting)

    Returns:
        List of recipient names on success, None on failure
    """
    identity = SenderIdentity(kind='system', name=sender_name, instance_data=None)
    return send_message(identity, message)


def get_unread_messages(instance_name: str, update_position: bool = False) -> tuple[list[dict[str, str]], int]:
    """Get unread messages for instance with scope-based filtering
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

    messages = []
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

        # Skip own messages
        sender_name = event_data['from']
        if sender_name == instance_name:
            continue

        # Apply scope-based filtering
        try:
            if should_deliver_message(event_data, instance_name, sender_name):
                messages.append({
                    'timestamp': event['timestamp'],
                    'from': sender_name,
                    'message': event_data['text']
                })
        except ValueError as e:
            print(
                f"Error: Corrupt message data in event {event['id']}: {e}. "
                f"Run 'hcom reset logs' to clear corrupt messages.",
                file=sys.stderr
            )
            continue

    # Max event ID from events we processed
    max_event_id = events[-1]['id'] if events else last_event_id

    # Only update position (ie mark as read) if explicitly requested (after successful delivery)
    if update_position:
        update_instance_position(instance_name, {'last_event_id': max_event_id})

    return messages, max_event_id

# ==================== Message Filtering & Routing ====================

def should_deliver_message(event_data: dict, receiver_name: str, sender_name: str) -> bool:
    """Check if message should be delivered based on scope (15-line implementation).

    Args:
        event_data: Message event data with 'scope' field and scope-specific data
        receiver_name: Instance to check delivery for
        sender_name: Sender name (excluded from delivery)

    Returns:
        True if receiver should get the message

    Raises:
        KeyError: If scope field missing (old message format)
        ValueError: If scope value invalid
    """
    # Never deliver to self
    if receiver_name == sender_name:
        return False

    # Extract and validate scope
    if 'scope' not in event_data:
        raise KeyError(f"Message missing 'scope' field (old format)")

    scope = event_data['scope']
    validate_scope(scope)  # Raises ValueError if invalid

    # Scope-based routing
    if scope == 'broadcast':
        return True

    if scope == 'mentions':
        return receiver_name in event_data.get('mentioned_instances', [])

    if scope == 'parent_broadcast':
        receiver_data = load_instance_position(receiver_name)
        return is_parent_instance(receiver_data)

    if scope == 'subagent_group':
        group_id = event_data.get('group_id')
        if not group_id:
            return False
        receiver_data = load_instance_position(receiver_name)
        return in_same_group_by_id(group_id, receiver_data)

    # Should never reach here if validate_scope works
    return False


# Note: determine_message_recipients() removed - obsolete after scope refactor
# Use compute_scope() + _should_deliver() directly instead (see send_message() or get_recipient_feedback())


def get_subagent_messages(parent_name: str, since_id: int = 0, limit: int = 0) -> tuple[list[dict[str, str]], int, dict[str, int]]:
    """Get messages from/to subagents of parent instance with scope-based filtering
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

        # Validate scope field present
        if 'scope' not in event_data:
            print(
                f"Error: Message event {event['id']} missing 'scope' field (old format). "
                f"Run 'hcom reset logs' to clear old messages.",
                file=sys.stderr
            )
            continue

        sender_name = event_data['from']

        # Build message dict
        msg = {
            'timestamp': event['timestamp'],
            'from': sender_name,
            'message': event_data['text']
        }

        # Messages FROM subagents
        if sender_name in subagent_names_set:
            subagent_messages.append(msg)
            # Track which subagents would receive this message
            for subagent_name in subagent_names:
                if subagent_name != sender_name:
                    try:
                        if should_deliver_message(event_data, subagent_name, sender_name):
                            per_subagent_counts[subagent_name] += 1
                    except ValueError as e:
                        print(
                            f"Error: Corrupt message data in event {event['id']}: {e}. "
                            f"Run 'hcom reset logs' to clear corrupt messages.",
                            file=sys.stderr
                        )
                        continue
        # Messages TO subagents via @mentions or broadcasts
        elif subagent_names:
            # Check which subagents should receive this message
            matched = False
            for subagent_name in subagent_names:
                try:
                    if should_deliver_message(event_data, subagent_name, sender_name):
                        if not matched:
                            subagent_messages.append(msg)
                            matched = True
                        per_subagent_counts[subagent_name] += 1
                except ValueError as e:
                    print(
                        f"Error: Corrupt message data in event {event['id']}: {e}. "
                        f"Run 'hcom reset logs' to clear corrupt messages.",
                        file=sys.stderr
                    )
                    break  # Skip remaining subagents for this message

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

def get_read_receipts(identity: SenderIdentity, max_text_length: int = 50, limit: int = None) -> list[dict]:
    """Get read receipts for messages sent by sender.
    Args:
        identity: SenderIdentity for the sender (external or instance)
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

    # Determine storage name: external senders use ext_ prefix, instances use real name
    storage_name = f'ext_{identity.name}' if identity.kind == 'external' else identity.name

    # Query by storage name
    query = """
        SELECT e.id, e.timestamp, e.data
        FROM events e
        WHERE e.type = 'message'
          AND e.instance = ?
        ORDER BY e.id DESC
    """

    if limit is not None:
        query += f" LIMIT {int(limit)}"

    sent_messages = conn.execute(query, (storage_name,)).fetchall()

    if not sent_messages:
        return []

    # Get all active instances (previously_enabled = True)
    active_instances_query = """
        SELECT name, last_event_id, session_id, mapid, parent_session_id
        FROM instances
        WHERE previously_enabled = 1 AND name != ?
    """
    active_instances = conn.execute(active_instances_query, (identity.name,)).fetchall()

    if not active_instances:
        return []

    instance_reads = {row['name']: row['last_event_id'] for row in active_instances}
    instance_data_cache = {row['name']: {'session_id': row['session_id'], 'mapid': row['mapid'], 'parent_session_id': row['parent_session_id']} for row in active_instances}
    receipts = []
    now = datetime.now(timezone.utc)

    for msg_row in sent_messages:
        msg_id = msg_row['id']
        msg_timestamp = msg_row['timestamp']
        msg_data = json.loads(msg_row['data'])
        msg_text = msg_data['text']

        # Validate scope field present (skip old messages)
        if 'scope' not in msg_data:
            continue

        # Use snapshotted recipients from send time
        if 'recipients' not in msg_data:
            continue

        recipients = msg_data['recipients']

        # Find recipients that HAVE read this message
        read_by = []
        for inst_name in recipients:
            # Check if instance has advanced past this message
            if instance_reads.get(inst_name, 0) >= msg_id:
                # For external senders, only mark as read if they were @mentioned
                # (they only receive messages they're mentioned in)
                inst_data = instance_data_cache.get(inst_name)
                if inst_data:
                    from ..core.instances import is_external_sender
                    if is_external_sender(inst_data):
                        # Only count as read if @mentioned
                        if not is_mentioned(msg_text, inst_name):
                            continue
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
    'compute_scope',
    '_should_deliver',
    'unescape_bash',
    'send_message',
    'get_unread_messages',
    'should_deliver_message',
    'get_subagent_messages',
    'format_hook_messages',
    'get_read_receipts',
]
