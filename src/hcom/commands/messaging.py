"""Messaging commands for HCOM"""
import os
import sys
from .utils import format_error, validate_message, resolve_identity
from ..shared import MENTION_PATTERN, SENDER, CLAUDE_SENDER, MAX_MESSAGES_PER_DELIVERY, IS_WINDOWS
from ..core.config import get_config
from ..core.paths import ensure_hcom_directories
from ..core.db import init_db
from ..core.instances import load_instance_position, in_subagent_context, set_status, get_instance_status
from ..core.messages import unescape_bash, send_message, get_unread_messages, format_hook_messages, determine_message_recipients


def get_recipient_feedback(message: str, sender_name: str = None) -> str:
    """Get formatted recipient feedback showing who will receive the message.
    Using actual delivery logic including group isolation rules.
    Args:
        message: The message text (to extract @mentions)
        sender_name: Optional sender name to exclude from recipients
    Returns:
        Formatted string like "Sent to ⊙ alice, ◉ bob" or "Sent to 15 instances"
    """
    from ..shared import STATUS_ICONS

    # Use sender or fallback to SENDER for CLI context
    actual_sender = sender_name if sender_name else SENDER

    # Get recipients using actual delivery logic
    recipient_names = determine_message_recipients(message, actual_sender)

    # Format recipient feedback
    if len(recipient_names) > 10:
        return f"Sent to: {len(recipient_names)} instances"
    else:
        recipient_status = []
        for r_name in recipient_names:
            r_data = load_instance_position(r_name)
            # Always add recipient (we determined they'll receive it)
            # Use fallback icon if data unavailable
            if r_data:
                _, status, _, _, _ = get_instance_status(r_data)
                icon = STATUS_ICONS.get(status, '◦')
            else:
                icon = '◦'  # Unknown status fallback
            recipient_status.append(f"{icon} {r_name}")
        return f"Sent to: {', '.join(recipient_status)}" if recipient_status else "Message sent"


def cmd_send(argv: list[str], quiet: bool = False) -> int:
    """Send message to hcom: hcom send "message" [--_hcom_session ID] [--_hcom_sender NAME]"""
    from ..core.instances import initialize_instance_in_position_file

    if not ensure_hcom_directories():
        print(format_error("Failed to create HCOM directories"), file=sys.stderr)
        return 1

    init_db()

    # Parse flags
    subagent_id = None
    custom_sender = None

    # Extract --_hcom_sender if present (for subagents)
    if '--_hcom_sender' in argv:
        idx = argv.index('--_hcom_sender')
        if idx + 1 < len(argv):
            subagent_id = argv[idx + 1]
            argv = argv[:idx] + argv[idx + 2:]

    # Extract --from if present (for custom external sender)
    if '--from' in argv:
        idx = argv.index('--from')
        if idx + 1 < len(argv):
            custom_sender = argv[idx + 1]
            # Validate
            if '|' in custom_sender:
                print(format_error("Sender name cannot contain '|'"), file=sys.stderr)
                return 1
            if len(custom_sender) > 50:
                print(format_error("Sender name too long (max 50 chars)"), file=sys.stderr)
                return 1
            if not custom_sender or not all(c.isalnum() or c in '-_' for c in custom_sender):
                print(format_error("Sender name must be alphanumeric with hyphens/underscores"), file=sys.stderr)
                return 1
            argv = argv[:idx] + argv[idx + 2:]
        else:
            print(format_error("--from requires a sender name"), file=sys.stderr)
            return 1

    # First non-flag argument is the message
    message = unescape_bash(argv[0]) if argv else None

    # Check message provided
    if not message:
        print(format_error("No message provided"), file=sys.stderr)
        return 1

    # Validate message
    error = validate_message(message)
    if error:
        print(error, file=sys.stderr)
        return 1

    # Check for unmatched mentions (minimal warning)
    mentions = MENTION_PATTERN.findall(message)
    if mentions:
        try:
            from ..core.db import get_db
            conn = get_db()
            all_instances = [row['name'] for row in conn.execute("SELECT name FROM instances").fetchall()]
            all_names = all_instances + [SENDER]
            unmatched = [m for m in mentions
                        if not any(name.lower().startswith(m.lower()) for name in all_names)]
            if unmatched:
                print(f"Note: @{', @'.join(unmatched)} don't match any instances - broadcasting to all", file=sys.stderr)
        except Exception:
            pass

    # Resolve identity
    try:
        sender_name = resolve_identity(subagent_id)
    except ValueError as e:
        print(format_error(str(e)), file=sys.stderr)
        return 1

    # Handle CLAUDE_SENDER fallback (no identity available)
    if sender_name == CLAUDE_SENDER:
        if IS_WINDOWS:
            print("⚠️ No identity. Use 'hcom <n>' or Windows Terminal.", file=sys.stderr)
        else:
            print("⚠️ No identity. Launch via 'hcom <n>' for stable identity.", file=sys.stderr)
        if not send_message(CLAUDE_SENDER, message):
            print(format_error("Failed to send"), file=sys.stderr)
            return 1
        if not quiet:
            print(f"✓ Sent from {CLAUDE_SENDER}")
        return 0

    # Handle SENDER (CLI context)
    if sender_name == SENDER:
        sender_name = custom_sender if custom_sender else SENDER
        if not send_message(sender_name, message):
            print(format_error("Failed to send message"), file=sys.stderr)
            return 1
        if not quiet:
            print(get_recipient_feedback(message, sender_name))
        return 0

    # Instance context - load data and check state
    instance_data = load_instance_position(sender_name)
    if not instance_data:
        # Initialize if first use (shouldn't happen with _resolve_identity, but handle anyway)
        session_id = os.environ.get('HCOM_SESSION_ID')
        if session_id:
            initialize_instance_in_position_file(sender_name, session_id)
            instance_data = load_instance_position(sender_name)
        if not instance_data:
            print(format_error(f"Instance {sender_name} not found"), file=sys.stderr)
            return 1

    # Guard: If in subagent context, subagent MUST provide --_hcom_sender
    if not subagent_id and in_subagent_context(instance_data):
        from ..core.db import get_db
        conn = get_db()
        subagent_ids = [row['name'] for row in
                       conn.execute("SELECT name FROM instances WHERE parent_name = ?", (sender_name,)).fetchall()]

        suggestion = f"Use: hcom send 'message' --_hcom_sender {{alias}}"
        if subagent_ids:
            suggestion += f". Valid aliases: {', '.join(subagent_ids)}"

        print(format_error("Task tool subagent must provide sender identity", suggestion), file=sys.stderr)
        return 1

    # Check enabled state
    if not instance_data.get('enabled', False):
        previously_enabled = instance_data.get('previously_enabled', False)
        if previously_enabled:
            print(format_error("HCOM stopped. Cannot send messages."), file=sys.stderr)
        else:
            print(format_error("HCOM not started for this instance. To send a message first run: 'hcom start' then use hcom send"), file=sys.stderr)
        return 1

    # Set status to active for subagents
    if subagent_id:
        set_status(subagent_id, 'active', 'send')

    # Send message
    if not send_message(sender_name, message):
        print(format_error("Failed to send message"), file=sys.stderr)
        return 1

    # Get recipient feedback
    recipient_feedback = get_recipient_feedback(message, sender_name)

    # Show unread messages, grouped by subagent vs main
    from ..core.db import get_db
    conn = get_db()
    messages, _ = get_unread_messages(sender_name, update_position=True)
    if messages:
        subagent_names = {row['name'] for row in
                        conn.execute("SELECT name FROM instances WHERE parent_name = ?", (sender_name,)).fetchall()}

        # Separate subagent messages from main messages
        subagent_msgs = []
        main_msgs = []
        for msg in messages:
            sender = msg['from']
            if sender in subagent_names:
                subagent_msgs.append(msg)
            else:
                main_msgs.append(msg)

        output_parts = [recipient_feedback]
        max_msgs = MAX_MESSAGES_PER_DELIVERY

        if main_msgs:
            formatted = format_hook_messages(main_msgs[:max_msgs], sender_name)
            output_parts.append(f"\n{formatted}")

        if subagent_msgs:
            formatted = format_hook_messages(subagent_msgs[:max_msgs], sender_name)
            output_parts.append(f"\n[Subagent messages]\n{formatted}")

        print("".join(output_parts))
    else:
        print(recipient_feedback)

    return 0


def cmd_done(argv: list[str]) -> int:
    """Signal subagent completion: hcom done [--_hcom_sender ID]
    Control command used by subagents to signal they've finished work
    and are ready to receive messages.
    """
    subagent_id = None
    if '--_hcom_sender' in argv:
        idx = argv.index('--_hcom_sender')
        if idx + 1 < len(argv):
            subagent_id = argv[idx + 1]

    if not subagent_id:
        print(format_error("hcom done requires --_hcom_sender flag. Run: 'hcom done --_hcom_sender {alias}'"), file=sys.stderr)
        return 1

    instance_data = load_instance_position(subagent_id)
    if not instance_data:
        print(format_error(f"'{subagent_id}' not found"), file=sys.stderr)
        return 1

    if not instance_data.get('enabled', False):
        print(format_error(f"HCOM not started for '{subagent_id}'"), file=sys.stderr)
        return 1

    # PostToolUse will handle the actual polling loop
    print(f"{subagent_id}: waiting for messages...")
    return 0
