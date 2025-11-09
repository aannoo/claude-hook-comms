"""Messaging commands for HCOM"""
import os
import sys
from .utils import format_error, validate_message
from ..shared import MENTION_PATTERN, SENDER, MAX_MESSAGES_PER_DELIVERY
from ..core.config import get_config
from ..core.instances import load_instance_position, load_all_positions, in_subagent_context, set_status
from ..core.messages import unescape_bash, send_message, get_unread_messages, format_hook_messages


def cmd_send(argv: list[str], force_cli: bool = False, quiet: bool = False) -> int:
    """Send message to hcom: hcom send "message" [--_hcom_session ID] [--_hcom_sender NAME]"""
    # Import identity helpers from core
    from ..core.instances import resolve_instance_name, initialize_instance_in_position_file

    # Parse message and session_id
    message = None
    session_id = None
    subagent_id = None
    custom_sender = None

    # Extract --_hcom_sender if present (for subagents)
    if '--_hcom_sender' in argv:
        idx = argv.index('--_hcom_sender')
        if idx + 1 < len(argv):
            subagent_id = argv[idx + 1]
            argv = argv[:idx] + argv[idx + 2:]  # Remove flag and value

    # Extract --from if present (for custom external sender)
    if '--from' in argv:
        idx = argv.index('--from')
        if idx + 1 < len(argv):
            custom_sender = argv[idx + 1]
            # Validate: no pipes, max 50 chars, alphanumeric + hyphen/underscore
            if '|' in custom_sender:
                print(format_error("Sender name cannot contain '|'"), file=sys.stderr)
                return 1
            if len(custom_sender) > 50:
                print(format_error("Sender name too long (max 50 chars)"), file=sys.stderr)
                return 1
            if not custom_sender or not all(c.isalnum() or c in '-_' for c in custom_sender):
                print(format_error("Sender name must be alphanumeric with hyphens/underscores"), file=sys.stderr)
                return 1
            argv = argv[:idx] + argv[idx + 2:]  # Remove flag and value
        else:
            print(format_error("--from requires a sender name"), file=sys.stderr)
            return 1

    # Read session ID from env var (set by SessionStart hook)
    session_id = os.environ.get('HCOM_SESSION_ID')

    # First non-flag argument is the message
    if argv:
        message = unescape_bash(argv[0])

    # Check message is provided
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
            sender_name = SENDER
            all_names = all_instances + [sender_name]
            unmatched = [m for m in mentions
                        if not any(name.lower().startswith(m.lower()) for name in all_names)]
            if unmatched:
                print(f"Note: @{', @'.join(unmatched)} don't match any instances - broadcasting to all", file=sys.stderr)
        except Exception:
            pass  # Don't fail on warning

    # Determine sender from injected flags or CLI
    if session_id and not force_cli:
        # Instance context - use sender override if provided (subagent), otherwise resolve from session_id
        if subagent_id: #subagent id is same as subagent name
            sender_name = subagent_id
            instance_data = load_instance_position(sender_name)
            if not instance_data:
                print(format_error(f"Subagent instance file missing for {subagent_id}"), file=sys.stderr)
                return 1
        else:
            # Normal instance - resolve name from session_id
            try:
                sender_name, instance_data = resolve_instance_name(session_id, get_config().tag)
            except (ValueError, Exception) as e:
                print(format_error(f"Invalid session_id: {e}"), file=sys.stderr)
                return 1

            # Initialize instance if doesn't exist (first use)
            if not instance_data:
                initialize_instance_in_position_file(sender_name, session_id)
                instance_data = load_instance_position(sender_name)

            # Guard: If in subagent context, subagent MUST provide --_hcom_sender
            if in_subagent_context(instance_data):
                # Get only enabled subagents (active, can send messages)
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
                # Was enabled, now disabled - don't suggest re-enabling
                print(format_error("HCOM stopped. Cannot send messages."), file=sys.stderr)
            else:
                # Never enabled - helpful message
                print(format_error("HCOM not started for this instance. To send a message first run: 'hcom start' then use hcom send"), file=sys.stderr)
            return 1

        # Set status to active for subagents (identity confirmed, enabled verified)
        if subagent_id:
            set_status(subagent_id, 'active', 'send')

        # Send message
        if not send_message(sender_name, message):
            print(format_error("Failed to send message"), file=sys.stderr)
            return 1

        # Show unread messages, grouped by subagent vs main
        messages, _ = get_unread_messages(sender_name, update_position=True)
        if messages:
            # Get list of subagent names for this parent
            from ..core.db import get_db
            conn = get_db()
            subagent_names = {row['name'] for row in
                            conn.execute("SELECT name FROM instances WHERE parent_name = ?", (sender_name,)).fetchall()}

            # Separate subagent messages from main messages
            subagent_msgs = []
            main_msgs = []
            for msg in messages:
                sender = msg['from']
                # Check if sender is a subagent of this instance
                if sender in subagent_names:
                    subagent_msgs.append(msg)
                else:
                    main_msgs.append(msg)

            output_parts = ["Message sent"]
            max_msgs = MAX_MESSAGES_PER_DELIVERY

            if main_msgs:
                formatted = format_hook_messages(main_msgs[:max_msgs], sender_name)
                output_parts.append(f"\n{formatted}")

            if subagent_msgs:
                formatted = format_hook_messages(subagent_msgs[:max_msgs], sender_name)
                output_parts.append(f"\n[Subagent messages]\n{formatted}")

            print("".join(output_parts), file=sys.stderr)
        else:
            print("Message sent", file=sys.stderr)

        return 0
    else:
        # CLI context - no session_id or force_cli=True

        # Use custom sender if provided via --from, otherwise default
        sender_name = custom_sender if custom_sender else SENDER

        # Warn if inside Claude Code but no session_id (hooks not working(?))
        if os.environ.get('CLAUDECODE') == '1' and not session_id and not force_cli:
            if subagent_id:
                # Subagent command not auto-approved
                print(format_error(
                    "Cannot determine alias - hcom command not auto-approved",
                    "Run hcom commands directly with correct syntax: 'hcom send 'message' --_hcom_sender {alias}'"
                ), file=sys.stderr)
                return 1
            else:
                print(f"⚠️  Cannot determine alias - message sent as '{sender_name}'", file=sys.stderr)

        if not send_message(sender_name, message):
            print(format_error("Failed to send message"), file=sys.stderr)
            return 1

        if not quiet:
            print(f"✓ Sent from {sender_name}", file=sys.stderr)

        return 0


def send_cli(message: str, quiet: bool = False) -> int:
    """Force CLI sender (skip outbox, use config sender name)"""
    return cmd_send([message], force_cli=True, quiet=quiet)


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
