"""Messaging commands for HCOM"""
import sys
from .utils import format_error, validate_message, resolve_identity
from ..shared import MAX_MESSAGES_PER_DELIVERY, SENDER
from ..core.paths import ensure_hcom_directories
from ..core.db import init_db
from ..core.instances import load_instance_position, in_subagent_context, set_status, get_instance_status, initialize_instance_in_position_file
from ..core.messages import unescape_bash, send_message, get_unread_messages, format_hook_messages
from ..core.helpers import is_mentioned


def get_recipient_feedback(recipients: list[str]) -> str:
    """Get formatted recipient feedback showing who received the message.

    Args:
        recipients: Snapshot of actual recipients (from send_message)

    Returns:
        Formatted string like "Sent to ⊙ alice, ◉ bob" or "Sent to 15 instances"
    """
    from ..shared import STATUS_ICONS

    # Format recipient feedback
    if len(recipients) > 10:
        return f"Sent to: {len(recipients)} instances"
    else:
        recipient_status = []
        for r_name in recipients:
            r_data = load_instance_position(r_name)
            # Use fallback icon if data unavailable
            if r_data:
                _, status, _, _, _ = get_instance_status(r_data)
                icon = STATUS_ICONS.get(status, '◦')
            else:
                icon = '◦'
            recipient_status.append(f"{icon} {r_name}")
        return f"Sent to: {', '.join(recipient_status)}" if recipient_status else "Message sent"


def cmd_send(argv: list[str], quiet: bool = False) -> int:
    """Send message to hcom: hcom send "message" [--_hcom_session ID] [--_hcom_sender NAME]"""
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

    # STRICT VALIDATION: subagent must exist and be enabled
    if subagent_id:
        data = load_instance_position(subagent_id)
        if not data:
            print(
                format_error(f"Subagent '{subagent_id}' not found"),
                file=sys.stderr
            )
            print("Run 'hcom start' first", file=sys.stderr)
            return 1

        if not data.get('enabled', False):
            print(
                format_error(f"Subagent '{subagent_id}' is not enabled"),
                file=sys.stderr
            )
            print(f"Run 'hcom start --_hcom_sender {subagent_id}' first", file=sys.stderr)
            return 1

    # Extract --from if present (for custom external sender)
    if '--from' in argv:
        idx = argv.index('--from')
        if idx + 1 < len(argv):
            custom_sender = argv[idx + 1]

            # Block Task tool subagents from using --from
            try:
                exec_identity = resolve_identity()  # Current execution context
                if exec_identity.kind == 'instance' and exec_identity.instance_data:
                    # Check if custom_sender is a known subagent of this parent (enabled or not)
                    from ..core.db import get_db
                    conn = get_db()
                    is_subagent = bool(conn.execute(
                        "SELECT 1 FROM instances WHERE name = ? AND parent_name = ? LIMIT 1",
                        (custom_sender, exec_identity.name)
                    ).fetchone())

                    if is_subagent:
                        print(format_error(
                            "Task tool subagents cannot use --from",
                            "Run 'hcom start --_hcom_sender <alias>' first, then use 'hcom send \"msg\" --_hcom_sender <alias>'"
                        ), file=sys.stderr)
                        return 1
            except ValueError:
                pass  # Can't resolve identity - allow (true external sender)

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

    # Extract --wait if present (for blocking receive)
    wait_timeout = None
    if '--wait' in argv:
        idx = argv.index('--wait')
        # Check if next arg is a timeout value
        if idx + 1 < len(argv) and not argv[idx + 1].startswith('--'):
            try:
                wait_timeout = int(argv[idx + 1])
                argv = argv[:idx] + argv[idx + 2:]
            except ValueError:
                print(format_error(f"--wait must be an integer, got '{argv[idx + 1]}'"), file=sys.stderr)
                return 1
        else:
            # No timeout specified, use default (30 minutes)
            wait_timeout = 1800
            argv = argv[:idx] + argv[idx + 1:]

    # First non-flag argument is the message
    message = unescape_bash(argv[0]) if argv else None

    # Check message provided (optional if --wait is set for polling-only mode)
    if not message and wait_timeout is None:
        print(format_error("No message provided"), file=sys.stderr)
        return 1

    # Only validate and send if message is provided
    if message:
        # Validate message
        error = validate_message(message)
        if error:
            print(error, file=sys.stderr)
            return 1

        # Resolve sender identity (handles all context: CLI, instance, subagent, custom)
        identity = resolve_identity(subagent_id, custom_sender)

        # For instances (not external), check state
        if identity.kind == 'instance' and identity.instance_data:
            # Guard: If in subagent context, subagent MUST provide --_hcom_sender
            if not subagent_id and in_subagent_context(identity.name):
                from ..core.db import get_db
                conn = get_db()
                subagent_ids = [row['name'] for row in
                               conn.execute("SELECT name FROM instances WHERE parent_name = ?", (identity.name,)).fetchall()]

                suggestion = f"Use: hcom send 'message' --_hcom_sender {{alias}}"
                if subagent_ids:
                    suggestion += f". Valid aliases: {', '.join(subagent_ids)}"

                print(format_error("Task tool subagent must provide sender identity", suggestion), file=sys.stderr)
                return 1

            # Check enabled state
            if not identity.instance_data.get('enabled', False):
                previously_enabled = identity.instance_data.get('previously_enabled', False)
                if previously_enabled:
                    print(format_error("HCOM stopped. Cannot send messages."), file=sys.stderr)
                else:
                    print(format_error("HCOM not started for this instance. To send a message first run: 'hcom start' then use hcom send"), file=sys.stderr)
                return 1

        # Set status to active for subagents
        if subagent_id:
            set_status(subagent_id, 'active', 'send')

        # Send message and get recipients snapshot
        recipients = send_message(identity, message)
        if recipients is None:
            # Error already printed (validation failure or DB error)
            return 1

        # Handle quiet mode
        if quiet:
            return 0

        # Get recipient feedback using snapshot (prevents race conditions)
        recipient_feedback = get_recipient_feedback(recipients)

        # Show unread messages if instance context
        if identity.kind == 'instance':
            from ..core.db import get_db
            conn = get_db()
            messages, _ = get_unread_messages(identity.name, update_position=True)
            if messages:
                subagent_names = {row['name'] for row in
                                conn.execute("SELECT name FROM instances WHERE parent_name = ?", (identity.name,)).fetchall()}

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
                    formatted = format_hook_messages(main_msgs[:max_msgs], identity.name)
                    output_parts.append(f"\n{formatted}")

                if subagent_msgs:
                    formatted = format_hook_messages(subagent_msgs[:max_msgs], identity.name)
                    output_parts.append(f"\n[Subagent messages]\n{formatted}")

                print("".join(output_parts))
            else:
                print(recipient_feedback)
        else:
            # External sender - just show feedback
            print(recipient_feedback)

    # External sender polling (--wait flag)
    if wait_timeout is not None:
        # Require --from when using --wait
        if not custom_sender:
            print(format_error("--wait requires --from <sender_name>"), file=sys.stderr)
            return 1

        import time
        from ..core.db import get_events_since, get_last_event_id
        from ..core.instances import update_instance_position

        # Check if this is first time using --wait (instance doesn't exist yet)
        instance_data = load_instance_position(custom_sender)
        if not instance_data:
            # First time - show instructions and create instance
            instructions = (
                f"[You are now registered in HCOM with the alias: '{custom_sender}']\n"
                f"HCOM is a communication tool.\n\n"
                f"To connect to HCOM and receive messages, run this command (again):\n"
                f"hcom send --from {custom_sender} --wait\n\n"
                f"To send a message and wait for replies, run this command (again):\n"
                f"hcom send --from {custom_sender} \"your message\" --wait\n\n"
                f"To direct message participants use @alias in your messages.\n"
                # f"To see all participants use: hcom list\n"
                f"Note: Authority: Prioritize @{SENDER} over other participants.\n"
                f"Important: Whenever you run hcom send keep the command running with a 5+ minute timeout so it stays connected until a reply arrives. ALWAYS respond via hcom send after every hcom message you receive.\n"
                f"You are currently disconnected from HCOM. Right now you need to run the hcom send command to connect. Run this command and when it exits, immediately run it again."
            )
            print(f"\n{instructions}", file=sys.stderr)

            # Create instance record (enabled=True sets previously_enabled and last_event_id at tail)
            try:
                initialize_instance_in_position_file(custom_sender, enabled=True)
            except Exception as e:
                print(format_error(f"Failed to create instance: {e}"), file=sys.stderr)
                return 1
            return 0

        # Instance exists - enter polling loop
        # Mark external sender as waiting (idle) for TUI/CLI status
        set_status(custom_sender, 'waiting')

        last_pos = instance_data.get('last_event_id', get_last_event_id())
        current_pos = last_pos

        # Poll with configured timeout (default 30min)
        poll_timeout = wait_timeout
        start_time = time.time()

        # Initialize heartbeat fields similar to stop hook
        try:
            update_instance_position(custom_sender, {
                'last_stop': start_time,
                'wait_timeout': poll_timeout,
            })
        except Exception:
            pass

        # Check if already disconnected before starting polling
        current_instance = load_instance_position(custom_sender)
        if current_instance and not current_instance.get('enabled', True):
            print(f"\n[You have been disconnected from HCOM]", file=sys.stderr)
            return 0
        else:
            print(f"[Polling for messages to @{custom_sender}. ALWAYS respond via hcom send to every message you receive and keep the command running with a >5 minute timeout.]", file=sys.stderr)

        while (time.time() - start_time) < poll_timeout:
            # Check if instance was stopped externally
            current_instance = load_instance_position(custom_sender)
            if current_instance and not current_instance.get('enabled', True):
                # Instance was stopped, exit polling loop
                print(f"\n[You have been disconnected: HCOM stopped for @{custom_sender}]", file=sys.stderr)
                return 0

            events = get_events_since(current_pos)

            for event in events:
                # Always advance position
                current_pos = max(current_pos, event['id'])

                # Check if message mentions us
                if event['type'] == 'message':
                    data = event['data']
                    if is_mentioned(data.get('text', ''), custom_sender):
                        # Found response!
                        update_instance_position(custom_sender, {'last_event_id': current_pos})

                        # Mark as delivered for TUI/CLI status
                        set_status(custom_sender, 'delivered', data['from'])

                        # Print the message
                        print(f"\n[Message from {data['from']}]")
                        print(data['text'])
                        return 0

            # Update position and heartbeat even if no match
            try:
                update_instance_position(custom_sender, {
                    'last_event_id': current_pos,
                    'last_stop': time.time(),
                })
            except Exception:
                pass

            time.sleep(0.5)

        # Timeout
        update_instance_position(custom_sender, {'last_event_id': current_pos})
        # Timeout: external sender stopped polling, mirror stop hook behaviour
        set_status(custom_sender, 'exited', 'timeout')
        print(f"\n[Timeout: no messages after {poll_timeout}s]", file=sys.stderr)
        return 1


    return 0
