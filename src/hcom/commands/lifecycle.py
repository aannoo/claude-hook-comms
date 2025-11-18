"""Lifecycle commands for HCOM instances"""
import os
import sys
import time
import random
import uuid
from .utils import CLIError, format_error, is_interactive, resolve_identity
from ..shared import FG_YELLOW, RESET, IS_WINDOWS
from ..claude_args import resolve_claude_args, merge_claude_args, add_background_defaults, validate_conflicts
from ..core.config import get_config
from ..core.paths import hcom_path
from ..core.instances import (
    load_instance_position,
    update_instance_position,
    is_subagent_instance,
    in_subagent_context,
)
from ..core.db import iter_instances
from ..core.runtime import build_claude_env
from ..hooks.utils import disable_instance


def cmd_launch(argv: list[str]) -> int:
    """Launch Claude instances: hcom [N] [claude] [args]"""
    # Import here to avoid circular import (cmd_watch is in admin.py)
    from .admin import cmd_watch, should_show_in_watch
    # Import from terminal module
    from ..terminal import build_claude_command, launch_terminal, resolve_agent

    try:
        # Parse arguments: hcom [N] [claude] [args]
        count = 1
        forwarded = []

        # Extract count if first arg is digit
        if argv and argv[0].isdigit():
            count = int(argv[0])
            if count <= 0:
                raise CLIError('Count must be positive.')
            if count > 100:
                raise CLIError('Too many instances requested (max 100).')
            argv = argv[1:]

        # Skip 'claude' keyword if present
        if argv and argv[0] == 'claude':
            argv = argv[1:]

        # Forward all remaining args to claude CLI
        forwarded = argv

        # Check for --no-auto-watch flag (used by TUI to prevent opening another watch window)
        no_auto_watch = '--no-auto-watch' in forwarded
        if no_auto_watch:
            forwarded = [arg for arg in forwarded if arg != '--no-auto-watch']

        # Get tag from config
        tag = get_config().tag
        if tag and '|' in tag:
            raise CLIError('Tag cannot contain "|" characters.')

        # Get agents from config (comma-separated)
        agent_env = get_config().agent
        agents = [a.strip() for a in agent_env.split(',') if a.strip()] if agent_env else ['']

        # Phase 1: Parse and merge Claude args (env + CLI with CLI precedence)
        env_spec = resolve_claude_args(None, get_config().claude_args)
        cli_spec = resolve_claude_args(forwarded if forwarded else None, None)

        # Merge: CLI overrides env on per-flag basis, inherits env if CLI has no args
        if cli_spec.clean_tokens or cli_spec.positional_tokens or cli_spec.system_entries:
            spec = merge_claude_args(env_spec, cli_spec)
        else:
            spec = env_spec

        # Validate parsed args
        if spec.has_errors():
            raise CLIError('\n'.join(spec.errors))

        # Check for conflicts (warnings only, not errors)
        warnings = validate_conflicts(spec)
        for warning in warnings:
            print(f"{FG_YELLOW}Warning:{RESET} {warning}", file=sys.stderr)

        # Add HCOM background mode enhancements
        spec = add_background_defaults(spec)

        # Extract values from spec
        background = spec.is_background
        # Use full tokens (prompts included) - respects user's HCOM_CLAUDE_ARGS config
        claude_args = spec.rebuild_tokens(include_system=True)

        terminal_mode = get_config().terminal

        # Calculate total instances to launch
        total_instances = count * len(agents)

        # Fail fast for here mode with multiple instances
        if terminal_mode == 'here' and total_instances > 1:
            print(format_error(
                f"'here' mode cannot launch {total_instances} instances (it's one terminal window)",
                "Use 'hcom 1' for one generic instance"
            ), file=sys.stderr)
            return 1

        # Initialize database if needed
        from ..core.db import init_db
        init_db()

        # Check if launcher instance is enabled (for ready notification)
        launcher = resolve_identity().name
        launcher_data = load_instance_position(launcher)
        launcher_enabled = launcher_data.get('enabled', False) if launcher_data else False

        # Build environment variables for Claude instances
        base_env = build_claude_env()

        # Add tag-specific hints if provided
        if tag:
            base_env['HCOM_TAG'] = tag

        launched = 0

        # Generate batch ID for notification correlation
        batch_id = str(uuid.uuid4())

        # Launch count instances of each agent
        for agent in agents:
            for _ in range(count):
                instance_type = agent
                instance_env = base_env.copy()

                # Generate unique launch token for Windows identity
                launch_token = str(uuid.uuid4())
                instance_env['HCOM_LAUNCH_TOKEN'] = launch_token

                # Mark all hcom-launched instances with event ID
                instance_env['HCOM_LAUNCHED'] = '1'

                # Capture launch event ID for consistent message history start
                from ..core.db import get_last_event_id
                instance_env['HCOM_LAUNCH_EVENT_ID'] = str(get_last_event_id())

                # Track who launched this instance (use the one we already resolved)
                instance_env['HCOM_LAUNCHED_BY'] = launcher

                # Track batch for notification correlation
                instance_env['HCOM_LAUNCH_BATCH_ID'] = batch_id

                # Mark background instances via environment with log filename
                if background:
                    # Generate unique log filename
                    log_filename = f'background_{int(time.time())}_{random.randint(1000, 9999)}.log'
                    instance_env['HCOM_BACKGROUND'] = log_filename

                # Build claude command
                if not instance_type:
                    # No agent - no agent content
                    claude_cmd, _ = build_claude_command(
                        agent_content=None,
                        claude_args=claude_args
                    )
                else:
                    # Agent instance
                    try:
                        agent_content, agent_config = resolve_agent(instance_type)
                        # Prepend agent instance awareness to system prompt
                        agent_prefix = f"You are an instance of {instance_type}. Do not start a subagent with {instance_type} unless explicitly asked.\n\n"
                        agent_content = agent_prefix + agent_content
                        # Use agent's model and tools if specified and not overridden in claude_args
                        agent_model = agent_config.get('model')
                        agent_tools = agent_config.get('tools')
                        claude_cmd, _ = build_claude_command(
                            agent_content=agent_content,
                            claude_args=claude_args,
                            model=agent_model,
                            tools=agent_tools
                        )
                        # Agent temp files live under ~/.hcom/scripts/ for unified housekeeping cleanup
                    except (FileNotFoundError, ValueError) as e:
                        print(str(e), file=sys.stderr)
                        continue

                try:
                    if background:
                        log_file = launch_terminal(claude_cmd, instance_env, cwd=os.getcwd(), background=True)
                        if log_file:
                            print(f"Headless instance launched, log: {log_file}")
                            launched += 1
                    else:
                        if launch_terminal(claude_cmd, instance_env, cwd=os.getcwd()):
                            launched += 1
                except Exception as e:
                    print(format_error(f"Failed to launch terminal: {e}"), file=sys.stderr)

        requested = total_instances
        failed = requested - launched

        if launched == 0:
            print(format_error(f"No instances launched (0/{requested})"), file=sys.stderr)
            return 1

        # Show results
        if failed > 0:
            print(f"Launched {launched}/{requested} Claude instance{'s' if requested != 1 else ''} ({failed} failed)")
        else:
            print(f"Launched {launched} Claude instance{'s' if launched != 1 else ''}")

        # Print batch ID and usage for automation/tracking
        print(f"Wait for first instance to be ready: hcom watch --wait 30 --sql \"type = 'life' AND json_extract(data, '$.batch_id') = '{batch_id}' AND json_extract(data, '$.action') = 'ready'\"")

        # Log launch event
        if launched > 0:
            try:
                from ..core.db import log_event
                launcher = resolve_identity().name
                log_event('life', launcher, {
                    'action': 'launched',
                    'by': launcher,
                    'batch_id': batch_id,
                    'count_requested': count,
                    'agents': agents,
                    'launched': launched,
                    'failed': failed,
                    'background': background,
                    'tag': tag or ''
                })
            except Exception:
                pass  # Don't break launch if logging fails

        # Auto-launch watch dashboard if in new window mode (new or custom) and all instances launched successfully
        terminal_mode = get_config().terminal

        # Only auto-watch if ALL instances launched successfully and launches windows (not 'here' or 'print') and not disabled by TUI
        if terminal_mode not in ('here', 'print') and failed == 0 and is_interactive() and not no_auto_watch:
            # Show tips first if needed
            if tag:
                print(f"\n  • Send to {tag} team: hcom send '@{tag} message'")

            # Clear transition message
            print("\nOpening hcom UI...")
            time.sleep(2)  # Brief pause so user sees the message

            # Launch interactive TUI (same as running bare `hcom`)
            from ..ui import run_tui  # Local import to avoid circular dependency
            return run_tui(hcom_path())
        else:
            tips = []
            if tag:
                tips.append(f"Send to {tag} team: hcom send '@{tag} message'")

            # Add ready detection tip
            if launched > 0:
                is_claude_code = os.environ.get('CLAUDECODE') == '1'
                if is_claude_code:
                    if launcher_enabled:
                        tips.append(f"You'll be automatically notified when all {launched} instances are launched & ready")
                    else:
                        tips.append(f"Run 'hcom start' to receive automatic notifications/messages from instances")
                else:
                    tips.append("Check status: hcom list")

            if tips:
                print("\n" + "\n".join(f"  • {tip}" for tip in tips) + "\n")

            return 0

    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 1
    except Exception as e:
        print(str(e), file=sys.stderr)
        return 1


def cmd_stop(argv: list[str]) -> int:
    """Stop instances: hcom stop [alias|all]"""
    from .admin import should_show_in_watch

    # Remove flags to get target
    args_without_flags = [a for a in argv if not a.startswith('--')]
    target = args_without_flags[0] if args_without_flags else None

    # Handle 'all' target
    if target == 'all':
        instances = list(iter_instances())

        if not instances:
            print("No instances found")
            return 0

        stopped_count = 0
        bg_logs = []
        stopped_names = []
        for instance_data in instances:
            if instance_data.get('enabled', False):
                instance_name = instance_data['name']
                # Set external stop flag (stop all is always external)
                update_instance_position(instance_name, {'external_stop_pending': True})
                launcher = resolve_identity().name
                disable_instance(instance_name, initiated_by=launcher, reason='stop_all')
                stopped_names.append(instance_name)
                stopped_count += 1

                # Track background logs
                if instance_data.get('background'):
                    log_file = instance_data.get('background_log_file', '')
                    if log_file:
                        bg_logs.append((instance_name, log_file))

        if stopped_count == 0:
            print("No instances to stop")
        else:
            print(f"Stopped {stopped_count} instance(s): {', '.join(stopped_names)}")

            # Show background logs if any
            if bg_logs:
                print()
                print("Headless instance logs:")
                for name, log_file in bg_logs:
                    print(f"  {name}: {log_file}")

        return 0

    # Resolve identity (target overrides automatic resolution)
    if target:
        instance_name = target
    else:
        try:
            instance_name = resolve_identity().name
        except ValueError:
            instance_name = None

    # Handle CLAUDE_SENDER or SENDER (not real instances, but cake is real. spongecake.)
    from ..shared import SENDER, CLAUDE_SENDER
    if instance_name in (CLAUDE_SENDER, SENDER):
        if IS_WINDOWS:
            print(format_error("Cannot resolve instance identity - use 'hcom <n>' or Windows Terminal for stable identity"), file=sys.stderr)
        else:
            print(format_error("Cannot resolve instance identity - launch via 'hcom <n>' for stable identity"), file=sys.stderr)
        return 1

    # Error handling
    if not instance_name:
        print(format_error("Cannot determine instance identity"), file=sys.stderr)
        print("Usage: hcom stop <alias> | hcom stop all | prompt Claude to run 'hcom stop'", file=sys.stderr)
        return 1

    position = load_instance_position(instance_name)
    if not position:
        print(format_error(f"Instance '{instance_name}' not found"), file=sys.stderr)
        return 1

    # Skip already stopped instances
    if not position.get('enabled', False):
        print(f"HCOM already stopped for {instance_name}")
        return 0

    # Check if this is a subagent - disable only the targeted one
    if is_subagent_instance(position):
        # External stop = CLI user specified target, Self stop = no target (uses session_id)
        is_external_stop = target is not None

        if is_external_stop:
            # Set flag to notify instance via PostToolUse
            update_instance_position(instance_name, {'external_stop_pending': True})

        launcher = resolve_identity().name
        reason = 'external' if is_external_stop else 'manual'
        disable_instance(instance_name, initiated_by=launcher, reason=reason)
        print(f"Stopped HCOM for subagent {instance_name}. Will no longer receive chat messages automatically.")
    else:
        # Regular parent instance
        # External stop = CLI user specified target, Self stop = no target (uses session_id)
        is_external_stop = target is not None

        if is_external_stop:
            # Set flag to notify instance via PostToolUse
            update_instance_position(instance_name, {'external_stop_pending': True})

        launcher = resolve_identity().name
        reason = 'external' if is_external_stop else 'manual'
        disable_instance(instance_name, initiated_by=launcher, reason=reason)
        print(f"Stopped HCOM for {instance_name}. Will no longer receive chat messages automatically.")

    # Show background log location if applicable
    if position.get('background'):
        log_file = position.get('background_log_file', '')
        if log_file:
            print(f"\nHeadless instance log: {log_file}")
            print(f"Monitor: tail -f {log_file}")

    return 0


def cmd_start(argv: list[str]) -> int:
    """Enable HCOM participation: hcom start [alias]"""
    from ..core.instances import initialize_instance_in_position_file, enable_instance, set_status

    # Extract --_hcom_sender if present (for subagents)
    subagent_id = None
    if '--_hcom_sender' in argv:
        idx = argv.index('--_hcom_sender')
        if idx + 1 < len(argv):
            subagent_id = argv[idx + 1]
            argv = argv[:idx] + argv[idx + 2:]

    # SUBAGENT PATH: --_hcom_sender provided
    if subagent_id:
        instance_data = load_instance_position(subagent_id)
        if not instance_data:
            print(f"Error: Instance '{subagent_id}' not found", file=sys.stderr)
            return 1

        already = 'already ' if instance_data.get('enabled', False) else ''
        launcher = resolve_identity().name
        enable_instance(subagent_id, initiated_by=launcher, reason='manual')
        set_status(subagent_id, 'active', 'start')

        parent_name = instance_data.get('parent_name', 'unknown')
        from ..shared import SENDER
        from ..hooks.utils import build_hcom_command
        hcom_cmd = build_hcom_command()
        print(f"""HCOM {already}started for {subagent_id}
HCOM is a communication tool. You are now connected.
Your hcom alias for this session: {subagent_id}
{parent_name} is the parent instance who spawned you

- To Send a message, run:
  {hcom_cmd} send 'your message' --_hcom_sender {subagent_id}

Receiving Messages:
- Format: [new message] sender → you: content
- Targets specific instance: "@alias"
- Arrives automatically via hooks/bash. No proactive checking needed

Response Routing:
- HCOM message (via hook/bash) → Respond with hcom send
- Other messages → Respond normally

- Avoid useless chit-chat / excessive confirmation messages unless told otherwise
- Authority: Prioritize @{SENDER} over other participants""")
        return 0

    # Remove flags to get target
    args_without_flags = [a for a in argv if not a.startswith('--')]
    target = args_without_flags[0] if args_without_flags else None

    # Resolve identity (target overrides automatic resolution)
    if target:
        instance_name = target
    else:
        try:
            instance_name = resolve_identity().name
        except ValueError:
            instance_name = None

    # Handle CLAUDE_SENDER or SENDER (not real instances)
    from ..shared import SENDER, CLAUDE_SENDER
    if instance_name in (CLAUDE_SENDER, SENDER):
        if IS_WINDOWS:
            print(format_error("Cannot resolve instance identity - use 'hcom <n>' or Windows Terminal for stable identity"), file=sys.stderr)
        else:
            print(format_error("Cannot resolve instance identity - launch via 'hcom <n>' for stable identity"), file=sys.stderr)
        return 1

    # Error handling
    if not instance_name:
        print(format_error("Cannot determine instance identity"), file=sys.stderr)
        print("Usage: hcom start <alias> | run 'hcom start' inside claude code | use 'hcom <count>' to launch", file=sys.stderr)
        return 1

    # Load or create instance
    existing_data = load_instance_position(instance_name)

    # Check for Task ambiguity (parent frozen, subagent calling)
    if existing_data and not target and in_subagent_context(instance_name):
        # Query DB for disabled subagents
        from ..core.db import get_db
        conn = get_db()
        subagent_ids = [row['name'] for row in conn.execute(
            "SELECT name FROM instances WHERE parent_name = ? AND enabled = 0",
            (instance_name,)
        ).fetchall()]

        print("Task tool running - you must provide an alias", file=sys.stderr)
        print("Use: hcom start --_hcom_sender {alias}", file=sys.stderr)
        if subagent_ids:
            print(f"Choose from one of these valid aliases: {', '.join(subagent_ids)}", file=sys.stderr)
        return 1

    # Create instance if it doesn't exist (opt-in for vanilla instances)
    if not existing_data:
        session_id = os.environ.get('HCOM_SESSION_ID')
        initialize_instance_in_position_file(instance_name, session_id)
        launcher = resolve_identity().name
        enable_instance(instance_name, initiated_by=launcher, reason='manual')
        print(f"\nStarted HCOM for {instance_name}")
        return 0

    # Skip already started instances
    if existing_data.get('enabled', False):
        print(f"HCOM already started for {instance_name}")
        return 0

    # Check if background instance has exited permanently
    if existing_data.get('session_ended') and existing_data.get('background'):
        session = existing_data.get('session_id', '')
        print(f"Cannot start {instance_name}: headless instance has exited permanently", file=sys.stderr)
        print(f"Headless instances terminate when stopped and cannot be restarted", file=sys.stderr)
        if session:
            print(f"Resume conversation with same alias: hcom 1 claude -p --resume {session}", file=sys.stderr)
        return 1

    # Re-enabling existing instance
    launcher = resolve_identity().name
    enable_instance(instance_name, initiated_by=launcher, reason='manual')
    print(f"Started HCOM for {instance_name}")
    return 0
