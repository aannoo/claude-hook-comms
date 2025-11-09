"""Lifecycle commands for HCOM instances"""
import os
import sys
import time
import random
from .utils import CLIError, format_error, is_interactive
from ..shared import FG_YELLOW, RESET
from ..claude_args import resolve_claude_args, merge_claude_args, add_background_defaults, validate_conflicts
from ..core.config import get_config
from ..core.paths import hcom_path
from ..core.instances import (
    load_instance_position,
    update_instance_position,
    load_all_positions,
    is_subagent_instance,
    in_subagent_context,
)
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

        # Build environment variables for Claude instances
        base_env = build_claude_env()

        # Add tag-specific hints if provided
        if tag:
            base_env['HCOM_TAG'] = tag

        launched = 0

        # Launch count instances of each agent
        for agent in agents:
            for _ in range(count):
                instance_type = agent
                instance_env = base_env.copy()

                # Mark all hcom-launched instances with timestamp and event ID
                instance_env['HCOM_LAUNCHED'] = '1'
                instance_env['HCOM_LAUNCH_TIME'] = str(time.time())

                # Capture launch event ID for consistent message history start
                from ..core.db import get_last_event_id
                instance_env['HCOM_LAUNCH_EVENT_ID'] = str(get_last_event_id())

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
                        # Mark this as a subagent instance for SessionStart hook
                        instance_env['HCOM_SUBAGENT_TYPE'] = instance_type
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
            tips = [
                "Run 'hcom' to view/send in conversation dashboard",
            ]
            if tag:
                tips.append(f"Send to {tag} team: hcom send '@{tag} message'")

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
    # Import identity helpers from core
    from ..core.instances import resolve_instance_name
    from .admin import should_show_in_watch

    # Parse arguments
    target = None

    # Read session ID from env var (set by SessionStart hook)
    session_id = os.environ.get('HCOM_SESSION_ID')

    # Remove flags to get target
    args_without_flags = [a for a in argv if not a.startswith('--')]
    if args_without_flags:
        target = args_without_flags[0]

    # Handle 'all' target
    if target == 'all':
        positions = load_all_positions()

        if not positions:
            print("No instances found")
            return 0

        stopped_count = 0
        bg_logs = []
        stopped_names = []
        for instance_name, instance_data in positions.items():
            if instance_data.get('enabled', False):
                # Set external stop flag (stop all is always external)
                update_instance_position(instance_name, {'external_stop_pending': True})
                disable_instance(instance_name)
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

    # Stop specific instance or self
    # Get instance name from injected session or target
    if session_id and not target:
        instance_name, _ = resolve_instance_name(session_id, get_config().tag)
    else:
        instance_name = target

    position = load_instance_position(instance_name) if instance_name else None

    if not instance_name:
        if os.environ.get('CLAUDECODE') == '1':
            print("Error: Cannot determine instance", file=sys.stderr)
            print("Usage: Prompt Claude to run 'hcom stop' (or directly use: hcom stop <alias> or hcom stop all)", file=sys.stderr)
        else:
            print("Error: Alias required", file=sys.stderr)
            print("Usage: hcom stop <alias>", file=sys.stderr)
            print("   Or: hcom stop all", file=sys.stderr)
            print("   Or: prompt claude to run 'hcom stop' on itself", file=sys.stderr)
            positions = load_all_positions()
            visible = [alias for alias, data in positions.items() if should_show_in_watch(data)]
            if visible:
                print(f"Active aliases: {', '.join(sorted(visible))}", file=sys.stderr)
        return 1

    if not position:
        print(format_error(f"Instance '{instance_name}' not found"), file=sys.stderr)
        return 1

    # Skip already stopped instances
    if not position.get('enabled', False):
        print(f"HCOM already stopped for {instance_name}")
        return 0

    # Check if this is a subagent - disable all siblings
    if is_subagent_instance(position):
        parent_session_id = position.get('parent_session_id')
        positions = load_all_positions()
        disabled_count = 0
        disabled_names = []

        for name, data in positions.items():
            if data.get('parent_session_id') == parent_session_id and data.get('enabled', False):
                update_instance_position(name, {'external_stop_pending': True})
                disable_instance(name)
                disabled_count += 1
                disabled_names.append(name)

        if disabled_count > 0:
            print(f"Stopped {disabled_count} subagent(s): {', '.join(disabled_names)}")
            print("Note: All subagents of the same parent must be disabled together.")
        else:
            print(f"No enabled subagents found for {instance_name}")
    else:
        # Regular parent instance
        # External stop = CLI user specified target, Self stop = no target (uses session_id)
        is_external_stop = target is not None

        if is_external_stop:
            # Set flag to notify instance via PostToolUse
            update_instance_position(instance_name, {'external_stop_pending': True})

        disable_instance(instance_name)
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
    # Import identity helpers from core
    from ..core.instances import resolve_instance_name, initialize_instance_in_position_file, enable_instance, set_status

    # Parse arguments
    target = None

    # Read session ID from env var (set by SessionStart hook)
    session_id = os.environ.get('HCOM_SESSION_ID')

    # Extract --_hcom_sender if present (for subagents)
    sender_override = None
    if '--_hcom_sender' in argv:
        idx = argv.index('--_hcom_sender')
        if idx + 1 < len(argv):
            sender_override = argv[idx + 1]
            argv = argv[:idx] + argv[idx + 2:]

    # Remove flags to get target
    args_without_flags = [a for a in argv if not a.startswith('--')]
    if args_without_flags:
        target = args_without_flags[0]

    # SUBAGENT PATH: --_hcom_sender provided
    if sender_override:
        instance_data = load_instance_position(sender_override)
        if not instance_data or instance_data.get('status') == 'exited':
            print(f"Error: Instance '{sender_override}' not found or has exited", file=sys.stderr)
            return 1

        already = 'already ' if instance_data.get('enabled', False) else ''
        enable_instance(sender_override)
        set_status(sender_override, 'active', 'start')
        print(f"HCOM {already} started for {sender_override}")
        print(f"Send: hcom send 'message' --_hcom_sender {sender_override}")
        print(f"When finished working always run: hcom done --_hcom_sender {sender_override}")
        return 0

    # Get instance name from injected session or target
    if session_id and not target:
        instance_name, existing_data = resolve_instance_name(session_id, get_config().tag)

        # Check for Task ambiguity (parent frozen, subagent calling)
        if existing_data and in_subagent_context(existing_data):
            # Get list of subagents from THIS Task execution that are disabled
            active_list = existing_data.get('current_subagents', [])
            positions = load_all_positions()
            subagent_ids = [
                sid for sid in active_list
                if sid in positions and not positions[sid].get('enabled', False)
            ]

            print("Task tool running - you must provide an alias", file=sys.stderr)
            print("Use: hcom start --_hcom_sender {alias}", file=sys.stderr)
            if subagent_ids:
                print(f"Choose from one of these valid aliases: {', '.join(subagent_ids)}", file=sys.stderr)
            return 1

        # Create instance if it doesn't exist (opt-in for vanilla instances)
        if not existing_data:
            initialize_instance_in_position_file(instance_name, session_id)
            # Enable instance (clears all stop flags)
            enable_instance(instance_name)
            print(f"\nStarted HCOM for {instance_name}")
        else:
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
            enable_instance(instance_name)
            print(f"Started HCOM for {instance_name}")

        return 0

    # CLI path: start specific instance
    positions = load_all_positions()

    # Handle missing target from external CLI
    if not target:
        if os.environ.get('CLAUDECODE') == '1':
            print("Error: Cannot determine instance", file=sys.stderr)
            print("Usage: Prompt Claude to run 'hcom start' (or: hcom start <alias>)", file=sys.stderr)
        else:
            print("Error: Alias required", file=sys.stderr)
            print("Usage: hcom start <alias> (or: prompt claude to run 'hcom start')", file=sys.stderr)
            print("To launch new instances: hcom <count>", file=sys.stderr)
        return 1

    # Start specific instance
    instance_name = target
    position = positions.get(instance_name)

    if not position:
        print(f"Instance not found: {instance_name}")
        return 1

    # Skip already started instances
    if position.get('enabled', False):
        print(f"HCOM already started for {instance_name}")
        return 0

    # Check if background instance has exited permanently
    if position.get('session_ended') and position.get('background'):
        session = position.get('session_id', '')
        print(f"Cannot start {instance_name}: headless instance has exited permanently")
        print(f"Headless instances terminate when stopped and cannot be restarted")
        if session:
            print(f"Resume conversation with same alias: hcom 1 claude -p --resume {session}")
        return 1

    # Enable instance (clears all stop flags)
    enable_instance(instance_name)

    print(f"Started HCOM for {instance_name}")
    return 0
