"""Command utilities for HCOM"""
import sys
import re
from ..shared import __version__, MAX_MESSAGE_SIZE, SenderIdentity, SENDER, CLAUDE_SENDER


class CLIError(Exception):
    """Raised when arguments cannot be mapped to command semantics."""


def get_help_text() -> str:
    """Generate help text with current version"""
    return f"""hcom {__version__}
Hook-based communication bus for real-time messaging between Claude Code instances

Usage: hcom                           # TUI dashboard
       [ENV_VARS] hcom <COUNT> [claude <ARGS>...]
       hcom watch [--last N] [--wait SEC] [--sql EXPR]
       hcom list [--json] [-v|--verbose]
       hcom send "message"
       hcom stop [alias|all]
       hcom start [alias]
       hcom reset [logs|hooks|config]

Launch Examples:
  hcom 3             open 3 terminals with claude connected to hcom
  hcom 3 claude -p                                       + headless
  HCOM_TAG=api hcom 3 claude -p               + @-mention group tag
  claude 'run hcom start'        claude code with prompt also works

Commands:
  watch               Query recent events (JSON)
    --last N          Limit to last N events (default: 20)
    --wait [SEC]      Block until matching event (default: 60s timeout)
    --sql EXPR        SQL WHERE clause filter

  list                List current instances status
    -v, --verbose     Show detailed metadata
    --json            Emit JSON with detailed data

  send "msg"          Send message to all instances
  send "@alias msg"   Send to specific instance/group
    --from <name>     Custom external identity
    --wait            Block until reply with --from

  stop                Stop current instance (from inside Claude)
  stop <alias>        Stop specific instance
  stop all            Stop all instances

  start               Start current instance (from inside Claude)
  start <alias>       Start specific instance

  reset               Stop all + archive logs + remove hooks + clear config
  reset logs          Clear + archive conversation
  reset hooks         Safely remove hcom hooks from claude settings.json
  reset config        Clear + archive config.env

Environment Variables:
  HCOM_TAG=name               Group tag (creates name-* instances)
  HCOM_AGENT=type             Agent from .claude/agents/ (comma-separated for multiple)
  HCOM_TERMINAL=mode          Terminal: new|here|print|"custom {{script}}"
  HCOM_HINTS=text             Text appended to all messages received by instance
  HCOM_TIMEOUT=secs           Time until disconnected from hcom chat (default: 1800s / 30m)
  HCOM_SUBAGENT_TIMEOUT=secs  Subagent idle timeout (default: 30s)
  HCOM_CLAUDE_ARGS=args       Claude CLI defaults (e.g., '-p --model opus "hello!"')

  ANTHROPIC_MODEL=opus # Any env var passed through to Claude Code

  Persist Env Vars in `~/.hcom/config.env`
"""


def format_error(message: str, suggestion: str | None = None) -> str:
    """Format error message consistently"""
    base = f"Error: {message}"
    if suggestion:
        base += f". {suggestion}"
    return base


def is_interactive() -> bool:
    """Check if running in interactive mode"""
    return sys.stdin.isatty() and sys.stdout.isatty()


def resolve_identity(subagent_id: str | None = None, custom_from: str | None = None, system_sender: str | None = None) -> SenderIdentity:
    """Resolve identity in CLI/hook context.

    Args:
        subagent_id: Explicit subagent ID (from Task tool context)
        custom_from: Custom display name (--from flag)
        system_sender: System notification sender name (e.g., 'hcom-launcher')

    Returns:
        SenderIdentity with kind, name, and instance_data

    Identity kind:
        - 'external': Custom sender or CLI (--from or bigboss)
        - 'instance': Real instance (Claude Code with session)
        - 'system': System notifications (launcher, watchdog, etc)
    """
    import os
    from ..shared import MAPID
    from ..core.instances import load_instance_position, resolve_instance_name
    from ..core.config import get_config

    # System sender (internal notifications) - always system
    if system_sender:
        return SenderIdentity(kind='system', name=system_sender, instance_data=None)

    # Custom sender (--from) - always external
    if custom_from:
        return SenderIdentity(kind='external', name=custom_from, instance_data=None)

    # Subagent explicit (Task tool)
    if subagent_id:
        data = load_instance_position(subagent_id)
        if not data:
            # This shouldn't happen - cmd_send validates before calling
            raise ValueError(f"Subagent '{subagent_id}' position data missing")
        return SenderIdentity(kind='instance', name=subagent_id, instance_data=data)

    # CLI context (not in Claude Code)
    if os.environ.get('CLAUDECODE') != '1':
        return SenderIdentity(kind='external', name=SENDER, instance_data=None)

    # Inside Claude: try session_id (Unix only - CLAUDE_ENV_FILE doesn't work on Windows)
    session_id = os.environ.get('HCOM_SESSION_ID')
    if session_id:
        name, data = resolve_instance_name(session_id, get_config().tag)
        # Return instance identity (data may be None if not opted in yet)
        return SenderIdentity(kind='instance', name=name, instance_data=data)

    # Try MAPID (Windows fallback - terminal session ID like WT_SESSION)
    if MAPID:
        from ..core.db import get_instance_by_mapid, get_db
        from ..core.instances import resolve_instance_name

        # First try to find existing instance by MAPID
        data = get_instance_by_mapid(MAPID)
        if data:
            return SenderIdentity(kind='instance', name=data['name'], instance_data=data)

        # No instance for this MAPID - look up session_id from mapping
        # This handles Windows resume in different terminal (MAPID changes but session_id stays same)
        conn = get_db()
        row = conn.execute(
            "SELECT session_id FROM mapid_sessions WHERE mapid = ?",
            (MAPID,)
        ).fetchone()

        if row:
            # Found session_id mapping - use it for consistent naming across terminals
            session_id = row['session_id']
            name, data = resolve_instance_name(session_id, get_config().tag)
            # Return instance identity (data may be None if not opted in yet)
            return SenderIdentity(kind='instance', name=name, instance_data=data)

    # No identity available - fallback to external 'john'
    return SenderIdentity(kind='external', name=CLAUDE_SENDER, instance_data=None)


def validate_message(message: str) -> str | None:
    """Validate message size and content. Returns error message or None if valid."""
    if not message or not message.strip():
        return format_error("Message required")

    # Reject control characters (except \n, \r, \t)
    if re.search(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\u0080-\u009F]', message):
        return format_error("Message contains control characters")

    if len(message) > MAX_MESSAGE_SIZE:
        return format_error(f"Message too large (max {MAX_MESSAGE_SIZE} chars)")

    return None
