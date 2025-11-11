"""Command utilities for HCOM"""
import sys
import re
from ..shared import __version__, MAX_MESSAGE_SIZE


class CLIError(Exception):
    """Raised when arguments cannot be mapped to command semantics."""


def get_help_text() -> str:
    """Generate help text with current version"""
    return f"""hcom {__version__}
Hook-based communication bus for real-time messaging between Claude Code instances

Usage: hcom                           # TUI dashboard
       [ENV_VARS] hcom <COUNT> [claude <ARGS>...]
       hcom watch [--type TYPE] [--instance NAME] [--last N] [--wait SEC]
       hcom list [--json] [--verbose]
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
  watch               Query recent events (JSON per line)
    --type TYPE       Filter by event type (message, status)
    --instance ALIAS  Filter by instance
    --last N          Limit to last N events (default: 20)
    --wait SEC        Block until a matching event arrives

  list                Show instance status/metadata
    --json            Emit JSON (one instance per line)
    --verbose         Include additional metadata

  send "msg"          Send message to all instances
  send "@alias msg"   Send to specific instance/group
    --from <name>     Custom external identity

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


def resolve_identity(subagent_id: str | None = None) -> str:
    """Resolve identity in CLI/non-hook context. Returns instance name, 'bigboss', or 'john'"""
    import os
    from ..shared import MAPID

    # Subagent explicit (Task tool)
    if subagent_id:
        from ..core.instances import load_instance_position
        data = load_instance_position(subagent_id)
        if not data:
            raise ValueError(f"Subagent {subagent_id} not found")
        return subagent_id

    # CLI context
    if os.environ.get('CLAUDECODE') != '1':
        return 'bigboss'

    # Inside Claude: try session_id, mapid, fallback to john
    session_id = os.environ.get('HCOM_SESSION_ID')
    if session_id:
        from ..core.instances import resolve_instance_name
        from ..core.config import get_config
        name, data = resolve_instance_name(session_id, get_config().tag)
        if data:
            return name

    if MAPID:
        from ..core.db import get_instance_by_mapid
        data = get_instance_by_mapid(MAPID)
        if data:
            return data['name']

    # No identity available
    return 'john'


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
