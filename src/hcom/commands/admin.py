"""Admin commands for HCOM"""
import sys
import json
import time
import shutil
from pathlib import Path
from datetime import datetime
from typing import Any
from .utils import get_help_text, format_error
from ..core.paths import hcom_path, SCRIPTS_DIR, LOGS_DIR, ARCHIVE_DIR
from ..core.instances import get_instance_status, is_external_sender
from ..shared import STATUS_ICONS


def get_archive_timestamp() -> str:
    """Get timestamp for archive files"""
    return datetime.now().strftime("%Y-%m-%d_%H%M%S")


def should_show_in_watch(d: dict[str, Any]) -> bool:
    """Show previously-enabled instances, hide vanilla never-enabled instances"""
    # Hide instances that never participated
    if not d.get('previously_enabled', False):
        return False
    return True


def cmd_help() -> int:
    """Show help text"""
    print(get_help_text())
    return 0


def cmd_watch(argv: list[str]) -> int:
    """Query events from SQLite: hcom watch [--last N] [--wait SEC] [--sql EXPR]"""
    from ..core.db import get_db, init_db, get_last_event_id

    init_db()  # Ensure schema exists

    # Parse arguments
    last_n = 20  # Default: last 20 events
    wait_timeout = None
    sql_where = None

    i = 0
    while i < len(argv):
        if argv[i] == '--last' and i + 1 < len(argv):
            try:
                last_n = int(argv[i + 1])
            except ValueError:
                print(f"Error: --last must be an integer, got '{argv[i + 1]}'", file=sys.stderr)
                return 1
            i += 2
        elif argv[i] == '--wait':
            if i + 1 < len(argv) and not argv[i + 1].startswith('--'):
                try:
                    wait_timeout = int(argv[i + 1])
                except ValueError:
                    print(f"Error: --wait must be an integer, got '{argv[i + 1]}'", file=sys.stderr)
                    return 1
                i += 2
            else:
                wait_timeout = 60  # Default: 60 seconds
                i += 1
        elif argv[i] == '--sql' and i + 1 < len(argv):
            sql_where = argv[i + 1]
            i += 2
        else:
            i += 1

    # Build base query for filters
    db = get_db()
    filter_query = ""

    # Add user SQL WHERE clause directly (no validation needed)
    # Note: SQL injection is not a security concern in hcom's threat model.
    # User (or ai) owns ~/.hcom/hcom.db and can already run: sqlite3 ~/.hcom/hcom.db "anything"
    # Validation would block legitimate queries while providing no actual security.
    if sql_where:
        filter_query += f" AND ({sql_where})"

    # Wait mode: block until matching event or timeout
    if wait_timeout:
        # Check for matching events in last 10s (race condition window)
        from datetime import timezone
        lookback_timestamp = datetime.fromtimestamp(time.time() - 10, tz=timezone.utc).isoformat()
        lookback_query = f"SELECT * FROM events WHERE timestamp > ?{filter_query} ORDER BY id DESC LIMIT 1"

        try:
            lookback_row = db.execute(lookback_query, [lookback_timestamp]).fetchone()
        except Exception as e:
            print(f"Error in SQL WHERE clause: {e}", file=sys.stderr)
            return 2

        if lookback_row:
            try:
                event = {
                    'ts': lookback_row['timestamp'],
                    'type': lookback_row['type'],
                    'instance': lookback_row['instance'],
                    'data': json.loads(lookback_row['data'])
                }
                # Found recent matching event, return immediately
                print(json.dumps(event))
                return 0
            except (json.JSONDecodeError, TypeError):
                pass  # Ignore corrupt event, continue to wait loop

        start_time = time.time()
        last_id = get_last_event_id()

        while time.time() - start_time < wait_timeout:
            query = f"SELECT * FROM events WHERE id > ?{filter_query} ORDER BY id"

            try:
                rows = db.execute(query, [last_id]).fetchall()
            except Exception as e:
                print(f"Error in SQL WHERE clause: {e}", file=sys.stderr)
                return 2

            if rows:
                # Process matching events
                for row in rows:
                    try:
                        event = {
                            'ts': row['timestamp'],
                            'type': row['type'],
                            'instance': row['instance'],
                            'data': json.loads(row['data'])
                        }

                        # Event matches all conditions, print and exit
                        print(json.dumps(event))
                        return 0

                    except (json.JSONDecodeError, TypeError) as e:
                        # Skip corrupt events, log to stderr
                        print(f"Warning: Skipping corrupt event ID {row['id']}: {e}", file=sys.stderr)
                        continue

                # All events processed, update last_id and continue waiting
                last_id = rows[-1]['id']

            # Check if current instance received @mention (interrupt wait)
            from .utils import resolve_identity
            from ..core.messages import get_unread_messages

            identity = resolve_identity()
            if identity.kind == 'instance':
                messages, _ = get_unread_messages(identity.name, update_position=False)
                if messages:
                    # Interrupted by @mention, exit early
                    # PostToolUse will deliver the message
                    return 3

            time.sleep(0.1)

        return 1  # Timeout, no matches

    # Snapshot mode (default)
    query = "SELECT * FROM events WHERE 1=1"
    query += filter_query
    query += " ORDER BY id DESC"
    query += f" LIMIT {last_n}"

    try:
        rows = db.execute(query).fetchall()
    except Exception as e:
        print(f"Error in SQL WHERE clause: {e}", file=sys.stderr)
        return 2
    # Reverse to chronological order
    for row in reversed(rows):
        try:
            event = {
                'ts': row['timestamp'],
                'type': row['type'],
                'instance': row['instance'],
                'data': json.loads(row['data'])
            }
            print(json.dumps(event))
        except (json.JSONDecodeError, TypeError) as e:
            # Skip corrupt events, log to stderr
            print(f"Warning: Skipping corrupt event ID {row['id']}: {e}", file=sys.stderr)
            continue
    return 0


def cmd_list(argv: list[str]) -> int:
    """List instances: hcom list [--json] [-v|--verbose]"""
    from .utils import resolve_identity
    from ..core.instances import load_instance_position
    from ..core.messages import get_read_receipts
    from ..core.db import get_db
    from ..shared import SENDER, CLAUDE_SENDER

    # Parse arguments
    json_output = False
    verbose_output = False

    for arg in argv:
        if arg == '--json':
            json_output = True
        elif arg in ['-v', '--verbose']:
            verbose_output = True

    # Resolve current instance identity
    identity = resolve_identity()
    current_alias = identity.name
    sender_identity = identity

    # Load read receipts for all contexts (bigboss, john, instances)
    # Limit based on verbose flag: 3 messages if verbose, 1 otherwise
    read_limit = 3 if verbose_output else 1
    read_receipts = get_read_receipts(sender_identity, limit=read_limit)

    # Only show connection status for actual instances (not CLI/fallback)
    show_connection = current_alias not in (SENDER, CLAUDE_SENDER)
    current_enabled = False
    if show_connection:
        current_data = load_instance_position(current_alias)
        current_enabled = current_data.get('enabled', False) if current_data else False

    # Query all instances
    db = get_db()
    query = "SELECT * FROM instances WHERE previously_enabled = 1 ORDER BY created_at DESC"
    rows = db.execute(query).fetchall()

    # Convert rows to dictionaries
    sorted_instances = [dict(row) for row in rows]

    if json_output:
        # JSON per line - _self entry first always
        self_payload = {
            "_self": {
                "alias": current_alias,
                "read_receipts": read_receipts
            }
        }
        # Only include connection status for actual instances
        if show_connection:
            self_payload["_self"]["hcom_connected"] = current_enabled
        print(json.dumps(self_payload))

        for data in sorted_instances:
            if not should_show_in_watch(data):
                continue
            name = data['name']
            enabled, status, age_str, description, age_seconds = get_instance_status(data)
            payload = {
                name: {
                    "hcom_connected": enabled,
                    "status": status,
                    "status_age_seconds": int(age_seconds),
                    "description": description,
                    "headless": bool(data.get("background", False)),
                    "wait_timeout": data.get("wait_timeout", 1800),
                    "session_id": data.get("session_id", ""),
                    "directory": data.get("directory", ""),
                    "parent_name": data.get("parent_name") or None,
                    "agent_id": data.get("agent_id") or None,
                    "background_log_file": data.get("background_log_file") or None,
                    "transcript_path": data.get("transcript_path") or None,
                    "created_at": data.get("created_at"),
                    "tcp_mode": bool(data.get("tcp_mode", False)),
                }
            }
            print(json.dumps(payload))
    else:
        # Human-readable - show header with alias and read receipts
        print(f"Your alias: {current_alias}")

        # Show connection status only for actual instances (not bigboss/john)
        if show_connection:
            state_symbol = "+" if current_enabled else "-"
            state_text = "enabled" if current_enabled else "disabled"
            print(f"  Your hcom connection: {state_text} ({state_symbol})")

        # Show read receipts if any
        if read_receipts:
            print(f"  Read receipts:")
            for msg in read_receipts:
                read_count = len(msg['read_by'])
                total = msg['total_recipients']

                if verbose_output:
                    # Verbose: show list of who has read + ratio
                    readers = ", ".join(msg['read_by']) if msg['read_by'] else "(none)"
                    print(f"    #{msg['id']} {msg['age']} \"{msg['text']}\" | read by ({read_count}/{total}): {readers}")
                else:
                    # Default: just show ratio
                    print(f"    #{msg['id']} {msg['age']} \"{msg['text']}\" | read by {read_count}/{total}")

        print()

        for data in sorted_instances:
            if not should_show_in_watch(data):
                continue
            name = data['name']
            enabled, status, age_str, description, age_seconds = get_instance_status(data)
            icon = STATUS_ICONS.get(status, '◦')
            state = "+" if enabled else "-"
            age_display = f"{age_str} ago" if age_str else ""
            desc_sep = ": " if description else ""

            # Add badges
            headless_badge = "[headless]" if data.get("background", False) else ""
            external_badge = "[external]" if is_external_sender(data) else ""
            badge_parts = [b for b in [headless_badge, external_badge] if b]
            badge_str = (" " + " ".join(badge_parts)) if badge_parts else ""
            name_with_badges = f"{name}{badge_str}"

            # Main status line
            print(f"{icon} {name_with_badges:30} {state}  {age_display}{desc_sep}{description}")

            if verbose_output:
                # Multi-line detailed view
                import os

                # Format fields
                session_id = data.get("session_id", "(none)")
                directory = data.get("directory", "(none)")
                timeout = data.get("wait_timeout", 1800)

                parent = data.get("parent_name") or "(none)"

                # Format paths (shorten with ~)
                log_file = data.get("background_log_file")
                if log_file:
                    log_file = log_file.replace(os.path.expanduser("~"), "~")
                else:
                    log_file = "(none)"

                transcript = data.get("transcript_path")
                if transcript:
                    transcript = transcript.replace(os.path.expanduser("~"), "~")
                else:
                    transcript = "(none)"

                # Format created_at timestamp
                created_ts = data.get("created_at")
                if created_ts:
                    created_seconds = time.time() - created_ts
                    if created_seconds < 60:
                        created = f"{int(created_seconds)}s ago"
                    elif created_seconds < 3600:
                        created = f"{int(created_seconds / 60)}m ago"
                    elif created_seconds < 86400:
                        created = f"{int(created_seconds / 3600)}h ago"
                    else:
                        created = f"{int(created_seconds / 86400)}d ago"
                else:
                    created = "(unknown)"

                # Format tcp_mode
                tcp = "TCP" if data.get("tcp_mode") else "polling"

                # Get subagent agentId if this is a subagent
                agent_id = None
                if parent != "(none)":
                    # This is a subagent - get agentId from its own data
                    agent_id = data.get("agent_id") or "(none)"

                # Print indented details
                print(f"    session_id:   {session_id}")
                print(f"    created:      {created}")
                print(f"    directory:    {directory}")
                print(f"    timeout:      {timeout}s")
                if parent != "(none)":
                    print(f"    parent:       {parent}")
                    print(f"    agent_id:     {agent_id}")
                print(f"    tcp_mode:     {tcp}")
                if log_file != "(none)":
                    print(f"    headless log: {log_file}")
                print(f"    transcript:   {transcript}")
                print()  # Blank line between instances


    return 0


def clear() -> int:
    """Clear and archive conversation"""
    from ..core.db import DB_FILE, close_db, get_db

    db_file = hcom_path(DB_FILE)
    db_wal = hcom_path(f'{DB_FILE}-wal')
    db_shm = hcom_path(f'{DB_FILE}-shm')

    # cleanup: temp files, old scripts, old background logs
    cutoff_time_24h = time.time() - (24 * 60 * 60)  # 24 hours ago
    cutoff_time_30d = time.time() - (30 * 24 * 60 * 60)  # 30 days ago

    scripts_dir = hcom_path(SCRIPTS_DIR)
    if scripts_dir.exists():
        for f in scripts_dir.glob('*'):
            if f.is_file() and f.stat().st_mtime < cutoff_time_24h:
                f.unlink(missing_ok=True)

    # Rotate hooks.log at 1MB
    logs_dir = hcom_path(LOGS_DIR)
    hooks_log = logs_dir / 'hooks.log'
    if hooks_log.exists() and hooks_log.stat().st_size > 1_000_000:  # 1MB
        archive_logs = logs_dir / f'hooks.log.{get_archive_timestamp()}'
        hooks_log.rename(archive_logs)

    # Clean background logs older than 30 days
    if logs_dir.exists():
        for f in logs_dir.glob('background_*.log'):
            if f.stat().st_mtime < cutoff_time_30d:
                f.unlink(missing_ok=True)

    # Check if DB exists
    if not db_file.exists():
        print("No HCOM conversation to clear")
        return 0

    # Archive database if it has content
    timestamp = get_archive_timestamp()
    archived = False

    try:
        # Check if DB has content
        db = get_db()
        event_count = db.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        instance_count = db.execute("SELECT COUNT(*) FROM instances").fetchone()[0]

        if event_count > 0 or instance_count > 0:
            # Create session archive folder with timestamp
            session_archive = hcom_path(ARCHIVE_DIR, f'session-{timestamp}')
            session_archive.mkdir(parents=True, exist_ok=True)

            # Checkpoint WAL before archiving (attempts to consolidate WAL into main DB)
            # Using PASSIVE mode - doesn't force if writers active
            db.execute("PRAGMA wal_checkpoint(PASSIVE)")
            db.commit()
            close_db()

            # Copy all DB files to archive (DB + WAL + SHM)
            # This preserves WAL data in case checkpoint was incomplete
            # SQLite can recover from WAL when opening archived DB
            shutil.copy2(db_file, session_archive / DB_FILE)
            if db_wal.exists():
                shutil.copy2(db_wal, session_archive / f'{DB_FILE}-wal')
            if db_shm.exists():
                shutil.copy2(db_shm, session_archive / f'{DB_FILE}-shm')

            # Delete main DB and WAL/SHM files
            db_file.unlink()
            db_wal.unlink(missing_ok=True)
            db_shm.unlink(missing_ok=True)

            archived = True
        else:
            # Empty DB, just delete
            close_db()
            db_file.unlink()
            db_wal.unlink(missing_ok=True)
            db_shm.unlink(missing_ok=True)

        if archived:
            print(f"Archived to archive/session-{timestamp}/")
        print("Started fresh HCOM conversation")
        return 0

    except Exception as e:
        print(format_error(f"Failed to archive: {e}"), file=sys.stderr)
        return 1


def remove_global_hooks() -> bool:
    """Remove HCOM hooks from ~/.claude/settings.json"""
    from ..hooks.settings import get_claude_settings_path, load_settings_json, _remove_hcom_hooks_from_settings
    from ..core.paths import atomic_write

    settings_path = get_claude_settings_path()

    if not settings_path.exists():
        return True

    try:
        settings = load_settings_json(settings_path, default=None)
        if not settings:
            return False

        _remove_hcom_hooks_from_settings(settings)
        atomic_write(settings_path, json.dumps(settings, indent=2))
        return True
    except Exception:
        return False


def cmd_reset(argv: list[str]) -> int:
    """Reset HCOM components: logs, hooks, config
    Usage:
        hcom reset              # Everything (stop all + logs + hooks + config)
        hcom reset logs         # Archive conversation only
        hcom reset hooks        # Remove hooks only
        hcom reset config       # Clear config (archive to archive/config/)
        hcom reset logs hooks   # Combine targets
    """
    # Import from lifecycle for cmd_stop
    from .lifecycle import cmd_stop
    from ..core.paths import CONFIG_FILE

    # No args = everything
    do_everything = not argv
    targets = argv if argv else ['logs', 'hooks', 'config']

    # Validate targets
    valid = {'logs', 'hooks', 'config'}
    invalid = [t for t in targets if t not in valid]
    if invalid:
        print(f"Invalid target(s): {', '.join(invalid)}", file=sys.stderr)
        print("Valid targets: logs, hooks, config", file=sys.stderr)
        return 1

    exit_codes = []

    # Stop all instances if doing everything
    if do_everything:
        exit_codes.append(cmd_stop(['all']))

    # Execute based on targets
    if 'logs' in targets:
        exit_codes.append(clear())

    if 'hooks' in targets:
        if remove_global_hooks():
            print("Removed hooks")
            exit_codes.append(0)
        else:
            print("Warning: Could not remove hooks. Check your claude settings.json file it might be invalid", file=sys.stderr)
            exit_codes.append(1)

    if 'config' in targets:
        config_path = hcom_path(CONFIG_FILE)
        if config_path.exists():
            # Archive with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            archive_config_dir = hcom_path(ARCHIVE_DIR, 'config')
            archive_config_dir.mkdir(parents=True, exist_ok=True)
            archive_path = archive_config_dir / f'config.env.{timestamp}'
            shutil.copy2(config_path, archive_path)
            config_path.unlink()
            print(f"Config archived to archive/config/config.env.{timestamp} and cleared")
            exit_codes.append(0)
        else:
            print("No config file to clear")
            exit_codes.append(0)

    return max(exit_codes) if exit_codes else 0
