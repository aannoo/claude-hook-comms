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
from ..core.instances import get_instance_status
from ..core.db import iter_instances
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
    """Query events from SQLite: hcom watch [--type TYPE] [--instance NAME] [--last N] [--wait SEC]"""
    from ..core.db import get_db, init_db, get_last_event_id

    init_db()  # Ensure schema exists

    # Parse arguments
    event_type = None
    instance_filter = None
    last_n = 20  # Default: last 20 events
    wait_timeout = None

    i = 0
    while i < len(argv):
        if argv[i] == '--type' and i + 1 < len(argv):
            event_type = argv[i + 1]
            i += 2
        elif argv[i] == '--instance' and i + 1 < len(argv):
            instance_filter = argv[i + 1]
            i += 2
        elif argv[i] == '--last' and i + 1 < len(argv):
            last_n = int(argv[i + 1])
            i += 2
        elif argv[i] == '--wait' and i + 1 < len(argv):
            wait_timeout = int(argv[i + 1])
            i += 2
        else:
            i += 1

    # Build base query for filters
    db = get_db()
    filter_query = ""
    params = []

    if event_type:
        filter_query += " AND type = ?"
        params.append(event_type)

    if instance_filter:
        filter_query += " AND instance = ?"
        params.append(instance_filter)

    # Wait mode: block until matching event or timeout
    if wait_timeout:
        start_time = time.time()
        last_id = get_last_event_id()

        while time.time() - start_time < wait_timeout:
            query = f"SELECT * FROM events WHERE id > ?{filter_query} ORDER BY id"
            rows = db.execute(query, [last_id] + params).fetchall()

            if rows:
                # Print matching events and exit success
                for row in rows:
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

            time.sleep(0.1)

        return 1  # Timeout, no matches

    # Snapshot mode (default)
    query = "SELECT * FROM events WHERE 1=1"
    query += filter_query
    query += " ORDER BY id DESC"
    query += f" LIMIT {last_n}"

    rows = db.execute(query, params).fetchall()
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
    """List instances: hcom list [--json] [--verbose]"""
    from .utils import resolve_identity
    from ..core.instances import load_instance_position
    from ..core.messages import get_read_receipts
    from ..shared import SENDER, CLAUDE_SENDER

    json_output = '--json' in argv
    verbose_output = '--verbose' in argv

    # Resolve current instance identity
    current_alias = resolve_identity()

    # Load read receipts for all contexts (bigboss, john, instances)
    # Limit based on verbose flag: 3 messages if verbose, 1 otherwise
    read_limit = 3 if verbose_output else 1
    read_receipts = get_read_receipts(current_alias, limit=read_limit)

    # Only show connection status for actual instances (not CLI/fallback)
    show_connection = current_alias not in (SENDER, CLAUDE_SENDER)
    current_enabled = False
    if show_connection:
        current_data = load_instance_position(current_alias)
        current_enabled = current_data.get('enabled', False) if current_data else False

    # Sort by creation time (newest first) - same as TUI
    sorted_instances = sorted(
        iter_instances(),
        key=lambda x: -x.get('created_at', 0.0)
    )

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
                }
            }
            if verbose_output:
                payload[name]["session_id"] = data.get("session_id", "")
                payload[name]["directory"] = data.get("directory", "")
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

            # Add [headless] badge
            headless_badge = " [headless]" if data.get("background", False) else ""

            if verbose_output:
                directory = data.get("directory", "unknown")
                session_id = data.get("session_id", "")
                timeout = data.get("wait_timeout", 1800)
                print(f"{icon} {name:15}{headless_badge} {state}  {age_display}{desc_sep}{description} (dir: {directory}, session_id: {session_id}, timeout: {timeout}s)")
            else:
                print(f"{icon} {name:15}{headless_badge} {state}  {age_display}{desc_sep}{description}")

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
        sum(1 for f in scripts_dir.glob('*') if f.is_file() and f.stat().st_mtime < cutoff_time_24h and f.unlink(missing_ok=True) is None)

    # Rotate hooks.log at 1MB
    logs_dir = hcom_path(LOGS_DIR)
    hooks_log = logs_dir / 'hooks.log'
    if hooks_log.exists() and hooks_log.stat().st_size > 1_000_000:  # 1MB
        archive_logs = logs_dir / f'hooks.log.{get_archive_timestamp()}'
        hooks_log.rename(archive_logs)

    # Clean background logs older than 30 days
    if logs_dir.exists():
        sum(1 for f in logs_dir.glob('background_*.log') if f.stat().st_mtime < cutoff_time_30d and f.unlink(missing_ok=True) is None)

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
