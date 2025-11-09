"""SQLite event storage - unified database for messages and status events"""
from __future__ import annotations
import sqlite3
import json
import threading
from typing import Any, Optional
from datetime import datetime, timezone

from .paths import hcom_path

# Database configuration
DB_FILE = "hcom.db"
_thread_local = threading.local()  # Per-thread connection storage
_write_lock = threading.Lock()  # Protect concurrent writes

# ==================== Connection Management ====================

def get_db() -> sqlite3.Connection:
    """Get thread-local database connection, creating if needed.

    Returns per-thread connection with WAL mode enabled for concurrent access.
    Each thread gets its own connection to avoid SQLite threading issues.
    """
    if not hasattr(_thread_local, 'conn') or _thread_local.conn is None:
        db_path = hcom_path(DB_FILE)
        _thread_local.conn = sqlite3.connect(str(db_path))
        _thread_local.conn.row_factory = sqlite3.Row

        # Enable foreign key constraints (disabled by default in SQLite)
        _thread_local.conn.execute("PRAGMA foreign_keys = ON")

        # Enable WAL mode for concurrent reads/writes
        _thread_local.conn.execute("PRAGMA journal_mode=WAL")
        _thread_local.conn.execute("PRAGMA wal_autocheckpoint=1000")
        _thread_local.conn.execute("PRAGMA busy_timeout=5000")

        init_db(_thread_local.conn)

    return _thread_local.conn

def close_db() -> None:
    """Close thread-local database connection and clear cache.

    Idempotent - safe to call multiple times or when no connection exists.
    Only closes connection for current thread.
    """
    if hasattr(_thread_local, 'conn') and _thread_local.conn is not None:
        _thread_local.conn.close()
        _thread_local.conn = None

# ==================== Schema Management ====================

def init_db(conn: Optional[sqlite3.Connection] = None) -> None:
    """Create database schema if not exists. Idempotent.

    Schema:
        events(id, timestamp, type, instance, data)
        - id: autoincrement primary key for position tracking
        - timestamp: ISO 8601 datetime for time-based queries
        - type: event type ('message', 'status')
        - instance: instance alias (sender for messages, subject for status)
        - data: JSON blob with type-specific fields

        instances(name, session_id, parent_session_id, ...)
        - name: instance alias (primary key)
        - session_id: unique session identifier (NULL for vanilla)
        - parent_session_id: parent's session_id for subagents
        - current_subagents: JSON array of active subagent names
        - subagent_mappings: JSON object mapping subagent names to session_ids

    Indexes:
        - timestamp for time-range queries
        - type for filtering by event type
        - instance for per-instance queries
        - type+instance composite for common filtered queries
        - session_id, parent_session_id, tag, enabled, created_at, status

    Note: Foreign key constraints are enabled per-connection in get_db(),
    not here (PRAGMA settings are connection-specific).
    """
    if conn is None:
        conn = get_db()

    # Create events table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            type TEXT NOT NULL,
            instance TEXT NOT NULL,
            data TEXT NOT NULL
        )
    """)

    # Create instances table with JSON columns for complex fields
    conn.execute("""
        CREATE TABLE IF NOT EXISTS instances (
            name TEXT PRIMARY KEY,
            session_id TEXT UNIQUE,
            parent_session_id TEXT,
            parent_name TEXT,
            tag TEXT,
            last_event_id INTEGER DEFAULT 0,
            enabled INTEGER DEFAULT 0,
            previously_enabled INTEGER DEFAULT 0,
            status TEXT DEFAULT 'unknown',
            status_time INTEGER DEFAULT 0,
            status_context TEXT DEFAULT '',
            last_stop INTEGER DEFAULT 0,
            directory TEXT,
            created_at REAL NOT NULL,
            transcript_path TEXT DEFAULT '',
            tcp_mode INTEGER DEFAULT 0,
            wait_timeout INTEGER DEFAULT 1800,
            notify_port INTEGER,
            background INTEGER DEFAULT 0,
            background_log_file TEXT DEFAULT '',
            notification_message TEXT DEFAULT '',
            alias_announced INTEGER DEFAULT 0,
            launch_context_announced INTEGER DEFAULT 0,
            external_stop_pending INTEGER DEFAULT 0,
            session_ended INTEGER DEFAULT 0,
            current_subagents TEXT DEFAULT '[]' CHECK(json_valid(current_subagents)),
            subagent_mappings TEXT DEFAULT '{}' CHECK(json_valid(subagent_mappings)),
            FOREIGN KEY (parent_session_id) REFERENCES instances(session_id) ON DELETE SET NULL
        )
    """)

    # Create indexes for common query patterns
    conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON events(timestamp)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_type ON events(type)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_instance ON events(instance)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_type_instance ON events(type, instance)")

    # Create instance indexes
    conn.execute("CREATE INDEX IF NOT EXISTS idx_session_id ON instances(session_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_parent_session_id ON instances(parent_session_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_parent_name ON instances(parent_name)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_tag ON instances(tag)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_enabled ON instances(enabled)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON instances(created_at DESC)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_status ON instances(status)")

    conn.commit()

# ==================== Event Operations ====================

def log_event(
    event_type: str,
    instance: str,
    data: dict[str, Any],
    timestamp: Optional[str] = None
) -> int:
    """Insert event and return its ID.

    Thread-safe: Uses lock to protect concurrent writes.

    Args:
        event_type: Event type ('message', 'status')
        instance: Instance alias (sender for messages, subject for status)
        data: Type-specific event data
        timestamp: Optional ISO 8601 timestamp (defaults to now)

    Returns:
        Event ID (autoincrement primary key)
    """
    conn = get_db()
    if timestamp is None:
        timestamp = datetime.now(timezone.utc).isoformat()

    with _write_lock:
        cursor = conn.execute(
            "INSERT INTO events (timestamp, type, instance, data) VALUES (?, ?, ?, ?)",
            (timestamp, event_type, instance, json.dumps(data))
        )
        conn.commit()
        return cursor.lastrowid

def get_events_since(
    last_event_id: int = 0,
    event_type: Optional[str] = None,
    instance: Optional[str] = None
) -> list[dict[str, Any]]:
    """Query events by ID position with optional filters.

    Args:
        last_event_id: Return events with ID > this value (0 = all events)
        event_type: Optional filter by event type
        instance: Optional filter by instance

    Returns:
        List of events ordered by ID, each with: id, timestamp, type, instance, data (parsed JSON)
    """
    conn = get_db()

    query = "SELECT id, timestamp, type, instance, data FROM events WHERE id > ?"
    params: list[Any] = [last_event_id]

    if event_type is not None:
        query += " AND type = ?"
        params.append(event_type)

    if instance is not None:
        query += " AND instance = ?"
        params.append(instance)

    query += " ORDER BY id"

    cursor = conn.execute(query, params)
    return [
        {
            "id": row["id"],
            "timestamp": row["timestamp"],
            "type": row["type"],
            "instance": row["instance"],
            "data": json.loads(row["data"])
        }
        for row in cursor.fetchall()
    ]

def query_events(
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    event_type: Optional[str] = None,
    instance: Optional[str] = None,
    limit: Optional[int] = None
) -> list[dict[str, Any]]:
    """Query events by time range with optional filters.

    Primary use case: orchestrators querying event snapshots to monitor instances.

    Args:
        start_time: Optional ISO 8601 timestamp (inclusive)
        end_time: Optional ISO 8601 timestamp (exclusive)
        event_type: Optional filter by event type
        instance: Optional filter by instance
        limit: Optional limit on number of results

    Returns:
        List of events ordered by timestamp, each with: id, timestamp, type, instance, data (parsed JSON)
    """
    conn = get_db()

    query = "SELECT id, timestamp, type, instance, data FROM events WHERE 1=1"
    params: list[Any] = []

    if start_time is not None:
        query += " AND timestamp >= ?"
        params.append(start_time)

    if end_time is not None:
        query += " AND timestamp < ?"
        params.append(end_time)

    if event_type is not None:
        query += " AND type = ?"
        params.append(event_type)

    if instance is not None:
        query += " AND instance = ?"
        params.append(instance)

    query += " ORDER BY timestamp"

    if limit is not None:
        query += " LIMIT ?"
        params.append(limit)

    cursor = conn.execute(query, params)
    return [
        {
            "id": row["id"],
            "timestamp": row["timestamp"],
            "type": row["type"],
            "instance": row["instance"],
            "data": json.loads(row["data"])
        }
        for row in cursor.fetchall()
    ]

def get_last_event_id() -> int:
    """Get current maximum event ID.

    Returns:
        Maximum event ID, or 0 if no events exist
    """
    conn = get_db()
    cursor = conn.execute("SELECT MAX(id) FROM events")
    result = cursor.fetchone()[0]
    return result if result is not None else 0

# ==================== Instance Operations ====================

def get_instance(name: str) -> dict[str, Any] | None:
    """Get instance by name. Returns dict or None."""
    conn = get_db()
    row = conn.execute("SELECT * FROM instances WHERE name = ?", (name,)).fetchone()
    if not row:
        return None

    # Convert Row to dict, parse JSON fields
    data = dict(row)
    data['current_subagents'] = json.loads(data['current_subagents'])
    data['subagent_mappings'] = json.loads(data['subagent_mappings'])

    return data

def get_all_instances() -> dict[str, dict[str, Any]]:
    """Get all instances. Returns {name: data} dict."""
    conn = get_db()
    rows = conn.execute("SELECT * FROM instances ORDER BY created_at DESC").fetchall()

    result = {}
    for row in rows:
        data = dict(row)
        # Parse JSON fields
        data['current_subagents'] = json.loads(data['current_subagents'])
        data['subagent_mappings'] = json.loads(data['subagent_mappings'])
        result[row['name']] = data

    return result

def save_instance(name: str, data: dict[str, Any]) -> bool:
    """Insert or update instance using UPSERT. Returns True on success."""
    conn = get_db()

    try:
        with _write_lock:
            # Serialize JSON fields
            data_copy = data.copy()
            if 'current_subagents' in data_copy:
                data_copy['current_subagents'] = json.dumps(data_copy['current_subagents'])
            if 'subagent_mappings' in data_copy:
                data_copy['subagent_mappings'] = json.dumps(data_copy['subagent_mappings'])

            # UPSERT - simpler and race-free
            columns = ', '.join(data_copy.keys())
            placeholders = ', '.join('?' * len(data_copy))
            update_clause = ', '.join(f"{k} = excluded.{k}" for k in data_copy.keys() if k != 'name')

            conn.execute(
                f"""
                INSERT INTO instances ({columns}) VALUES ({placeholders})
                ON CONFLICT(name) DO UPDATE SET {update_clause}
                """,
                tuple(data_copy.values())
            )

            conn.commit()
            return True
    except sqlite3.Error as e:
        import sys
        print(f"DB error saving instance {name}: {e}", file=sys.stderr)
        return False
    except Exception as e:
        import sys
        print(f"Unexpected error saving instance {name}: {e}", file=sys.stderr)
        return False

def update_instance(name: str, updates: dict[str, Any]) -> bool:
    """Update specific instance fields. Returns True on success."""
    if not updates:
        return True

    conn = get_db()

    try:
        with _write_lock:
            # Serialize JSON fields if present
            updates_copy = updates.copy()
            if 'current_subagents' in updates_copy:
                updates_copy['current_subagents'] = json.dumps(updates_copy['current_subagents'])
            if 'subagent_mappings' in updates_copy:
                updates_copy['subagent_mappings'] = json.dumps(updates_copy['subagent_mappings'])

            # Simple UPDATE - JSON columns handled like any other field
            set_clause = ', '.join(f"{k} = ?" for k in updates_copy.keys())
            conn.execute(
                f"UPDATE instances SET {set_clause} WHERE name = ?",
                (*updates_copy.values(), name)
            )

            conn.commit()
            return True
    except sqlite3.Error as e:
        import sys
        print(f"DB error updating instance {name}: {e}", file=sys.stderr)
        return False
    except Exception as e:
        import sys
        print(f"Unexpected error updating instance {name}: {e}", file=sys.stderr)
        return False

def find_instance_by_session(session_id: str) -> str | None:
    """Find instance name by session_id. Returns name or None.

    Note: Vanilla instances have session_id=NULL, not matched by this lookup.
    """
    if not session_id:
        return None  # Don't match empty/None session_id

    conn = get_db()
    row = conn.execute("SELECT name FROM instances WHERE session_id = ?", (session_id,)).fetchone()
    return row['name'] if row else None

def delete_instance(name: str) -> bool:
    """Delete instance. CASCADE handles cleanup. Returns True on success."""
    conn = get_db()
    try:
        with _write_lock:
            conn.execute("DELETE FROM instances WHERE name = ?", (name,))
            conn.commit()
            return True
    except Exception:
        return False

# ==================== High-Level Query Helpers ====================

def iter_instances(enabled_only: bool = False):
    """Iterate instances efficiently (generator, not giant dict)."""
    conn = get_db()
    query = "SELECT * FROM instances"
    if enabled_only:
        query += " WHERE enabled = 1"
    query += " ORDER BY created_at DESC"

    for row in conn.execute(query):
        data = dict(row)
        data['current_subagents'] = json.loads(data['current_subagents'])
        data['subagent_mappings'] = json.loads(data['subagent_mappings'])
        yield data


def list_instances_for_watch(limit: int = 0):
    """Get instances for watch/list commands (structured query)."""
    conn = get_db()
    query = "SELECT name, enabled, status, status_time, status_context FROM instances WHERE previously_enabled = 1 ORDER BY created_at DESC"
    if limit > 0:
        query += f" LIMIT {limit}"

    return conn.execute(query).fetchall()

__all__ = [
    # Events
    'get_db',
    'close_db',
    'init_db',
    'log_event',
    'get_events_since',
    'query_events',
    'get_last_event_id',
    'DB_FILE',
    # Instances (low-level)
    'get_instance',
    'get_all_instances',
    'save_instance',
    'update_instance',
    'find_instance_by_session',
    'delete_instance',
    # Instances (high-level queries)
    'iter_instances',
    'list_instances_for_watch',
]