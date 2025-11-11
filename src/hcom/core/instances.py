"""Instance state management - tracking, status, and group membership"""
from __future__ import annotations
from pathlib import Path
from typing import Any
import time
import os
import sys

from ..shared import format_age

# Configuration
SKIP_HISTORY = True  # New instances start at current log position (skip old messages)

# ==================== Core Instance I/O ====================

def load_instance_position(instance_name: str) -> dict[str, Any]:
    """Load position data for a single instance (DB wrapper)"""
    from .db import get_instance
    data = get_instance(instance_name)
    return data if data else {}

def save_instance_position(instance_name: str, data: dict[str, Any]) -> bool:
    """Save instance data with smart UPSERT (merge with existing or create new).

    SEMANTICS:
    - If instance exists: Merge data into existing (preserves unspecified fields)
    - If instance new: Fill defaults for missing fields

    This preserves backwards compatibility with callers passing partial dicts.

    Returns True on success, False on failure.
    """
    from .db import save_instance, get_instance

    # Check if instance exists
    existing = get_instance(instance_name)

    if existing:
        # MERGE: Update only provided fields, preserve existing values
        merged = existing.copy()
        merged.update(data)
        complete_data = merged
    else:
        # NEW: Fill defaults for missing fields
        defaults = {
            "name": instance_name,
            "session_id": None,
            "mapid": "",
            "parent_session_id": None,
            "parent_name": None,
            "created_at": time.time(),
            "directory": str(Path.cwd()),
            "last_event_id": 0,
            "enabled": 0,
            "previously_enabled": 0,
            "status": "unknown",
            "status_time": 0,
            "status_context": "",
            "last_stop": 0,
            "transcript_path": "",
            "tcp_mode": 0,
            "wait_timeout": 1800,
            "notify_port": None,
            "background": 0,
            "background_log_file": "",
            "notification_message": "",
            "alias_announced": 0,
            "launch_context_announced": 0,
            "external_stop_pending": 0,
            "session_ended": 0,
            "current_subagents": [],
            "subagent_mappings": {}
        }
        # Merge: data overrides defaults
        complete_data = {**defaults, **data}

    # Convert booleans to integers for SQLite
    for bool_field in ['enabled', 'previously_enabled', 'tcp_mode', 'background',
                       'alias_announced', 'launch_context_announced',
                       'external_stop_pending', 'session_ended']:
        if bool_field in complete_data and isinstance(complete_data[bool_field], bool):
            complete_data[bool_field] = int(complete_data[bool_field])

    return save_instance(instance_name, complete_data)

def update_instance_position(instance_name: str, update_fields: dict[str, Any]) -> None:
    """Update instance position atomically (DB wrapper)

    Creates instance with defaults if doesn't exist (auto-vivification).
    """
    from .db import update_instance, get_instance
    from ..hooks.utils import log_hook_error

    try:
        # Auto-vivify if needed
        if not get_instance(instance_name):
            initialize_instance_in_position_file(instance_name)

        # Convert booleans to integers for SQLite
        update_copy = update_fields.copy()
        for bool_field in ['enabled', 'previously_enabled', 'tcp_mode', 'background',
                           'alias_announced', 'launch_context_announced',
                           'external_stop_pending', 'session_ended']:
            if bool_field in update_copy and isinstance(update_copy[bool_field], bool):
                update_copy[bool_field] = int(update_copy[bool_field])

        update_instance(instance_name, update_copy)
    except Exception as e:
        log_hook_error(f'update_instance_position:{instance_name}', e)
        pass  # Silent to user, logged for debugging

# ==================== Instance Helper Functions ====================

def is_parent_instance(instance_data: dict[str, Any] | None) -> bool:
    """Check if instance is a parent (has session_id, no parent_session_id)"""
    if not instance_data:
        return False
    has_session = bool(instance_data.get('session_id'))
    has_parent = bool(instance_data.get('parent_session_id'))
    return has_session and not has_parent

def is_subagent_instance(instance_data: dict[str, Any] | None) -> bool:
    """Check if instance is a subagent (has parent_session_id)"""
    if not instance_data:
        return False
    return bool(instance_data.get('parent_session_id'))

def get_group_session_id(instance_data: dict[str, Any] | None) -> str | None:
    """Get the session_id that defines this instance's group.
    For parents: their own session_id, for subagents: parent_session_id
    """
    if not instance_data:
        return None
    # Subagent - use parent_session_id
    if parent_sid := instance_data.get('parent_session_id'):
        return parent_sid
    # Parent - use own session_id
    return instance_data.get('session_id')

def in_same_group(sender_data: dict[str, Any] | None, receiver_data: dict[str, Any] | None) -> bool:
    """Check if sender and receiver are in same group (share session_id)"""
    sender_group = get_group_session_id(sender_data)
    receiver_group = get_group_session_id(receiver_data)
    if not sender_group or not receiver_group:
        return False
    return sender_group == receiver_group

def in_subagent_context(instance_data: dict[str, Any] | None) -> bool:
    """Check if hook (or any code) is being called from subagent context (not parent).
    Returns True when parent has current_subagents list, meaning:
    - A subagent is calling this code (parent is frozen during Task)
    - instance_data is the parent's data (hooks resolve to parent session_id)
    """
    if not instance_data:
        return False
    return bool(instance_data.get('current_subagents'))

# ==================== Status Functions ====================

def get_instance_status(pos_data: dict[str, Any]) -> tuple[bool, str, str, str, int]:
    """Get current status of instance. Returns (enabled, status, age_string, description, age_seconds).

    age_string format: "16m" (clean format, no parens/suffix - consumers handle display)
    age_seconds: raw integer seconds for programmatic filtering

    Status is activity state (what instance is doing).
    Enabled is participation flag (whether instance can send/receive HCOM).
    These are orthogonal - can be disabled but still active.
    """
    enabled = pos_data.get('enabled', False)
    status = pos_data.get('status', 'unknown')
    status_time = pos_data.get('status_time', 0)
    status_context = pos_data.get('status_context', '')

    now = int(time.time())
    age = now - status_time if status_time else 0

    # Subagent-specific status detection
    if pos_data.get('parent_session_id'):
        # Subagent in done polling loop: status='active' but heartbeat still updating
        # PreToolUse sets all subagents to 'active', but one in polling loop has fresh heartbeat
        if status == 'active' and enabled:
            heartbeat_age = now - pos_data.get('last_stop', 0)
            if heartbeat_age < 1.5:  # Heartbeat active (1s poll interval + margin)
                status = 'waiting'
                age = heartbeat_age

    # Heartbeat timeout check: instance was waiting but heartbeat died
    # This detects terminated instances (closed window/crashed) that were idle
    if status == 'waiting':
        heartbeat_age = now - pos_data.get('last_stop', 0)
        tcp_mode = pos_data.get('tcp_mode', False)
        threshold = 40 if tcp_mode else 2
        if heartbeat_age > threshold:
            status_context = status  # Save what it was doing
            status = 'stale'
            age = heartbeat_age

    # Activity timeout check: no status updates for extended period
    # This detects terminated instances that were active/blocked/etc when closed
    if status not in ['exited', 'stale']:
        timeout = pos_data.get('wait_timeout', 1800)
        min_threshold = max(timeout + 60, 600)  # Timeout + 1min buffer, minimum 10min
        status_age = now - status_time if status_time else 0
        if status_age > min_threshold:
            status_context = status  # Save what it was doing
            status = 'stale'
            age = status_age

    # Build description from status and context
    description = get_status_description(status, status_context)

    return (enabled, status, format_age(age), description, age)


def get_status_description(status: str, context: str = '') -> str:
    """Build human-readable status description"""
    if status == 'active':
        return f"{context} executing" if context else "active"
    elif status == 'delivered':
        return f"msg from {context}" if context else "message delivered"
    elif status == 'waiting':
        return "idle"
    elif status == 'blocked':
        return f"{context}" if context else "permission needed"
    elif status == 'exited':
        return f"exited: {context}" if context else "exited"
    elif status == 'stale':
        # Show what it was doing when it went stale
        if context == 'waiting':
            return "idle [stale]"
        elif context == 'active':
            return "active [stale]"
        elif context == 'blocked':
            return "blocked [stale]"
        elif context == 'delivered':
            return "delivered [stale]"
        else:
            return "stale"
    else:
        return "unknown"

def set_status(instance_name: str, status: str, context: str = ''):
    """Set instance status with timestamp and log status change event"""
    # Update instance file
    update_instance_position(instance_name, {
        'status': status,
        'status_time': int(time.time()),
        'status_context': context
    })

    # Log status change event (best-effort, non-blocking)
    try:
        from .db import log_event
        log_event(
            event_type='status',
            instance=instance_name,
            data={
                'status': status,
                'context': context
            }
        )
    except Exception:
        pass  # Don't break hooks if event logging fails

# ==================== Identity Management ====================

def get_display_name(session_id: str | None, tag: str | None = None, collision_attempt: int = 0) -> str:
    """Get display name for instance using session_id deterministically.

    Args:
        session_id: Session ID to hash (required)
        tag: Optional tag prefix
        collision_attempt: Collision counter for race resolution (default 0)

    Returns:
        Generated name (may already exist in DB - caller must check)
    """
    if not session_id:
        raise ValueError("session_id required for instance naming")

    # ~90 recognizable 3-letter words
    words = [
        'ace', 'air', 'ant', 'arm', 'art', 'axe', 'bad', 'bag', 'bar', 'bat',
        'bed', 'bee', 'big', 'box', 'boy', 'bug', 'bus', 'cab', 'can', 'cap',
        'car', 'cat', 'cop', 'cow', 'cry', 'cup', 'cut', 'day', 'dog', 'dry',
        'ear', 'egg', 'eye', 'fan', 'pig', 'fly', 'fox', 'fun', 'gem', 'gun',
        'hat', 'hit', 'hot', 'ice', 'ink', 'jet', 'key', 'law', 'map', 'mix',
        'man', 'bob', 'noo', 'yes', 'poo', 'sue', 'tom', 'the', 'and', 'but',
        'age', 'aim', 'bro', 'bid', 'shi', 'buy', 'den', 'dig', 'dot', 'dye',
        'end', 'era', 'eve', 'few', 'fix', 'gap', 'gas', 'god', 'gym', 'nob',
        'hip', 'hub', 'hug', 'ivy', 'jab', 'jam', 'jay', 'jog', 'joy', 'lab',
        'lag', 'lap', 'leg', 'lid', 'lie', 'log', 'lot', 'mat', 'mop', 'mud',
        'net', 'new', 'nod', 'now', 'oak', 'odd', 'off', 'oil', 'old', 'one',
        'lol', 'owe', 'own', 'pad', 'pan', 'pat', 'pay', 'pea', 'pen', 'pet',
        'pie', 'pig', 'pin', 'pit', 'pot', 'pub', 'nah', 'rag', 'ran', 'rap',
        'rat', 'raw', 'red', 'rib', 'rid', 'rip', 'rod', 'row', 'rub', 'rug',
        'run', 'sad', 'sap', 'sat', 'saw', 'say', 'sea', 'set', 'wii', 'she',
        'shy', 'sin', 'sip', 'sir', 'sit', 'six', 'ski', 'sky', 'sly', 'son',
        'boo', 'soy', 'spa', 'spy', 'rat', 'sun', 'tab', 'tag', 'tan', 'tap',
        'pls', 'tax', 'tea', 'ten', 'tie', 'tip', 'toe', 'ton', 'top', 'toy',
        'try', 'tub', 'two', 'use', 'van', 'bum', 'war', 'wax', 'way', 'web',
        'wed', 'wet', 'who', 'why', 'wig', 'win', 'moo', 'won', 'wow', 'yak',
        'too', 'gay', 'yet', 'you', 'zip', 'zoo', 'ann'
    ]

    # Hash to select word and suffix (with collision entropy)
    hash_val = sum(ord(c) * (i + collision_attempt) for i, c in enumerate(session_id))
    word = words[hash_val % len(words)]

    # Add letter suffix for pronounceability
    last_char = word[-1]
    if last_char in 'aeiou':
        suffix_options = 'snrl'
    else:
        suffix_options = 'aeiouy'

    letter_hash = sum(ord(c) for c in session_id[1:]) if len(session_id) > 1 else hash_val
    suffix = suffix_options[letter_hash % len(suffix_options)]

    base_name = f"{word}{suffix}"

    # Add single collision word if attempt > 0 (deterministic per attempt number)
    if collision_attempt > 0:
        collision_hash = sum(ord(c) * (collision_attempt + 1) for c in session_id)
        collision_word = words[collision_hash % len(words)]
        base_name = f"{base_name}{collision_word}"

    # Add tag prefix if provided
    if tag:
        sanitized_tag = ''.join(c for c in tag if c not in '|\n\r\t')
        if not sanitized_tag:
            raise ValueError("Tag contains only invalid characters")
        if sanitized_tag != tag:
            print(f"Warning: Tag contained invalid characters, sanitized to '{sanitized_tag}'", file=sys.stderr)
        return f"{sanitized_tag}-{base_name}"

    return base_name

def resolve_instance_name(session_id: str, tag: str | None = None) -> tuple[str, dict | None]:
    """
    Resolve instance name for a session_id with race condition handling.
    Searches existing instances first (reuses if found), generates new name if not found.

    CRITICAL: Handles name generation race conditions:
    - Multiple threads may generate same name from get_display_name()
    - Retry with collision counter (preserves session_id identity)
    - DB UNIQUE constraint is authoritative source of truth

    Returns: (instance_name, existing_data_or_none)
    """
    from .db import find_instance_by_session, get_instance

    # Search for existing instance with this session_id (DB query, not glob)
    if session_id:
        existing_name = find_instance_by_session(session_id)
        if existing_name:
            data = get_instance(existing_name)
            return existing_name, data

    # Not found - generate new name with race condition retry
    max_retries = 100
    for attempt in range(max_retries):
        # Pass collision_attempt to get_display_name (preserves session_id identity)
        instance_name = get_display_name(session_id, tag, collision_attempt=attempt)

        # Check if name already taken in DB (might be taken between get_display_name and here)
        existing = get_instance(instance_name)
        if existing:
            # Name exists - check if it's ours or collision
            if existing.get('session_id') == session_id:
                return instance_name, existing  # Our instance
            # Real collision - try next attempt (increments collision_attempt)
            continue

        # Name appears free
        return instance_name, None

    raise RuntimeError(f"Cannot generate unique name for session after {max_retries} attempts")

def initialize_instance_in_position_file(instance_name: str, session_id: str | None = None, parent_session_id: str | None = None, enabled: bool | None = None, parent_name: str | None = None, mapid: str | None = None) -> bool:
    """Initialize instance in DB with required fields (idempotent). Returns True on success, False on failure."""
    from .db import get_instance, save_instance, get_last_event_id
    import sqlite3

    try:
        # Check if already exists - if so, update it with provided params (don't skip)
        existing = get_instance(instance_name)
        if existing:
            # Instance exists (possibly placeholder) - update with provided metadata
            updates = {}
            if parent_session_id is not None:
                updates['parent_session_id'] = parent_session_id
            if parent_name is not None:
                updates['parent_name'] = parent_name
            if enabled is not None:
                updates['enabled'] = int(enabled)
                updates['previously_enabled'] = int(enabled)
            if mapid is not None:
                updates['mapid'] = mapid
            if updates:
                from .db import update_instance
                update_instance(instance_name, updates)
            return True

        # Determine default enabled state: True for hcom-launched, False for vanilla
        is_hcom_launched = os.environ.get('HCOM_LAUNCHED') == '1'

        # Determine starting event ID: skip history or read from beginning
        initial_event_id = 0
        if SKIP_HISTORY:
            # Use launch event ID if available (for hcom-launched instances)
            # Otherwise use current max event ID (for vanilla instances)
            launch_event_id_str = os.environ.get('HCOM_LAUNCH_EVENT_ID')
            if launch_event_id_str:
                initial_event_id = int(launch_event_id_str)
            else:
                initial_event_id = get_last_event_id()

        # Determine enabled state: explicit param > hcom-launched > False
        if enabled is not None:
            default_enabled = enabled
        else:
            default_enabled = is_hcom_launched

        data = {
            "name": instance_name,
            "last_event_id": initial_event_id,
            "enabled": int(default_enabled),
            "previously_enabled": int(default_enabled),
            "directory": str(Path.cwd()),
            "last_stop": 0,
            "created_at": time.time(),
            "session_id": session_id if session_id else None,  # NULL not empty string
            "mapid": mapid or "",
            "transcript_path": "",
            "notification_message": "",
            "alias_announced": 0,
            "tag": None,
            "current_subagents": [],
            "subagent_mappings": {}
        }

        # Add parent_session_id and parent_name for subagents
        if parent_session_id:
            data["parent_session_id"] = parent_session_id
        if parent_name:
            data["parent_name"] = parent_name

        try:
            success = save_instance(instance_name, data)

            # Log creation event (only for HCOM participants)
            if success and default_enabled:
                try:
                    from .db import log_event

                    # Determine who launched this instance
                    launcher = os.environ.get('HCOM_LAUNCHED_BY', 'unknown')

                    log_event('life', instance_name, {
                        'action': 'created',
                        'by': launcher,
                        'enabled': default_enabled,
                        'is_hcom_launched': is_hcom_launched,
                        'is_subagent': bool(parent_session_id),
                        'parent_name': parent_name or ''
                    })
                except Exception:
                    pass  # Don't break creation if logging fails

            return success
        except sqlite3.IntegrityError:
            # UNIQUE constraint violation (race condition - another thread created same name)
            # This is expected and safe - just return success (instance exists)
            return True
    except Exception:
        return False

def enable_instance(instance_name: str, initiated_by: str = 'unknown', reason: str = '') -> None:
    """Enable instance
    Args:
        instance_name: Instance to enable
        initiated_by: Who initiated (from resolve_identity())
        reason: Context (e.g., 'manual', 'resume', 'launch')
    """
    update_instance_position(instance_name, {
        'enabled': True,
        'previously_enabled': True
    })
    # Log all enable operations
    try:
        from .db import log_event
        log_event('life', instance_name, {
            'action': 'started',
            'by': initiated_by,
            'reason': reason
        })
    except Exception:
        pass  # Don't break enable if logging fails

__all__ = [
    'load_instance_position',
    'save_instance_position',
    'update_instance_position',
    'is_parent_instance',
    'is_subagent_instance',
    'get_group_session_id',
    'in_same_group',
    'in_subagent_context',
    'get_instance_status',
    'get_status_description',
    'set_status',
    # Identity management
    'get_display_name',
    'resolve_instance_name',
    'initialize_instance_in_position_file',
    'enable_instance',
]
