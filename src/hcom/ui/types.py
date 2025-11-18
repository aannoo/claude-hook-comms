"""UI type definitions"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Literal, Any


@dataclass
class Field:
    """Field representation for rendering expandable sections"""
    key: str
    display_name: str
    field_type: Literal['checkbox', 'text', 'cycle', 'numeric']
    value: str | bool
    options: List[str] | None = None
    hint: str = ""


class Mode(Enum):
    MANAGE = "manage"
    LAUNCH = "launch"


class LaunchField(Enum):
    COUNT = 0
    LAUNCH_BTN = 1
    CLAUDE_SECTION = 2
    HCOM_SECTION = 3
    CUSTOM_ENV_SECTION = 4
    OPEN_EDITOR = 5


@dataclass
class UIState:
    """Shared state accessed by screen classes and HcomTUI orchestrator"""

    # Manage screen state
    cursor: int = 0
    cursor_instance_name: Optional[str] = None
    instances: dict = field(default_factory=dict)
    status_counts: dict = field(default_factory=dict)
    messages: list = field(default_factory=list)
    message_buffer: str = ""
    message_cursor_pos: int = 0
    instance_scroll_pos: int = 0

    # Launch screen state
    launch_count: str = "1"
    launch_prompt: str = ""
    launch_system_prompt: str = ""
    launch_append_system_prompt: str = ""
    launch_background: bool = False
    launch_field: LaunchField = LaunchField.COUNT
    launch_prompt_cursor: int = 0
    launch_system_prompt_cursor: int = 0
    launch_append_system_prompt_cursor: int = 0
    config_field_cursors: dict = field(default_factory=dict)

    # Config state
    config_edit: dict = field(default_factory=dict)
    config_mtime: float = 0.0

    # Section expansion
    claude_expanded: bool = False
    hcom_expanded: bool = False
    custom_env_expanded: bool = False
    claude_cursor: int = -1
    hcom_cursor: int = -1
    custom_env_cursor: int = -1

    # Confirmation state
    pending_toggle: Optional[str] = None
    pending_toggle_time: float = 0.0
    completed_toggle: Optional[str] = None
    completed_toggle_time: float = 0.0
    pending_stop_all: bool = False
    pending_stop_all_time: float = 0.0
    pending_reset: bool = False
    pending_reset_time: float = 0.0

    # Launch scrolling
    launch_scroll_pos: int = 0

    # Flash notifications
    flash_message: Optional[str] = None
    flash_until: float = 0.0
    flash_color: str = 'orange'

    # Validation
    validation_errors: dict = field(default_factory=dict)

    # Rendering cache
    last_event_id: int = 0
    last_message_time: float = 0.0

    # LOG filtering
    log_filter: str = ""                   # Current filter query (empty = inactive)
    log_filter_cursor: int = 0             # Cursor position in filter input
    log_event_type: str = "all"            # Event type filter: "all", "message", "status", "life"
