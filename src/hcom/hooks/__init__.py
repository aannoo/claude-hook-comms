"""Hook system for HCOM"""
from .dispatcher import handle_hook
from .handlers import (
    handle_pretooluse,
    handle_posttooluse,
    handle_stop,
    handle_subagent_stop,
    handle_userpromptsubmit,
    handle_sessionstart,
    handle_sessionend,
    handle_notify,
)
from .settings import (
    HOOK_CONFIGS,
    ACTIVE_HOOK_TYPES,
    HOOK_COMMANDS,
    HCOM_HOOK_PATTERNS,
    get_claude_settings_path,
    load_settings_json,
    _remove_hcom_hooks_from_settings,
)

__all__ = [
    'handle_hook',
    'handle_pretooluse',
    'handle_posttooluse',
    'handle_stop',
    'handle_subagent_stop',
    'handle_userpromptsubmit',
    'handle_sessionstart',
    'handle_sessionend',
    'handle_notify',
    'HOOK_CONFIGS',
    'ACTIVE_HOOK_TYPES',
    'HOOK_COMMANDS',
    'HCOM_HOOK_PATTERNS',
    'get_claude_settings_path',
    'load_settings_json',
    '_remove_hcom_hooks_from_settings',
]
