#!/usr/bin/env python3
"""Terminal launching and agent management for HCOM"""
from __future__ import annotations

import os
import sys
import shlex
import re
import subprocess
import tempfile
import shutil
import platform
import random
from pathlib import Path
from typing import Any

from .shared import (
    IS_WINDOWS, CREATE_NO_WINDOW, is_wsl, is_termux,
    AGENT_NAME_PATTERN,
)
from .claude_args import extract_system_prompt_args, merge_system_prompts
from .core.paths import hcom_path, SCRIPTS_DIR
from .core.config import get_config
from .core.paths import read_file_with_retry

# ==================== Agent Management ====================

def list_available_agents() -> list[str]:
    """List available agent types from .claude/agents/"""
    agents = []
    for base_path in (Path.cwd(), Path.home()):
        agents_dir = base_path / '.claude' / 'agents'
        if agents_dir.exists():
            for agent_file in agents_dir.glob('*.md'):
                name = agent_file.stem
                if name not in agents and AGENT_NAME_PATTERN.fullmatch(name):
                    agents.append(name)
    return sorted(agents)

def extract_agent_config(content: str) -> dict[str, str]:
    """Extract configuration from agent YAML frontmatter"""
    if not content.startswith('---'):
        return {}

    # Find YAML section between --- markers
    if (yaml_end := content.find('\n---', 3)) < 0:
        return {}  # No closing marker

    yaml_section = content[3:yaml_end]
    config = {}

    # Extract model field
    if model_match := re.search(r'^model:\s*(.+)$', yaml_section, re.MULTILINE):
        value = model_match.group(1).strip()
        if value and value.lower() != 'inherit':
            config['model'] = value

    # Extract tools field
    if tools_match := re.search(r'^tools:\s*(.+)$', yaml_section, re.MULTILINE):
        value = tools_match.group(1).strip()
        if value:
            config['tools'] = value.replace(', ', ',')

    return config

def strip_frontmatter(content: str) -> str:
    """Strip YAML frontmatter from agent file"""
    if content.startswith('---'):
        # Find the closing --- on its own line
        lines = content.splitlines()
        for i, line in enumerate(lines[1:], 1):
            if line.strip() == '---':
                return '\n'.join(lines[i+1:]).strip()
    return content

def resolve_agent(name: str) -> tuple[str, dict[str, str]]:
    """Resolve agent file by name with validation.
    Looks for agent files in:
    1. .claude/agents/{name}.md (local)
    2. ~/.claude/agents/{name}.md (global)
    Returns tuple: (content without YAML frontmatter, config dict)
    """
    from .commands.utils import format_error

    hint = 'Agent names must use lowercase letters and dashes only'

    if not isinstance(name, str):
        raise FileNotFoundError(format_error(
            f"Agent '{name}' not found",
            hint
        ))

    candidate = name.strip()
    display_name = candidate or name

    if not candidate or not AGENT_NAME_PATTERN.fullmatch(candidate):
        raise FileNotFoundError(format_error(
            f"Agent '{display_name}' not found",
            hint
        ))

    for base_path in (Path.cwd(), Path.home()):
        agents_dir = base_path / '.claude' / 'agents'
        try:
            agents_dir_resolved = agents_dir.resolve(strict=True)
        except FileNotFoundError:
            continue

        agent_path = agents_dir / f'{candidate}.md'
        if not agent_path.exists():
            continue

        try:
            resolved_agent_path = agent_path.resolve(strict=True)
        except FileNotFoundError:
            continue

        try:
            resolved_agent_path.relative_to(agents_dir_resolved)
        except ValueError:
            continue

        content = read_file_with_retry(
            agent_path,
            lambda f: f.read(),
            default=None
        )
        if content is None:
            continue

        config = extract_agent_config(content)
        stripped = strip_frontmatter(content)
        if not stripped.strip():
            raise ValueError(format_error(
                f"Agent '{candidate}' has empty content",
                'Check the agent file is a valid format and contains text'
            ))
        return stripped, config

    available = list_available_agents()
    if available:
        hint = f"Available agents: {', '.join(available)}"
    else:
        hint = 'No agents found. Create one in .claude/agents/'

    raise FileNotFoundError(format_error(
        f"Agent '{candidate}' not found in project or user .claude/agents/ folder",
        hint
    ))

# ==================== Claude Command Building ====================

def has_claude_arg(claude_args: list[str] | None, arg_names: list[str], arg_prefixes: tuple[str, ...]) -> bool:
    """Check if argument already exists in claude_args"""
    return any(
        arg in arg_names or arg.startswith(arg_prefixes)
        for arg in (claude_args or [])
    )

def build_env_string(env_vars: dict[str, Any], format_type: str = "bash") -> str:
    """Build environment variable string for bash shells"""
    if format_type == "bash_export":
        # Properly escape values for bash
        return ' '.join(f'export {k}={shlex.quote(str(v))};' for k, v in env_vars.items())
    else:
        return ' '.join(f'{k}={shlex.quote(str(v))}' for k, v in env_vars.items())

def build_claude_command(agent_content: str | None = None, claude_args: list[str] | None = None, model: str | None = None, tools: str | None = None) -> tuple[str, str | None]:
    """Build Claude command with proper argument handling
    Returns tuple: (command_string, temp_file_path_or_none)
    For agent content, writes to temp file and uses cat to read it.
    Merges user's --system-prompt/--append-system-prompt with agent content.
    Prompt comes from claude_args (positional tokens from HCOM_CLAUDE_ARGS).
    """
    cmd_parts = ['claude']
    temp_file_path = None

    # Extract user's system prompt flags
    cleaned_args, user_append, user_system = extract_system_prompt_args(claude_args or [])

    # Detect print mode
    is_print_mode = bool(cleaned_args and any(arg in cleaned_args for arg in ['-p', '--print']))

    # Add model if specified and not already in cleaned_args
    if model:
        if not has_claude_arg(cleaned_args, ['--model'], ('--model=',)):
            cmd_parts.extend(['--model', model])

    # Add allowed tools if specified and not already in cleaned_args
    if tools:
        if not has_claude_arg(cleaned_args, ['--allowedTools', '--allowed-tools'],
                              ('--allowedTools=', '--allowed-tools=')):
            cmd_parts.extend(['--allowedTools', tools])

    # Add cleaned user args (system prompt flags removed, but positionals/prompt included)
    if cleaned_args:
        for arg in cleaned_args:
            cmd_parts.append(shlex.quote(arg))

    # Merge and apply system prompts
    merged_content, flag = merge_system_prompts(user_append, user_system, agent_content, prefer_system_flag=is_print_mode)

    if merged_content:
        # Write merged content to temp file
        scripts_dir = hcom_path(SCRIPTS_DIR)
        temp_file = tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', suffix='.txt', delete=False,
                                              prefix='hcom_agent_', dir=str(scripts_dir))
        temp_file.write(merged_content)
        temp_file.close()
        temp_file_path = temp_file.name

        cmd_parts.append(flag)
        cmd_parts.append(f'"$(cat {shlex.quote(temp_file_path)})"')

    return ' '.join(cmd_parts), temp_file_path

# ==================== Script Creation ====================

def find_bash_on_windows() -> str | None:
    """Find Git Bash on Windows, avoiding WSL's bash launcher"""
    # Build prioritized list of bash candidates
    candidates = []
    # 1. Common Git Bash locations (highest priority)
    for base in [os.environ.get('PROGRAMFILES', r'C:\Program Files'),
                 os.environ.get('PROGRAMFILES(X86)', r'C:\Program Files (x86)')]:
        if base:
            candidates.extend([
                str(Path(base) / 'Git' / 'usr' / 'bin' / 'bash.exe'),  # usr/bin is more common
                str(Path(base) / 'Git' / 'bin' / 'bash.exe')
            ])
    # 2. Portable Git installation
    if local_appdata := os.environ.get('LOCALAPPDATA', ''):
        git_portable = Path(local_appdata) / 'Programs' / 'Git'
        candidates.extend([
            str(git_portable / 'usr' / 'bin' / 'bash.exe'),
            str(git_portable / 'bin' / 'bash.exe')
        ])
    # 3. PATH bash (if not WSL's launcher)
    if (path_bash := shutil.which('bash')) and not path_bash.lower().endswith(r'system32\bash.exe'):
        candidates.append(path_bash)
    # 4. Hardcoded fallbacks (last resort)
    candidates.extend([
        r'C:\Program Files\Git\usr\bin\bash.exe',
        r'C:\Program Files\Git\bin\bash.exe',
        r'C:\Program Files (x86)\Git\usr\bin\bash.exe',
        r'C:\Program Files (x86)\Git\bin\bash.exe'
    ])
    # Find first existing bash
    for bash in candidates:
        if bash and Path(bash).exists():
            return bash

    return None

def create_bash_script(script_file: str, env: dict[str, Any], cwd: str | None, command_str: str, background: bool = False) -> None:
    """Create a bash script for terminal launch
    Scripts provide uniform execution across all platforms/terminals.
    Cleanup behavior:
    - Normal scripts: append 'rm -f' command for self-deletion
    - Background scripts: persist until `hcom reset logs` cleanup (24 hours)
    - Agent scripts: treated like background (contain 'hcom_agent_')
    """
    try:
        script_path = Path(script_file)
    except (OSError, IOError) as e:
        raise Exception(f"Cannot create script directory: {e}")

    with open(script_file, 'w', encoding='utf-8') as f:
        f.write('#!/bin/bash\n')
        f.write('echo "Starting Claude Code..."\n')

        if platform.system() != 'Windows':
            # 1. Discover paths once
            claude_path = shutil.which('claude')
            node_path = shutil.which('node')

            # 2. Add to PATH for minimal environments
            paths_to_add = []
            for p in [node_path, claude_path]:
                if p:
                    dir_path = str(Path(p).resolve().parent)
                    if dir_path not in paths_to_add:
                        paths_to_add.append(dir_path)

            if paths_to_add:
                path_addition = ':'.join(paths_to_add)
                f.write(f'export PATH="{path_addition}:$PATH"\n')
            elif not claude_path:
                # Warning for debugging
                print("Warning: Could not locate 'claude' in PATH", file=sys.stderr)

            # 3. Write environment variables
            f.write(build_env_string(env, "bash_export") + '\n')

            if cwd:
                f.write(f'cd {shlex.quote(cwd)}\n')

            # 4. Platform-specific command modifications
            if claude_path:
                if is_termux():
                    # Termux: explicit node to bypass shebang issues
                    final_node = node_path or '/data/data/com.termux/files/usr/bin/node'
                    # Quote paths for safety
                    command_str = command_str.replace(
                        'claude ',
                        f'{shlex.quote(final_node)} {shlex.quote(claude_path)} ',
                        1
                    )
                else:
                    # Mac/Linux: use full path (PATH now has node if needed)
                    command_str = command_str.replace('claude ', f'{shlex.quote(claude_path)} ', 1)
        else:
            # Windows: no PATH modification needed
            f.write(build_env_string(env, "bash_export") + '\n')
            if cwd:
                f.write(f'cd {shlex.quote(cwd)}\n')

        f.write(f'{command_str}\n')

        # Self-delete for normal mode (not background or agent)
        if not background and 'hcom_agent_' not in command_str:
            f.write(f'rm -f {shlex.quote(script_file)}\n')

    # Make executable on Unix
    if platform.system() != 'Windows':
        os.chmod(script_file, 0o755)

# ==================== Terminal Launching ====================

def get_macos_terminal_argv() -> list[str]:
    """Return macOS Terminal.app launch command as argv list."""
    return ['osascript', '-e', 'tell app "Terminal" to do script "bash {script}"', '-e', 'tell app "Terminal" to activate']

def get_windows_terminal_argv() -> list[str]:
    """Return Windows terminal launcher as argv list."""
    from .commands.utils import format_error

    if not (bash_exe := find_bash_on_windows()):
        raise Exception(format_error("Git Bash not found"))

    if shutil.which('wt'):
        return ['wt', bash_exe, '{script}']
    return ['cmd', '/c', 'start', 'Claude Code', bash_exe, '{script}']

def get_linux_terminal_argv() -> list[str] | None:
    """Return first available Linux terminal as argv list."""
    terminals = [
        ('gnome-terminal', ['gnome-terminal', '--', 'bash', '{script}']),
        ('konsole', ['konsole', '-e', 'bash', '{script}']),
        ('xterm', ['xterm', '-e', 'bash', '{script}']),
    ]
    for term_name, argv_template in terminals:
        if shutil.which(term_name):
            return argv_template

    # WSL fallback integrated here
    if is_wsl() and shutil.which('cmd.exe'):
        if shutil.which('wt.exe'):
            return ['cmd.exe', '/c', 'start', 'wt.exe', 'bash', '{script}']
        return ['cmd.exe', '/c', 'start', 'bash', '{script}']

    return None

def windows_hidden_popen(argv: list[str], *, env: dict[str, str] | None = None, cwd: str | None = None, stdout: Any = None) -> subprocess.Popen:
    """Create hidden Windows process without console window."""
    if IS_WINDOWS:
        startupinfo = subprocess.STARTUPINFO()  # type: ignore[attr-defined]
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW  # type: ignore[attr-defined]
        startupinfo.wShowWindow = subprocess.SW_HIDE  # type: ignore[attr-defined]

        return subprocess.Popen(
            argv,
            env=env,
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=subprocess.STDOUT,
            startupinfo=startupinfo,
            creationflags=CREATE_NO_WINDOW
        )
    else:
        raise RuntimeError("windows_hidden_popen called on non-Windows platform")

# Platform dispatch map
PLATFORM_TERMINAL_GETTERS = {
    'Darwin': get_macos_terminal_argv,
    'Windows': get_windows_terminal_argv,
    'Linux': get_linux_terminal_argv,
}

def _parse_terminal_command(template: str, script_file: str) -> list[str]:
    """Parse terminal command template safely to prevent shell injection.
    Parses the template FIRST, then replaces {script} placeholder in the
    parsed tokens. This avoids shell injection and handles paths with spaces.
    Args:
        template: Terminal command template with {script} placeholder
        script_file: Path to script file to substitute
    Returns:
        list: Parsed command as argv array
    Raises:
        ValueError: If template is invalid or missing {script} placeholder
    """
    from .commands.utils import format_error

    if '{script}' not in template:
        raise ValueError(format_error("Custom terminal command must include {script} placeholder",
                                    'Example: open -n -a kitty.app --args bash "{script}"'))

    try:
        parts = shlex.split(template)
    except ValueError as e:
        raise ValueError(format_error(f"Invalid terminal command syntax: {e}",
                                    "Check for unmatched quotes or invalid shell syntax"))

    # Replace {script} in parsed tokens
    replaced = []
    placeholder_found = False
    for part in parts:
        if '{script}' in part:
            replaced.append(part.replace('{script}', script_file))
            placeholder_found = True
        else:
            replaced.append(part)

    if not placeholder_found:
        raise ValueError(format_error("{script} placeholder not found after parsing",
                                    "Ensure {script} is not inside environment variables"))

    return replaced

def launch_terminal(command: str, env: dict[str, str], cwd: str | None = None, background: bool = False) -> str | bool | None:
    """Launch terminal with command using unified script-first approach

    Environment precedence: config.env < shell environment
    Internal hcom vars (HCOM_LAUNCHED, etc) don't conflict with user vars.

    Args:
        command: Command string from build_claude_command
        env: Contains config.env defaults + hcom internal vars
        cwd: Working directory
        background: Launch as background process
    """
    from .commands.utils import format_error
    from .core.paths import LOGS_DIR
    import time

    # config.env defaults + internal vars, then shell env overrides
    env_vars = env.copy()

    # Ensure SHELL is in env dict BEFORE os.environ update
    # (Critical for Termux Activity Manager which launches scripts in clean environment)
    if 'SHELL' not in env_vars:
        shell_path = os.environ.get('SHELL')
        if not shell_path:
            shell_path = shutil.which('bash') or shutil.which('sh')
        if not shell_path:
            # Platform-specific fallback
            if is_termux():
                shell_path = '/data/data/com.termux/files/usr/bin/bash'
            else:
                shell_path = '/bin/bash'
        if shell_path:
            env_vars['SHELL'] = shell_path

    env_vars.update(os.environ)
    command_str = command

    # 1) Always create a script
    script_file = str(hcom_path(SCRIPTS_DIR,
        f'hcom_{os.getpid()}_{random.randint(1000,9999)}.sh'))
    create_bash_script(script_file, env_vars, cwd, command_str, background)

    # 2) Background mode
    if background:
        logs_dir = hcom_path(LOGS_DIR)
        log_file = logs_dir / env['HCOM_BACKGROUND']

        try:
            with open(log_file, 'w', encoding='utf-8') as log_handle:
                if IS_WINDOWS:
                    # Windows: hidden bash execution with Python-piped logs
                    bash_exe = find_bash_on_windows()
                    if not bash_exe:
                        raise Exception("Git Bash not found")

                    process = windows_hidden_popen(
                        [bash_exe, script_file],
                        env=env_vars,
                        cwd=cwd,
                        stdout=log_handle
                    )
                else:
                    # Unix(Mac/Linux/Termux): detached bash execution with Python-piped logs
                    process = subprocess.Popen(
                        ['bash', script_file],
                        env=env_vars, cwd=cwd,
                        stdin=subprocess.DEVNULL,
                        stdout=log_handle, stderr=subprocess.STDOUT,
                        start_new_session=True
                    )

        except OSError as e:
            print(format_error(f"Failed to launch headless instance: {e}"), file=sys.stderr)
            return None

        # Health check
        time.sleep(0.2)
        if process.poll() is not None:
            error_output = read_file_with_retry(log_file, lambda f: f.read()[:1000], default="")
            print(format_error("Headless instance failed immediately"), file=sys.stderr)
            if error_output:
                print(f"  Output: {error_output}", file=sys.stderr)
            return None

        return str(log_file)

    # 3) Terminal modes
    terminal_mode = get_config().terminal

    if terminal_mode == 'print':
        # Print script path and contents
        try:
            with open(script_file, 'r', encoding='utf-8') as f:
                script_content = f.read()
            print(f"# Script: {script_file}")
            print(script_content)
            Path(script_file).unlink()  # Clean up immediately
            return True
        except Exception as e:
            print(format_error(f"Failed to read script: {e}"), file=sys.stderr)
            return False

    if terminal_mode == 'here':
        print("Launching Claude in current terminal...")
        if IS_WINDOWS:
            bash_exe = find_bash_on_windows()
            if not bash_exe:
                print(format_error("Git Bash not found"), file=sys.stderr)
                return False
            result = subprocess.run([bash_exe, script_file], env=env_vars, cwd=cwd)
        else:
            result = subprocess.run(['bash', script_file], env=env_vars, cwd=cwd)
        return result.returncode == 0

    # 4) New window or custom command mode
    # If terminal is not 'here' or 'print', it's either 'new' (platform default) or a custom command
    custom_cmd = None if terminal_mode == 'new' else terminal_mode

    if not custom_cmd:  # Platform default 'new' mode
        if is_termux():
            # Keep Termux as special case
            am_cmd = [
                'am', 'startservice', '--user', '0',
                '-n', 'com.termux/com.termux.app.RunCommandService',
                '-a', 'com.termux.RUN_COMMAND',
                '--es', 'com.termux.RUN_COMMAND_PATH', script_file,
                '--ez', 'com.termux.RUN_COMMAND_BACKGROUND', 'false'
            ]
            try:
                subprocess.run(am_cmd, check=False)
                return True
            except Exception as e:
                print(format_error(f"Failed to launch Termux: {e}"), file=sys.stderr)
                return False

        # Unified platform handling via helpers
        system = platform.system()
        if not (terminal_getter := PLATFORM_TERMINAL_GETTERS.get(system)):
            raise Exception(format_error(f"Unsupported platform: {system}"))

        custom_cmd = terminal_getter()
        if not custom_cmd:  # e.g., Linux with no terminals
            raise Exception(format_error("No supported terminal emulator found",
                                       "Install gnome-terminal, konsole, or xterm"))

    # Type-based dispatch for execution
    if isinstance(custom_cmd, list):
        # Our argv commands - safe execution without shell
        final_argv = [arg.replace('{script}', script_file) for arg in custom_cmd]
        try:
            if platform.system() == 'Windows':
                # Windows needs non-blocking for parallel launches
                subprocess.Popen(final_argv)
                return True  # Popen is non-blocking, can't check success
            else:
                result = subprocess.run(final_argv)
                if result.returncode != 0:
                    return False
                return True
        except Exception as e:
            print(format_error(f"Failed to launch terminal: {e}"), file=sys.stderr)
            return False
    else:
        # User-provided string commands - parse safely without shell=True
        try:
            final_argv = _parse_terminal_command(custom_cmd, script_file)
        except ValueError as e:
            print(str(e), file=sys.stderr)
            return False

        try:
            if platform.system() == 'Windows':
                # Windows needs non-blocking for parallel launches
                subprocess.Popen(final_argv)
                return True  # Popen is non-blocking, can't check success
            else:
                result = subprocess.run(final_argv)
                if result.returncode != 0:
                    return False
                return True
        except Exception as e:
            print(format_error(f"Failed to execute terminal command: {e}"), file=sys.stderr)
            return False

# ==================== Exports ====================

__all__ = [
    # Agent management
    'list_available_agents',
    'extract_agent_config',
    'strip_frontmatter',
    'resolve_agent',
    # Claude command building
    'has_claude_arg',
    'build_env_string',
    'build_claude_command',
    # Script creation
    'find_bash_on_windows',
    'create_bash_script',
    # Terminal launching
    'get_macos_terminal_argv',
    'get_windows_terminal_argv',
    'get_linux_terminal_argv',
    'windows_hidden_popen',
    'PLATFORM_TERMINAL_GETTERS',
    'launch_terminal',
]
