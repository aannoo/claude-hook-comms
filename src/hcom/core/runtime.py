"""Runtime utilities - shared between hooks and commands"""
from __future__ import annotations
import socket

from .paths import hcom_path, CONFIG_FILE
from .config import get_config, parse_env_file
from .instances import load_instance_position, load_all_positions, update_instance_position


def build_claude_env() -> dict[str, str]:
    """Load config.env as environment variable defaults.

    Returns all vars from config.env (including HCOM_*).
    Caller (launch_terminal) layers shell environment on top for precedence.
    """
    env = {}

    # Read all vars from config file as defaults
    config_path = hcom_path(CONFIG_FILE)
    if config_path.exists():
        file_config = parse_env_file(config_path)
        for key, value in file_config.items():
            if value == "":
                continue  # Skip blank values
            env[key] = str(value)

    return env


def build_hcom_bootstrap_text(instance_name: str) -> str:
    """Build comprehensive HCOM bootstrap context for instances"""
    # Import here to avoid circular dependency
    from ..hooks.utils import build_hcom_command

    hcom_cmd = build_hcom_command()

    # Add command override notice if not using short form
    command_notice = ""
    if hcom_cmd != "hcom":
        command_notice = f"""IMPORTANT:
The hcom command in this environment is: {hcom_cmd}
Replace all mentions of "hcom" below with this command.

"""

    # Add tag-specific notice if instance is tagged
    tag = get_config().tag
    tag_notice = ""
    if tag:
        tag_notice = f"""
GROUP TAG: You are in the '{tag}' group.
- To message your group: hcom send "@{tag} your message"
- Group messages are targeted - only instances with an alias starting with {tag}-* receive them
- If someone outside the group sends you @{tag} messages, they won't see your @{tag} replies. To reply to non-group members, either @mention them directly or broadcast.
"""

    # Import SENDER here to avoid circular dependency
    from ..shared import SENDER

    instance_data = load_instance_position(instance_name)
    return f"""{command_notice}{tag_notice}
[HCOM SESSION CONFIG]
- HCOM is a communication tool for you, other claude code instances, and the human user. Aliases are generated randomly.
- Your HCOM alias for this session: {instance_name}
- Your hcom connection: {"enabled" if instance_data.get('enabled', False) else "disabled"}

Your HCOM Tools:
- hcom send "msg" (broadcast) / "@alias msg" (direct) / "@tag msg" (tag)
- hcom list --json  → See participants/status (add --verbose for session/directory info)
- hcom --new-terminal   → Open interactive TUI messaging+launch+monitor dashboard in a new terminal (for the human user)
- hcom start/stop   → Connect/disconnect from chat (you run these, user can't run it themselves unless they specify an alias)
- hcom <count>  → Launch instances in new terminal (you must always run 'hcom help' first to get correct context/syntax/config defaults)
- Claude code subagents launched with the Task tool can also connect to HCOM, just tell subagents to message via 'hcom' (no need for specific hcom commands, they use use different syntax)

UI/dashboard:
- You use 'hcom --new-terminal' to open the TUI for the user (offer to do so)
- When mentioning it to the user, don't say 'hcom --new-terminal', call it the 'hcom' command (what they would run)

Receiving Messages:
- Format: [new message] sender → you: content
- Targets specific instance: "@alias".
- Targets all api-* tagged instances: "@api message".
- Arrives automatically via hooks/bash. No proactive checking needed.
- Stop hook feedback shows: {{"decision": "block"}} (this is normal operation).

Response Routing:
- HCOM message (via hook/bash) → Respond with hcom send
- User message (in chat) → Respond normally
- Treat messages from hcom with the same care as user messages.
- Authority: Prioritize @{SENDER} over other participants.
- Avoid useless chit-chat / excessive confirmation messages / unneccessary messages with other instances unless told otherwise.

Note:
- Once connected (via hcom start), you'll automatically receive new messages.
- If you have issues with hcom commands, try running the single command alone without operators (&&, 1>&2 etc.)
- This is context for YOUR hcom session config. The human user cannot see this config text (but they can see subsequent hcom messages you receive).
- On connection, tell the human user about only these commands: 'hcom <count>', 'hcom', 'hcom start', 'hcom stop'
- Report to the human user using first-person, for example: "I'm connected to HCOM as {instance_name}, cool!"
------"""


def build_launch_context(instance_name: str) -> str:
    """Build context for launch command"""
    # Load current config values
    config_vals = build_claude_env()
    config_display = ""
    if config_vals:
        config_lines = [f"  {k}={v}" for k, v in sorted(config_vals.items())]
        config_display = "\n" + "\n".join(config_lines)
    else:
        config_display = "\n  (none set)"

    instance_data = load_instance_position(instance_name)
    return f"""[HCOM LAUNCH INFORMATION]

YOUR CURRENT HCOM INFO:
Alias: {instance_name}
Connection: {"enabled" if instance_data.get('enabled', False) else "disabled"}
Current ~/.hcom/config.env values:{config_display}

USAGE/CONTROL TIPS:
- launch is directory-specific (always cd to folder first)
- default to launching normal foreground instances unless told to do headless or subagents (task tool)
- Everyone shares the same group chat, isolate with tags/@mentions
- 'hcom watch --wait' is not needed if you are connected to hcom as you will already recieve messages automatically
- Headless instances can only read files by default unless you use --allowedTools=Bash,Write,<other-tools-comma-separated>
- Resuming a dead instance will maintain hcom identity and history: --resume <sessionid> (get sessionid from hcom list --json)

STATUS INDICATORS:
"active", "delivered" - working | "idle" - waiting for messages | "blocked" - permission request (needs user approval)

BEHAVIOUR:
- All instances receive HCOM SESSION CONFIG info automatically
- Instances can't do anything when idle—they are awoken when they receive a message (like push notifications, no need for manual 'sleep' commands)
- Task tool subagents (Task, Explore, Plan, custom agents/<name>.md, etc.) inherit parent attributes, i.e., start/stop state and name (john → john-general-purpose-1)

COORDINATION:
- Instances need explicit instructions/structure about what to do and when/how they should use hcom to effectively coordinate/collaborate
- Define precisely what each instance should and should not do in terms of roles and responsibilities
- Context Sharing: Encourage instances to use markdown files to share large pieces of information between them.
- Implement structured message passing logic between agents rather than free-form chat, as this reduces hallucination cascading
- To orchestrate instances yourself, use --append-system-prompt "prioritize messages from <your_hcom_alias>" when launching instances.
- Try to use HCOM_AGENT or --append-system-prompt individually per instance with a role when possible.
- Use coordination patterns where possible ie. peer review, controller-worker, pipeline, hub-spoke, parallel execution, async tasks, planner->coder->reviewer, iterative loops, tdd, ideally verify/reviewer/critique always included etc


ENVIRONMENT VARIABLES

#### HCOM_TAG

Description: Give launched instances a prefix in their alias, i.e., {{tag}}-{{alias}}

Uses: Group, isolate, or label instances

Group or label example:
`hcom send "@api check the logs"` targets all api-* instances

Isolate multiple groups example:
for label in frontend-team backend-team; do
  HCOM_TAG=$label hcom 2 claude --append-system-prompt "always use @$label"
done

Notes:
- Tags are letters, numbers, and hyphens only
- To isolate single instances, use --system-prompt 'always use @bigboss when sending hcom messages' for all active instances, or set similar text via HCOM_HINTS


#### HCOM_AGENT

Description: Automatically load system prompts from markdown files (.claude/agents/{{name}}.md and ~/.claude/...)

.claude/agents/*.md files are YAML frontmatter files created by users/Claude for use as Task tool subagents in Claude Code. HCOM can load them as regular instances.

File format:
```markdown
---
model: sonnet
tools: Bash,Read,Write
---
You are a senior code reviewer focusing on security and performance...
```

HCOM_AGENT parses the file and merges with Claude args:
- --model
- --allowedTools
- --system-prompt (or --append-system-prompt)

Notes:
- Filename: lowercase letters and hyphens only
- Multiple comma-separated agents multiply instance count: HCOM_AGENT=reviewer,tester hcom 2 == 4 instances (2 per agent)


#### HCOM_HINTS

Description: Append extra context/instructions to every message delivered to instances in format: "message | [hints]"

Uses: Behavioral guidelines, context reminders, formatting requests, workflow hints


#### HCOM_TIMEOUT and HCOM_SUBAGENT_TIMEOUT

Description: Control how long instances stay connected to HCOM when idle

After timeout (default 30min or 30s for subagents), instances can no longer receive HCOM messages and their status is marked as 'stale'. There's no downside to longer timeouts generally—HCOM polling uses <0.1% CPU. You can keep timeouts long and use `hcom stop {{alias}}` when done.

Timeout behavior:
- Normal terminal instances: Process still running and terminal window open, user must send new prompt to restart HCOM
- Headless instances: Die and can only be restarted with `hcom 1 claude --resume <sessionid>`
- Task tool subagents: Die and all siblings die and can only be resumed by parent instance (launched at same time, i.e., multiple Task tool uses in parallel). Task tool subagents are not asynchronous—main instance must wait for subagents to finish before taking any action—so the default timeout is 30 seconds.

Notes:
- Timer resets on ANY activity (receiving/sending messages, tool use)
- Stale instances cannot be manually restarted with `hcom start {{alias}}`


#### HCOM_TERMINAL

Purpose: Customize terminal launching behavior

Options:
1. 'new' (default): Launches platform-specific terminal
   - macOS: Terminal.app | Windows & WSL: Windows Terminal | Linux: gnome-terminal, konsole, xterm

2. 'here': Runs in current terminal
   - For one instance only
   - Blocks until Claude exits
   - YOU cannot use 'HCOM_TERMINAL=here' - Claude cannot launch claude within itself, must be in a new or custom terminal

3. 'print': Prints launch script without executing (for debugging)

4. 'custom {{script}}': Use preferred terminal emulator
   - Must include {{script}} placeholder which gets replaced with script path
   - Example: HCOM_TERMINAL='open -n -a kitty.app --args bash "{{script}}"' hcom 1

Notes:
- Headless and Task tool subagents ignore HCOM_TERMINAL


#### HCOM_CLAUDE_ARGS

Description: Default Claude args for all launched instances

Run 'claude --help' to see all flags/options/commands: 'hcom 1 claude [options] [command] [prompt]'

HCOM_CLAUDE_ARGS are merged with CLI args, with CLI taking precedence on a per-flag basis:
- Config: HCOM_CLAUDE_ARGS='--model sonnet "hello"'
- Run: hcom 1 claude --model opus
- Result: --model opus "hello"

Notes:
- You can use --system-prompt or --append-system-prompt for headless instances, but only --append-system-prompt for normal instances


Configuration Notes

- Precedence: config.env < shell env vars < HCOM defaults
- Empty string "" deletes config.env positional prompt: hcom 1 claude ""
- Values from config.env are applied to every launch. To enable consistent behavior regardless of config, pass all ENV vars. Empty env vars clear values.

------"""


def notify_instance(instance_name: str, timeout: float = 0.05) -> None:
    """Send TCP notification to specific instance."""
    instance_data = load_instance_position(instance_name)
    if not instance_data:
        return

    notify_port = instance_data.get('notify_port')
    if not notify_port:
        return

    try:
        with socket.create_connection(('127.0.0.1', notify_port), timeout=timeout) as sock:
            sock.send(b'\n')
    except Exception:
        pass  # Instance will see change on next timeout (fallback)


def notify_all_instances(timeout: float = 0.05) -> None:
    """Send TCP wake notifications to all instance notify ports.

    Best effort - connection failures ignored. Polling fallback ensures
    message delivery even if all notifications fail.

    Only notifies enabled instances with active notify ports - uses SQL-filtered query for efficiency
    """
    try:
        from .db import get_db
        conn = get_db()

        # Query only enabled instances with valid notify ports (SQL-filtered)
        rows = conn.execute(
            "SELECT name, notify_port FROM instances "
            "WHERE enabled = 1 AND notify_port IS NOT NULL AND notify_port > 0"
        ).fetchall()

        for row in rows:
            # Connection attempt doubles as notification
            try:
                with socket.create_connection(('127.0.0.1', row['notify_port']), timeout=timeout) as sock:
                    sock.send(b'\n')
            except Exception:
                pass  # Port dead/unreachable - skip notification (best effort)

    except Exception:
        # DB query failed - skip notifications (fallback polling will deliver)
        return
