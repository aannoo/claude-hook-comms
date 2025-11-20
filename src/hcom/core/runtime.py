"""Runtime utilities - shared between hooks and commands"""
from __future__ import annotations
import socket

from .paths import hcom_path, CONFIG_FILE
from .config import get_config, parse_env_file
from .instances import load_instance_position, update_instance_position


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
- hcom send "msg" (broadcast to all) / "@alias msg" (direct) / "@tag msg" (tag)
- hcom list [--json]  → See all participants, read receipts, state/current info
- hcom start/stop   → Connect/disconnect from chat (you run these, user can't run it themselves unless they specify an alias)
- hcom <count>  → Launch instances in new terminal (you MUST always run 'hcom help' first to get correct context/syntax/config defaults)
- Claude code subagents launched with the Task tool can also connect to HCOM, tell subagents to run hcom start.

UI/dashboard:
- Use 'hcom --new-terminal' to open TUI (message+launch+monitor+manage) for user in new terminal (you can't display TUIs - no TTY)
- When mentioning to user (offer to do it): "I can open the hcom dashboard" (don't say --new-terminal)

Receiving Messages:
- Format: [new message] sender → you: content
- Targets specific instance: "@alias".
- Targets all api-* tagged instances: "@api message".
- Arrives automatically like push notifications via hooks/bash. No proactive checking needed.

Status Indicators:
- ▶ active (working on task)
- ◉ idle (ready for work)
- ■ blocked (needs user approval)
- ○ inactive (session ended, timeout, stale)

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
- Ensure you always run hcom help before launching instances for the first time.
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

## YOUR CURRENT HCOM INFO:
Alias: {instance_name}
Connection: {"enabled" if instance_data.get('enabled', False) else "disabled"}
Current ~/.hcom/config.env values have been set to:{config_display}

## USAGE/CONTROL:
- launch is directory-specific (always cd first)
- default to normal foreground instances unless told to use headless/subagents
- Everyone shares group chat, isolate with tags/@mentions
- Headless instances can only read files and respond in hcom by default, for more, use --tools Bash,Write
- Resume dead instances to maintain hcom identity/history: --resume <session_id> (get id from hcom list --verbose)
- Instances require an initial prompt to auto-connect to hcom otherwise needs human user intervention
- Do not use sleep commands, instead use hcom watch --wait with the specific condition you are waiting for


## EVENT/INSTANCE QUERY
watch - historical, from events table (NDJSON stream)
types:
- message: from,scope,recipients...
- status: active=working, idle=ready, blocked=needs approval, inactive=dead
- life: action: created|started|stopped|launched|ready

list - current, from instances table (NDJSON snapshot)
- read_receipts, hcom_connected, status, wait_timeout...

Usage:
hcom list --json | jq '.[] | select(.status_age_seconds < 300)' # Filter stale instances
hcom watch --sql "type = 'life'" | jq ....
sqlite3 ~/.hcom/hcom.db "SELECT * FROM..."  # Direct SQL (2 tables: instances & events)

Always use watch --wait with --sql instead of sleep:
Wrong: sleep 10 && hcom list && sleep 10 && hcom list && sleep 10 && hcom list
Right: hcom watch --wait 60 --sql "type = 'status' AND json_extract(data, '$.status') = 'idle'" # blocks until specific event or timeout (exit 0=match, 1=timeout, 2=error, 3=interrupted by @mention)


## BEHAVIOUR
- All instances receive HCOM SESSION CONFIG info automatically
- Idle instances can't do anything except wake on message delivery
- Task tool subagents inherit their parents hcom state/name (john → john-general-purpose-1)


## COORDINATION
- Define explicit roles/responsibilities/instructions (via system prompt, initial prompt, HCOM_AGENT, HCOM_HINTS) - how each instance should communicate (what, when, why, etc) hcom and what they should/shouldn't do. It is needed for effective collaboration.
- Share large context via markdown files - create md in folder -> hcom send 'everyone read shared-context.md'
- Use structured message passing over free-form chat (reduces hallucination cascading)
- To orchestrate instances yourself, use --append-system-prompt "prioritize messages from <your_hcom_alias>"
- Use system prompts (HCOM_AGENT, --append-system-prompt, etc) unless there's a good reason not to
- For long args or to manage multiple launch profiles: source long-custom-vars.env && hcom 1


## ENVIRONMENT VARIABLES

### HCOM_TAG

#### Different groups on the same project:
HCOM_TAG=backend hcom 3 && HCOM_TAG=frontend hcom 3
Instances use @mention tag for inside group and @mention alias (found via hcom list) or broadcast for outside group

#### Isolated via one central coordinator
Make sure all instances running are launched with: --append-system-prompt "always use @<coordinator_alias> when sending hcom messages"
Coordinator can route messages: instance_a <-> coordinator <-> instance_b
Or coordinator can not route (ie paralell execution): coordinator 1<->many instances

#### Isolate multiple groups:
for label in frontend-team backend-team; do
  HCOM_TAG=$label hcom 2 claude --append-system-prompt "always use @$label [and/or @coordinator_alias]"

Notes:
- Tags are letters, numbers, and hyphens only


### HCOM_AGENT
.claude/agents/*.md are YAML frontmatter files created by user/Claude for use as Task tool subagents.
HCOM can load them as regular instances. You can create them dynamically.

File format:
```markdown
---
model: sonnet
tools: Bash,Write,WebSearch
---
You are a senior code reviewer focusing on...
```

HCOM_AGENT parses the file and merges with Claude args:
--model
--allowedTools
--system-prompt (or --append-system-prompt)

Notes:
- Filename: lowercase letters and hyphens only
- Multiple comma-separated: HCOM_AGENT=reviewer,tester hcom 1 == 2 instances (1 reviewer, 1 tester)


### HCOM_HINTS
Uses: Behavioral guidelines, context reminders, formatting requests, workflow hints


### HCOM_TIMEOUT and HCOM_SUBAGENT_TIMEOUT
After timeout (default 30min, 30s for subagents), instances can't receive messages, marked stale (status=inactive). No downside to longer timeouts (polling <0.1% CPU). Use default or long timeout and `hcom stop {{alias}}` to stop early if needed.

Timeout behavior:
- Normal: terminal stays open, process still running, user must send prompt for instance to re-join hcom connection
- Headless: dies, can only be restarted with --resume <sessionid>
- Subagents: die and all siblings die, their parent must resume. Non-asynchronous (parent waits for completion). Default 30s.

Notes:
- Timer resets on any activity (messages, tool use)
- Stale instances cannot be manually restarted with `hcom start {{alias}}`


### HCOM_TERMINAL
- You cannot use HCOM_TERMINAL=here (Claude can't launch itself, no TTY, needs new terminal)
- Custom must include {{script}} placeholder. Example: HCOM_TERMINAL='open -n -a kitty.app --args bash "{{script}}"' hcom 1
- Headless and task tool subagents ignore HCOM_TERMINAL


### HCOM_CLAUDE_ARGS
Run 'claude --help' for all flags. Syntax: hcom 1 claude [options] [command] [prompt]

Merging with CLI: Per-flag precedence
- Example: HCOM_CLAUDE_ARGS='--model sonnet "hello"'
- Run: hcom 1 claude --model opus
- Result: --model opus "hello" (CLI --model wins, positional "hello" inherited from env)

Precedence:
1. Env var level: config.env HCOM_CLAUDE_ARGS < shell HCOM_CLAUDE_ARGS (overrides the complete string)
2. CLI level: env HCOM_CLAUDE_ARGS < CLI args (overrides per flag individually)
   - Positionals inherited from env if not provided at CLI
   - Empty string "" deletes env positional initial prompt: `hcom 1 claude ""`

Notes:
- Both --system-prompt and --append-system-prompt work in interactive and headless modes (since Claude Code v2.0.14)
- Use --append-system-prompt to add to Claude Code's default behavior
- Use --system-prompt to completely replace the system prompt
- Complex args/quoting? Use: source my-launch.env

#### Configuration Notes

Env var precedence (per variable): HCOM defaults < config.env < shell env vars

Each resolves independently:
- HCOM_TAG in config.env only → config.env wins
- HCOM_TAG in both → shell wins
- Empty (`HCOM_TAG=""`) clears config.env value

Behavior:
- config.env applies to every launch
- Explicitly use all ENV vars in custom.env files and remember to override values inline if needed


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