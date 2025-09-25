# hcom - Claude Hook Comms

[![PyPI - Version](https://img.shields.io/pypi/v/hcom)](https://pypi.org/project/hcom/)
 [![PyPI - License](https://img.shields.io/pypi/l/hcom)](https://opensource.org/license/MIT) [![Python Version](https://img.shields.io/badge/python-3.7+-blue.svg)](https://python.org)

CLI tool for launching multiple Claude Code terminals with interactive [subagents](https://docs.anthropic.com/en/docs/claude-code/sub-agents), headless persistence, and real-time communication via [hooks](https://docs.anthropic.com/en/docs/claude-code/hooks). Works on Mac, Linux, and Windows with zero dependencies.

![Claude Hook Comms Example](https://raw.githubusercontent.com/aannoo/claude-hook-comms/main/screenshot.jpg)

## 🥦 Usage

**Run without installing** ([uv](https://docs.astral.sh/uv/#installation))
```bash
uvx hcom open 2
```

**Install** (optional)
```bash
pip install hcom
hcom open 2
```

| Commands |  |
|---------|-------------|
| `hcom open [n]` | Launch `n` instances or named agents |
| `hcom watch` | Live dashboard / messaging |
| `hcom clear` | New conversation log |
| `hcom cleanup` | Safely remove hcom hooks, preserving your project settings |


## 🦆 What It Does

`hcom open` adds hooks to the `.claude/settings.local.json` file in the current folder and launches terminals with claude code that remain active, waiting to respond to messages in the shared chat.

**Subagents in their own terminal**
```bash
# Launch subagents from your .claude/agents
hcom open planner code-writer reviewer
```

**Persistent headless instances**
```bash
# Launch one headless instance (default 30min timeout)
hcom open -p
```

**Groups and direct messages**
```bash
hcom open --prefix cool  # Creates cool-hovoa7
hcom open --prefix cool  # Creates cool-homab8
hcom send '@cool hi, you are cool'
hcom send '@homab8 hi, you are cooler'
```

---


<details>
<summary><strong>🦷 Features</strong></summary>

- **Multi-Terminal Launch** - Launch Claude Code subagents in new terminals
- **Background Mode** - Run headless instances without terminal windows
- **Interactive subagents** - Run subagents in their own terminal window
- **Live Dashboard** - Real-time monitoring and messaging
- **Multi-Agent Communication** - Instances talk to each other via shared chat
- **@Mention Targeting** - Send messages to specific instances or teams
- **Session Persistence** - Resume previous conversations automatically
- **Zero Dependencies** - Pure Python stdlib, works everywhere

</details>

<details>
<summary><strong>🥨 All Commands</strong></summary>


| Command | Description |
|---------|-------------|
| `hcom open [n]` | Launch n Claude instances (or named agents) |
| `hcom open -p` | Launch headless process |
| `hcom open --prefix <p> [n]` | Launch with `<p>` prefix (e.g., api-hova7) |
| `hcom open --claude-args "..."` | Pass flags to Claude Code |
| `hcom watch` | Conversation/status dashboard |
| `hcom clear` | Clear and archive conversation |
| `hcom cleanup` | Safely Remove HCOM hooks from current directory while preserving your settings (`--all` for all directories) |
| `hcom kill [name]` | Kill specific instance (--all for all running instances) |

### Automation Commands
| Command | Description |
|---------|-------------|
| `hcom send 'message'` | Send message to all instances |
| `hcom send '@alias msg'` | Send to specific instances alias or prefix |
| `hcom watch --logs` | View message log history (non-interactive) |
| `hcom watch --status` | Show instance status as JSON (non-interactive) |
| `hcom watch --wait [timeout]` | Wait and notify for new messages |

</details>



<details>
<summary><strong>🗿 Examples</strong></summary>

```bash
# Instances can be privately @mentioned by alias or prefix
hcom open --prefix cool  # Creates cool-hovoa7
hcom open --prefix cool  # Creates cool-hovob8
hcom open --prefix notcool # creates notcool-hovoc9

# Send a targeted message in dashboard
@notcool i think you smell good
@cool that other guy is smelly
@hovoa7 im lying about the smelly thing

# Launch 3 headless instances that die after 60 seconds of inactivity
HCOM_WAIT_TIMEOUT="60" hcom open 3 -p
# Manually kill all instance
hcom kill --all

# Launch multiple of the same subagent
hcom open reviewer reviewer reviewer

# Launch agent with specific prompt
HCOM_INITIAL_PROMPT='write tests' hcom open test-writer

# Resume instance (hcom chat will continue)
hcom open --claude-args "--resume session_id"

# Text appended to all messages recieved by instance
HCOM_INSTANCE_HINTS="remember where you came from" hcom open

# Pass multiple Claude flags
hcom open orchestrator --claude-args "--model sonnet --resume session_id"
```

</details>

<details>
<summary><strong>🦖 Configuration</strong></summary>

### Configuration

Settings can be changed two ways:

#### Method 1: Environment variable (temporary, per-command/instance)


```bash
HCOM_INSTANCE_HINTS="always update chat with progress" hcom open nice-subagent-but-not-great-with-updates
```

#### Method 2: Config file (persistent, affects all instances)

### Config File Location

`~/.hcom/config.json`

| Setting | Default | Environment Variable | Description |
|---------|---------|---------------------|-------------|
| `wait_timeout` | 1800 | `HCOM_WAIT_TIMEOUT` | How long instances wait for messages (seconds) |
| `max_message_size` | 1048576 | `HCOM_MAX_MESSAGE_SIZE` | Maximum message length (1MB) |
| `max_messages_per_delivery` | 50 | `HCOM_MAX_MESSAGES_PER_DELIVERY` | Messages delivered per batch |
| `sender_name` | "bigboss" | `HCOM_SENDER_NAME` | Your name in chat |
| `sender_emoji` | "🐳" | `HCOM_SENDER_EMOJI` | Your emoji icon |
| `initial_prompt` | "Say hi in chat" | `HCOM_INITIAL_PROMPT` | What new instances are told to do |
| `first_use_text` | "Essential, concise messages only" | `HCOM_FIRST_USE_TEXT` | Welcome message for instances |
| `terminal_mode` | "new_window" | `HCOM_TERMINAL_MODE` | How to launch terminals ("new_window", "same_terminal", "show_commands") |
| `terminal_command` | null | `HCOM_TERMINAL_COMMAND` | Custom terminal command (see Terminal Options) |
| `cli_hints` | "" | `HCOM_CLI_HINTS` | Extra text added to CLI outputs |
| `instance_hints` | "" | `HCOM_INSTANCE_HINTS` | Extra text added to instance messages |
| `auto_watch` | true | `HCOM_AUTO_WATCH` | Auto-launch watch dashboard after open |
| `env_overrides` | {} | - | Additional environment variables for Claude Code |

### Examples

```bash
# Change your name for one command
HCOM_SENDER_NAME="coolguy" hcom send "LGTM!"

# Make instances timeout after 60 seconds instead of 30 minutes
HCOM_WAIT_TIMEOUT=60 hcom open 3

# Custom welcome message
HCOM_FIRST_USE_TEXT="Debug session for issue #123" hcom open 2

# Bigger delivery batches
HCOM_MAX_MESSAGES_PER_DELIVERY=100 hcom watch --logs
```

**Windows PowerShell**:
```powershell
# Set environment variables in PowerShell
$env:HCOM_TERMINAL_MODE="same_terminal"; hcom open agent-name
$env:HCOM_WAIT_TIMEOUT="60"; hcom open 3
$env:HCOM_INITIAL_PROMPT="go home buddy!"; hcom open
```

### Status Indicators
- ◉ **thinking** (cyan) - Processing input
- ▷ **responding** (green) - Generating text response  
- ▶ **executing** (green) - Running tools
- ◉ **waiting** (blue) - Waiting for messages
- ■ **blocked** (yellow) - Permission blocked
- ○ **inactive** (red) - Timed out/dead
- **(bg)** suffix - Instance running in background headless mode

</details>

<details>
<summary><strong>🎲 How It Works</strong></summary>

### Hooks!

hcom adds hooks to your project directory's `.claude/settings.local.json`:

1. **Sending**: Claude agents use `echo "HCOM_SEND:message"` internally (you use `hcom send` from terminal or dashboard)
2. **Receiving**: Other Claudes get notified via Stop hook
3. **Waiting**: Stop hook keeps Claude in a waiting state for new messages

- **Identity**: Each instance gets a unique name based on session ID (e.g., "hovoa7")
- **Persistence**: Names persist across `--resume` maintaining conversation context
- **Status Detection**: Notification hook tracks permission requests and activity
- **Agents**: When you run `hcom open researcher`, it loads an interactive Claude session with a system prompt from `.claude/agents/researcher.md` (local) or `~/.claude/agents/researcher.md` (global). Specified `model:` and `tools:` are carried over.

### Architecture
- **Single conversation** - All instances share one global conversation
- **Opt-in participation** - Only Claude Code instances launched with `hcom open` join the chat
- **@-mention filtering** - Target messages to specific instances or teams

### File Structure
```
~/.hcom/                             
├── hcom.log       # Conversation log
├── instances/     # Instance tracking
├── logs/          # Background process logs
├── config.json    # Configuration
└── archive/       # Archived sessions

your-project/  
└── .claude/
    └── settings.local.json  # hcom hooks
```

</details>


<details>
<summary><strong>🥔 Terminal Options</strong></summary>

### Terminal Mode

Configure terminal behavior in `~/.hcom/config.json`:
- `"terminal_mode": "new_window"` - Opens new terminal window(s) (default)
- `"terminal_mode": "same_terminal"` - Opens in current terminal
- `"terminal_mode": "show_commands"` - Prints commands without executing

### Default Terminals

- **macOS**: Terminal.app
- **Linux**: gnome-terminal, konsole, or xterm
- **Windows & WSL**: Windows Terminal / Git Bash

### Running in Current Terminal temporarily
```bash
# For single instances
HCOM_TERMINAL_MODE=same_terminal hcom open
```

### Custom Terminal Examples

Configure `terminal_command` in `~/.hcom/config.json` (permanent) or environment variables (temporary).

#### How to use this

The `{script}` placeholder is replaced by HCOM with the path to a temporary bash script that launches Claude Code.

Your custom command just needs to:
1. Accept `{script}` as a placeholder that will be replaced with a script path
2. Execute that script with bash

Example template: `your_terminal_command --execute "bash {script}"`

### iTerm2
```json
"terminal_command": "osascript -e 'tell app \"iTerm\" to tell (create window with default profile) to tell current session to write text \"{script}\"'"
```

### WezTerm
Windows:
```json
  "terminal_command": "wezterm start -- bash {script}"
```
Or open tabs from within WezTerm:
```json
  "terminal_command": "wezterm cli spawn -- bash {script}"
```
macOS/Linux:
```json
  "terminal_command": "wezterm start -- bash {script}"
```

### Wave Terminal
Windows. From within Wave Terminal:
```json
  "terminal_command": "wsh run -- bash {script}"
```

### Alacritty
macOS:
```json
  "terminal_command": "open -n -a Alacritty.app --args -e bash {script}"
```
Linux:
```json
  "terminal_command": "alacritty -e bash {script}"
```

### Kitty
macOS:
```json
  "terminal_command": "open -n -a kitty.app --args bash {script}"
```
Linux:
```json
  "terminal_command": "kitty bash {script}"
```

### Termux (Android)
```json
  "terminal_command": "am startservice --user 0 -n com.termux/com.termux.app.RunCommandService -a com.termux.RUN_COMMAND --es com.termux.RUN_COMMAND_PATH {script} --ez com.termux.RUN_COMMAND_BACKGROUND false"
```
Note: Requires `allow-external-apps=true` in `~/.termux/termux.properties`

### tmux
```json
  "terminal_command": "tmux new-window -n hcom {script}"
```
Then from a terminal:
```bash
# Run hcom open directly in new session
tmux new-session 'hcom open 3'
```
Or once off:
```bash
# Start tmux with split panes and 3 claude instances in hcom chat
HCOM_TERMINAL_COMMAND="tmux split-window -h {script}" hcom open 3
```


</details>


<details>
<summary><strong>🦆 Remove</strong></summary>


### Archive Conversation / Start New
```bash
hcom clear
```

### Kill Running Instances
```bash
# Kill specific instance
hcom kill hovoa7

# Kill all instances
hcom kill --all
```

### Remove HCOM hooks from current directory
```bash
hcom cleanup
```

### Remove HCOM hooks from all directories
```bash
hcom cleanup --all
```

### Remove hcom Completely
1. Remove hcom: `rm /usr/local/bin/hcom` (or wherever you installed hcom)
2. Remove all data: `rm -rf ~/.hcom`

</details>

## 🦐 Requirements

- Python 3.7+
- [Claude Code](https://claude.ai/code)


## 🌮 License

- MIT License

---