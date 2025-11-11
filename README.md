# hcom — Claude Hook Comms

[![PyPI - Version](https://img.shields.io/pypi/v/hcom)](https://pypi.org/project/hcom/)
 [![PyPI - License](https://img.shields.io/pypi/l/hcom)](https://opensource.org/license/MIT) [![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org) [![DeepWiki](https://img.shields.io/badge/DeepWiki-aannoo%2Fclaude--hook--comms-blue.svg?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACwAAAAyCAYAAAAnWDnqAAAAAXNSR0IArs4c6QAAA05JREFUaEPtmUtyEzEQhtWTQyQLHNak2AB7ZnyXZMEjXMGeK/AIi+QuHrMnbChYY7MIh8g01fJoopFb0uhhEqqcbWTp06/uv1saEDv4O3n3dV60RfP947Mm9/SQc0ICFQgzfc4CYZoTPAswgSJCCUJUnAAoRHOAUOcATwbmVLWdGoH//PB8mnKqScAhsD0kYP3j/Yt5LPQe2KvcXmGvRHcDnpxfL2zOYJ1mFwrryWTz0advv1Ut4CJgf5uhDuDj5eUcAUoahrdY/56ebRWeraTjMt/00Sh3UDtjgHtQNHwcRGOC98BJEAEymycmYcWwOprTgcB6VZ5JK5TAJ+fXGLBm3FDAmn6oPPjR4rKCAoJCal2eAiQp2x0vxTPB3ALO2CRkwmDy5WohzBDwSEFKRwPbknEggCPB/imwrycgxX2NzoMCHhPkDwqYMr9tRcP5qNrMZHkVnOjRMWwLCcr8ohBVb1OMjxLwGCvjTikrsBOiA6fNyCrm8V1rP93iVPpwaE+gO0SsWmPiXB+jikdf6SizrT5qKasx5j8ABbHpFTx+vFXp9EnYQmLx02h1QTTrl6eDqxLnGjporxl3NL3agEvXdT0WmEost648sQOYAeJS9Q7bfUVoMGnjo4AZdUMQku50McDcMWcBPvr0SzbTAFDfvJqwLzgxwATnCgnp4wDl6Aa+Ax283gghmj+vj7feE2KBBRMW3FzOpLOADl0Isb5587h/U4gGvkt5v60Z1VLG8BhYjbzRwyQZemwAd6cCR5/XFWLYZRIMpX39AR0tjaGGiGzLVyhse5C9RKC6ai42ppWPKiBagOvaYk8lO7DajerabOZP46Lby5wKjw1HCRx7p9sVMOWGzb/vA1hwiWc6jm3MvQDTogQkiqIhJV0nBQBTU+3okKCFDy9WwferkHjtxib7t3xIUQtHxnIwtx4mpg26/HfwVNVDb4oI9RHmx5WGelRVlrtiw43zboCLaxv46AZeB3IlTkwouebTr1y2NjSpHz68WNFjHvupy3q8TFn3Hos2IAk4Ju5dCo8B3wP7VPr/FGaKiG+T+v+TQqIrOqMTL1VdWV1DdmcbO8KXBz6esmYWYKPwDL5b5FA1a0hwapHiom0r/cKaoqr+27/XcrS5UwSMbQAAAABJRU5ErkJggg==)](https://deepwiki.com/aannoo/claude-hook-comms)

Launch multiple Claude Code instances (terminal, headless, or subagents) that communicate together in real time via hooks.

![Demo](https://raw.githubusercontent.com/aannoo/claude-hook-comms/main/screencapture.gif)

## Start

#### Run without installing
```bash
uvx hcom 2
```

#### Run with installing
```bash
pip install hcom

hcom                      # UI

claude 'run hcom start'   # toggle hcom for any claude code
```


## What it does

Adds hooks that enable instant messaging between claude code instances. Launch multiple terminals/headless/subagents that remain active, waiting to respond. Normal `claude` remains unaffected by hcom, but can opt-in (`hcom start`) or opt-out (`hcom stop`) at runtime. Safely remove hooks with `hcom reset`. Works on Mac, Linux, Windows, Android.


## Commands

| Command | Description
|---------|-------------|
| `hcom` | TUI dashboard |
| `hcom <n>` | Launch `n` instances |
| `hcom start` | Enable participation |
| `hcom stop` | Disable participation |


## Features

#### communicate with task tool subagents
```bash
claude 'use 3x task tool with task: say hi in hcom chat'
# Each subagent gets unique identity and can communicate with siblings
# Parent resumes with full conversation history
```

#### persistent headless instances
```bash
hcom 1 claude -p    # launch with default 30min timeout
hcom                # See what it's doing from dashboard
hcom stop           # Let it die earlier than timeout
```

#### .claude/agents in interactive claude window
```bash
HCOM_AGENT=code-writer hcom 1
HCOM_AGENT=reviewer hcom 1
```

#### @-mention: groups and direct
```bash
HCOM_TAG=cooltag hcom 2
hcom send '@cooltag hi, you both are cool'
hcom send '@john you are cooler'
```

#### toggle inside claude code
```bash
claude              # Normal Claude Code
'run hcom start'    # Opt-in to receive messages
'run hcom stop'     # Opt-out, continue as normal claude code
```

#### async push notifications from anywhere
```bash
# Any process can message claude instances with custom identity
hcom send --from background-worker 'i finished, now you go do stuff'
```


---

<details>
<summary><strong>All commands</strong></summary>


```bash
Usage: hcom                           # TUI dashboard
       [ENV_VARS] hcom <COUNT> [claude <ARGS>...]
       hcom watch [--type TYPE] [--instance NAME] [--last N] [--wait SEC]
       hcom list [--json] [--verbose]
       hcom send "message"
       hcom stop [alias|all]
       hcom start [alias]
       hcom reset [logs|hooks|config]

Launch Examples:
  hcom 3             open 3 terminals with claude connected to hcom
  hcom 3 claude -p                                       + headless
  HCOM_TAG=api hcom 3 claude -p               + @-mention group tag
  claude 'run hcom start'        claude code with prompt also works

Commands:
  watch               Query recent events (JSON per line)
    --type TYPE       Filter by event type (message, status)
    --instance ALIAS  Filter by instance
    --last N          Limit to last N events (default: 20)
    --wait SEC        Block until a matching event arrives

  list                Show instance status/metadata
    --json            Emit JSON (one instance per line)
    --verbose         Include additional metadata

  send "msg"          Send message to all instances
  send "@alias msg"   Send to specific instance/group
    --from <name>     Custom external identity

  stop                Stop current instance (from inside Claude)
  stop <alias>        Stop specific instance
  stop all            Stop all instances

  start               Start current instance (from inside Claude)
  start <alias>       Start specific instance

  reset               Stop all + archive logs + remove hooks + clear config
  reset logs          Clear + archive conversation
  reset hooks         Safely remove hcom hooks from claude settings.json
  reset config        Clear + archive config.env

Environment Variables:
  HCOM_TAG=name               Group tag (creates name-* instances)
  HCOM_AGENT=type             Agent from .claude/agents/ (comma-separated for multiple)
  HCOM_TERMINAL=mode          Terminal: new|here|print|"custom {script}"
  HCOM_HINTS=text             Text appended to all messages received by instance
  HCOM_TIMEOUT=secs           Time until disconnected from hcom chat (default: 1800s / 30m)
  HCOM_SUBAGENT_TIMEOUT=secs  Subagent idle timeout (default: 30s)
  HCOM_CLAUDE_ARGS=args       Claude CLI defaults (e.g., '-p --model opus "hello!"')

  ANTHROPIC_MODEL=opus # Any env var passed through to Claude Code

  Persist Env Vars in `~/.hcom/config.env`
```

---

### Requirements

- Python 3.10+
- Claude Code
 
  

</details>

---

<details>
<summary><strong> Terminal Options</strong></summary>

### Default Terminals

- **macOS**: Terminal.app
- **Linux**: gnome-terminal, konsole, or xterm
- **Windows (native) & WSL**: Windows Terminal / Git Bash
- **Android**: Termux

### Terminal Mode

- `HCOM_TERMINAL=new` - New terminal windows (default)
- `HCOM_TERMINAL=here` - Current terminal window
- `HCOM_TERMINAL="open -a iTerm {script}"` - Custom terminal


### Custom Terminal

Your custom command just needs to:
1. Accept `{script}` as a placeholder that will be replaced with a script path
2. Execute that script with bash

### Custom Terminal Examples

##### [ttab](https://github.com/mklement0/ttab) (new tab instead of new window in Terminal.app)
```bash
HCOM_TERMINAL="ttab {script}"
```

##### [wttab](https://github.com/lalilaloe/wttab) (new tab in Windows Terminal)
```bash
HCOM_TERMINAL="wttab {script}"
```

##### More
```bash
# Wave Terminal Mac/Linux/Windows. From within Wave Terminal:
HCOM_TERMINAL="wsh run -- bash {script}"

# Alacritty macOS:
HCOM_TERMINAL="open -n -a Alacritty.app --args -e bash {script}"

# Alacritty Linux:
HCOM_TERMINAL="alacritty -e bash {script}"

# Kitty macOS:
HCOM_TERMINAL="open -n -a kitty.app --args {script}"

# Kitty Linux
HCOM_TERMINAL="kitty {script}"

# tmux with split panes and 3 claude instances in hcom chat
HCOM_TERMINAL="tmux split-window -h {script}" hcom 3

# WezTerm Linux/Windows
HCOM_TERMINAL="wezterm start -- bash {script}"

# Tabs from within WezTerm
HCOM_TERMINAL="wezterm cli spawn -- bash {script}"

# WezTerm macOS:
HCOM_TERMINAL="open -n -a WezTerm.app --args start -- bash {script}"

# Tabs from within WezTerm macOS
HCOM_TERMINAL="/Applications/WezTerm.app/Contents/MacOS/wezterm cli spawn -- bash {script}"
```

#### Android (Termux)

```bash
#1. Install:
    Termux from F-Droid (not Google Play)
#2. Setup:
   pkg install python nodejs
   npm install -g @anthropic-ai/claude-cli
   pip install hcom
#3. Enable:
   echo "allow-external-apps=true" >> ~/.termux/termux.properties
   termux-reload-settings
#4. Enable: 
    "Display over other apps" permission for visible terminals
#5. Run: 
    `hcom 2`
```

</details>

---

### License

MIT

