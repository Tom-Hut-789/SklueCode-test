SklueCode is a terminal-first AI assistant built in Python. This focuses on a TUI chat experience with streaming model output, multi-turn context, configurable providers, and optional local session persistence.

# ✨ Key Features
- Dual Provider Support: Compatible with both OpenAI and Anthropic Claude API protocols, allowing free switching through configuration
- Streaming Output: Shows the model generation process in real-time, no need to wait for the full response
- Multi-turn Conversation: Full context memory, supports continuous dialogue
- Session Persistence: Optional temporary sessions or local persistent sessions, recent conversations can be restored after restart
- Claude Extended Thinking: Supports displaying Claude's thought process
- Clear Abstraction: Unified Provider interface, making it easy to extend with new model backends later

# 🏗️ Tech Stack
- Terminal Interface: Textual - modern TUI framework
- HTTP Client: httpx - asynchronous HTTP requests
- Configuration Parsing: PyYAML

# 🎯 Design Highlights
- Low-coupling Architecture: Provider layer, session storage layer, and TUI layer are completely decoupled
- Extensible Design: Adding a new Provider only requires implementing the unified interface, no need to modify the main flow
- Friendly Error Handling: Clear prompts for configuration errors, network issues, etc., so the program won't crash directly

# Operation Guide
## Requirements
- Python 3.11+
- A valid OpenAI or Anthropic API key

## Setup
1. Create and activate a virtual environment to prevent the global Python environment from getting messed up:
```bash
python -m venv .venv
```

2. Install dependencies in a virtual environment:
```bash
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```
Verify:
```bash
 .\.venv\Scripts\python.exe -m pip list
```

3. Copy the example config and fill in your provider details:
```bash
copy config\config.example.yaml config\config.yaml
```

4. Edit `config/config.yaml` and set:
- `protocol`
- `model`
- `base_url`
- `api_key`

## Run

Set the source directory on `PYTHONPATH` and launch the app:
(1)No need to manually activate the virtual environment, just run it with the installed uv tool. Because uv run automatically loads the virtual environment.
```bash
$env:PYTHONPATH = "src"
uv run python -m skluecode.main
```
(2)Run it directly using the full Python path from the virtual environment
```bash
$env:PYTHONPATH='src'
.\.venv\Scripts\python.exe -m skluecode.main
```

Inside the app:
- Type a message and press `Enter` to send it.
- Type `/quit` or `/exit` to close the program.
- Use `session_mode: persistent` to restore the latest saved conversation.

## Config Notes

- `protocol`: `openai` or `anthropic`
- `session_mode`: `ephemeral` or `persistent`
- `enable_extended_thinking`: only used for Anthropic requests
- `storage_path`: directory used to save the latest session JSON file
