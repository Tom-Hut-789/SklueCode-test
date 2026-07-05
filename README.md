# SklueCode

SklueCode is a terminal-first AI assistant built in Python. Chapter 01 focuses on a TUI chat experience with streaming model output, multi-turn context, configurable providers, and optional local session persistence.

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
