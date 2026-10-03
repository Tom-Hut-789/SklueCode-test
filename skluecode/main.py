from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .agent import AgentRunner
from .config import ConfigError, load_config
from .controller import ChatController
from .models import AppConfig, ProtocolType, SessionMode
from .providers import AnthropicProvider, OpenAIProvider
from .providers.base import Provider
from .session_store import FileSessionStore, InMemorySessionStore
from .session_store.base import SessionStore
from .tools.confirmation import DelegatingConfirmationBridge
from .tools.impl.edit_file import EditFileTool
from .tools.impl.find_files import FindFilesTool
from .tools.impl.read_file import ReadFileTool
from .tools.impl.run_command import RunCommandTool
from .tools.impl.search_code import SearchCodeTool
from .tools.impl.write_file import WriteFileTool
from .tools.registry import ToolRegistry
from .tools.sandbox import WorkspaceSandbox
from .tools.scheduler import ToolScheduler
from .tui import ChatApp


def main() -> int:
    args = _parse_args()
    root_dir = Path(__file__).resolve().parents[2]
    config_path = _resolve_path(root_dir, args.config)

    try:
        config = load_config(config_path)
    except ConfigError as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        return 1

    provider = build_provider(config)
    session_store = build_session_store(root_dir, config)
    registry, scheduler = build_tool_system(root_dir)
    agent = AgentRunner(provider, registry, scheduler, config, max_iterations=10)
    controller = ChatController(
        config=config,
        session_store=session_store,
        agent=agent,
    )
    app = ChatApp(controller)
    app.run()
    return 0


def build_provider(config: AppConfig) -> Provider:
    if config.protocol == ProtocolType.OPENAI:
        return OpenAIProvider()
    if config.protocol == ProtocolType.ANTHROPIC:
        return AnthropicProvider()
    raise ConfigError(f"Unsupported protocol: {config.protocol}")


def build_session_store(root_dir: Path, config: AppConfig) -> SessionStore:
    if config.session_mode == SessionMode.PERSISTENT:
        storage_path = _resolve_path(root_dir, config.storage_path or "data/sessions")
        return FileSessionStore(storage_path)
    return InMemorySessionStore()


def build_tool_system(root_dir: Path) -> tuple[ToolRegistry, ToolScheduler]:
    registry = ToolRegistry()
    registry.register(ReadFileTool())
    registry.register(WriteFileTool())
    registry.register(EditFileTool())
    registry.register(RunCommandTool())
    registry.register(FindFilesTool())
    registry.register(SearchCodeTool())

    sandbox = WorkspaceSandbox(root_dir)
    bridge = DelegatingConfirmationBridge()
    scheduler = ToolScheduler(registry, sandbox, bridge)
    return registry, scheduler


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the SklueCode TUI chat client.")
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to the YAML config file. Defaults to config/config.yaml.",
    )
    return parser.parse_args()


def _resolve_path(root_dir: Path, raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return root_dir / path


if __name__ == "__main__":
    raise SystemExit(main())
