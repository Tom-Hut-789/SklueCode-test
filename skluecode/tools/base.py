from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from .confirmation import ConfirmationGateway


class ToolError(RuntimeError):
    """Base class for tool failures caught by the scheduler."""


class ToolSchemaError(ToolError):
    """Raised when tool arguments fail the declared schema."""


class ToolNotFoundError(ToolError):
    """Raised when a tool name is not registered."""


class ToolPermissionError(ToolError):
    """Raised when sandbox or user confirmation denies execution."""


class ToolTargetError(ToolError):
    """Raised when the target of a tool is invalid (missing file, bad pattern, etc.)."""


class ToolExecutionError(ToolError):
    """Catch-all for unexpected runtime errors during tool execution."""


class ToolTimeoutError(ToolError):
    """Raised by the scheduler when a tool exceeds its timeout budget."""


@dataclass(slots=True)
class NeutralToolDef:
    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ToolResultContent:
    text: str
    truncated: bool = False
    target: str | None = None
    exit_code: int | None = None


@dataclass(slots=True)
class ToolExecutionContext:
    workspace_root: Path
    request_confirmation: "ConfirmationGateway"


class Tool(Protocol):
    name: str
    description: str
    parameters_json_schema: dict[str, Any]
    default_timeout_seconds: float

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolResultContent:
        ...
