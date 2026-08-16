from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Awaitable, Callable, Protocol


class ConfirmationType(StrEnum):
    PATH_OUTSIDE_WORKSPACE = "path_outside_workspace"
    DANGEROUS_COMMAND = "dangerous_command"
    OVERWRITE_EXISTING_FILE = "overwrite_existing_file"


@dataclass(slots=True)
class ConfirmationRequest:
    type: ConfirmationType
    title: str
    description: str
    target: str
    detail: str | None = None


@dataclass(slots=True)
class ConfirmationResponse:
    approved: bool
    reason: str | None = None


class ConfirmationGateway(Protocol):
    async def confirm(self, request: ConfirmationRequest) -> ConfirmationResponse:
        ...


class DelegatingConfirmationBridge:
    """Default ConfirmationGateway implementation used at bootstrap time.

    The TUI is instantiated after the scheduler, so we use a tiny bridge that
    forwards confirm() calls once the real callback is bound by the main app.
    """

    def __init__(self) -> None:
        self.callback: Callable[[ConfirmationRequest], Awaitable[ConfirmationResponse]] | None = None

    async def confirm(self, request: ConfirmationRequest) -> ConfirmationResponse:
        if self.callback is None:
            return ConfirmationResponse(
                approved=False,
                reason="Confirmation gateway not bound; refusing the operation by default.",
            )
        return await self.callback(request)
