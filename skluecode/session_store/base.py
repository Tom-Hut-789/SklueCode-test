from __future__ import annotations

from typing import Protocol

from ..models import SessionRecord


class SessionStoreError(RuntimeError):
    """Raised when a session store cannot persist or restore data."""


class SessionStore(Protocol):
    async def load_latest(self) -> SessionRecord | None:
        ...

    async def save(self, session: SessionRecord) -> None:
        ...
