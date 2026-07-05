from __future__ import annotations

from ..models import SessionRecord


class InMemorySessionStore:
    def __init__(self) -> None:
        self._latest: SessionRecord | None = None

    async def load_latest(self) -> SessionRecord | None:
        return self._latest

    async def save(self, session: SessionRecord) -> None:
        self._latest = session
